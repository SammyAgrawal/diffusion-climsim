# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

Credit for most of this boilerplate belongs to the HuggingFace Diffusers library. Many of these are blocks are directly from the Diffusers Library
Others have been modified to fit specific needs of this project or are because of bugs in the Diffusers Library.

"""

import math
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import rearrange_dims, get_activation, weight_init, Linear


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [-0.01171875, -0.03515625, 0.11328125, 0.43359375, 0.43359375, 0.11328125, -0.03515625, -0.01171875],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}

def safe_pad(x, pad, pad_mode, chunksize=24576):
    outs = []
    for chunk in x.split(chunksize, dim=0):
        outs.append(F.pad(chunk, pad, pad_mode))
    return torch.cat(outs, dim=0)

class GroupNorm(torch.nn.Module):
    """
    Group Normalization (GN) divides the channels of the input tensor into groups and
    normalizes the features within each group independently. It does not require the
    batch size as in Batch Normalization, making itsuitable for batch sizes of any size
    or even for batch-free scenarios.
    -----
    If `num_channels` is not divisible by `num_groups`, the actual number of groups
    might be adjusted to satisfy the `min_channels_per_group` condition.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 2,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(dtype=x.dtype, device=x.device),
            bias=self.bias.to(dtype=x.dtype, device=x.device),
            eps=self.eps,
        )
        return x

class ChannelAffine(nn.Module):
    def __init__(self, num_channels, cross_channel=False, bias=False, init_mode="kaiming_normal", init_weight=1, init_bias=0):
        super().__init__()
        if cross_channel:
            self.affine = Linear(num_channels, num_channels, bias, init_mode, init_weight, init_bias)
        else:
            self.weight = nn.Parameter(init_weight * torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels) + init_bias) if bias else None
            self.affine = None

    def forward(self, x):
        # x: [B, C, L]
        if self.affine is not None:
            # mix channels across C
            x = self.affine(x.transpose(1, 2))  # [B, L, C]
            return x.transpose(1, 2)      # [B, C, L]
        else:
            x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1) if self.bias else x * self.weight.view(1, -1, 1)
            return x

class Conv1d(torch.nn.Module):
    """
    A custom 1D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations. 
    kernel : int
        Size of the convolving kernel.
    up : bool, optional
        Whether to perform up-sampling. By default False.
    down : bool, optional
        Whether to perform down-sampling. By default False.
    resample_filter : List[int], optional
        Filter to be used for resampling. By default [1, 1]. If up or down, always resample filter
    fused_resample : bool, optional
        If True, performs fused up-sampling and convolution or fused down-sampling
        and convolution. By default False.
    """
    def __init__(
        self, in_channels: int, out_channels: int, kernel: int, bias: bool = True, stride=1, padding=None, 
        up: bool = False, down: bool = False, resample_filter: Optional[List[int]] = None, fused_resample: bool = False,
        init_mode: str = "kaiming_uniform", init_weight: float = 1.0, init_bias: float = 0.0,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding if padding is not None else kernel // 2
        resample_filter = resample_filter if resample_filter is not None else [1, 1] 
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel, fan_out=out_channels * kernel)
        self.weight = ( torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel], **init_kwargs) * init_weight
            ) if kernel else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias else None
        )
        f = torch.tensor(resample_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(1) / sum(resample_filter)
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x, **kwargs):
        assert x.shape[1] == self.in_channels
        w = self.weight.to(dtype=x.dtype, device=x.device) if self.weight is not None else None
        f = self.resample_filter

        w_pad = self.padding if w is not None else 0
        f_pad = (f.size(-1) - 1) // 2 if f is not None else 0

        fr = f.repeat(self.in_channels, 1, 1) if f is not None else None
        padding = max(f_pad - w_pad, 0) if f is not None and self.up and self.fused_resample else f_pad
        if f is not None and self.up:
            x = F.conv_transpose1d(x, fr*2, groups=self.in_channels, stride=2, padding=padding)
        elif f is not None and self.down and not self.fused_resample:
            x = F.conv1d(x, fr, groups=self.in_channels, stride=2, padding=f_pad)

        if w is not None:
            padding = (2*self.down - 1) * f_pad if (self.fused_resample and (self.up or self.down)) else 0
            x = F.conv1d(x, w, stride=self.stride, padding=max(w_pad + padding, 0))

        if f is not None and self.down and self.fused_resample:
            x = F.conv1d(x, f.repeat(self.out_channels, 1, 1), groups=self.out_channels, stride=2)

        if self.bias is not None:
            x = x.add_(self.bias.to(dtype=x.dtype, device=x.device).reshape(1, -1, 1))
        return x

class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        n_groups (`int`, default `8`): Number of groups to separate the channels into.
        activation (`str`, defaults to `mish`): Name of the activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        n_groups: int = 8,
        activation: str = "mish",
    ):
        super().__init__()

        self.conv1d = Conv1d(inp_channels, out_channels, kernel_size)
        self.group_norm = GroupNorm(out_channels, n_groups)
        self.act_fn = get_activation(activation)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        intermediate_repr = self.conv1d(inputs)
        intermediate_repr = self.group_norm(rearrange_dims(intermediate_repr))
        output = self.act_fn(rearrange_dims(intermediate_repr))
        return output

class Downsample1d(nn.Module):
    def __init__(self, kernel: str = "linear", pad_mode: str = "constant"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = safe_pad(hidden_states, (self.pad,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros([hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]])
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        weight[indices, indices] = kernel
        return F.conv1d(hidden_states, weight, stride=2)

class Upsample1d(nn.Module):
    def __init__(self, kernel: str = "linear", pad_mode: str = "constant"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = safe_pad(hidden_states, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros([hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]])
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        kernel = self.kernel.to(weight)[None, :].expand(hidden_states.shape[1], -1)
        weight[indices, indices] = kernel
        return F.conv_transpose1d(hidden_states, weight, stride=2, padding=self.pad * 2 + 1)

class ResidualTemporalBlock1D(nn.Module):
    """
    Residual 1D block with temporal convolutions.

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        embed_dim (`int`): Embedding dimension.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        activation (`str`, defaults `mish`): It is possible to choose the right activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        activation: str = "mish",
    ):
        super().__init__()
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)

        self.time_emb_act = get_activation(activation)
        self.time_emb = Linear(embed_dim, out_channels)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb(self.time_emb_act(t)) # Linear projection from embed_dim to channels
        out = self.conv_in(inputs) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(inputs)


class ValueFunctionMidBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.res1 = ResidualTemporalBlock1D(in_channels, in_channels // 2, embed_dim=embed_dim)
        self.down1 = Conv1d(out_channels // 2, out_channels // 2, kernel=3, bias=False, stride=2, padding=1)
        self.res2 = ResidualTemporalBlock1D(in_channels // 2, in_channels // 4, embed_dim=embed_dim)
        self.down2 = Conv1d(out_channels // 4, out_channels // 4, kernel=3, bias=False, stride=2, padding=1)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.res1(x, temb)
        x = self.down1(x)
        x = self.res2(x, temb)
        x = self.down2(x)
        return x

class MidResTemporalBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        add_downsample: bool = False,
        add_upsample: bool = False,
        non_linearity: Optional[str] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_downsample = add_downsample

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=embed_dim))

        self.resnets = nn.ModuleList(resnets)

        if non_linearity is None:
            self.nonlinearity = None
        else:
            self.nonlinearity = get_activation(non_linearity)

        self.upsample = None
        if add_upsample:
            self.upsample = Conv1d(out_channels, out_channels, 3, bias=False)

        self.downsample = None
        if add_downsample:
            self.downsample = Conv1d(out_channels, out_channels, kernel=3, bias=False, stride=2, padding=1)

        if self.upsample and self.downsample:
            raise ValueError("Block cannot downsample and upsample")

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.upsample:
            hidden_states = self.upsample(hidden_states)
        if self.downsample:
            hidden_states = self.downsample(hidden_states)

        return hidden_states

class OutConv1DBlock(nn.Module):
    def __init__(self, num_groups_out: int, out_channels: int, embed_dim: int, act_fn: str):
        super().__init__()
        self.final_conv1d_1 = nn.Conv1d(embed_dim, embed_dim, 5, padding=2)
        self.final_conv1d_gn = GroupNorm(embed_dim, num_groups_out)
        self.final_conv1d_act = get_activation(act_fn)
        self.final_conv1d_2 = nn.Conv1d(embed_dim, out_channels, 1)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.final_conv1d_1(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_gn(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_act(hidden_states)
        hidden_states = self.final_conv1d_2(hidden_states)
        return hidden_states

class OutValueFunctionBlock(nn.Module):
    def __init__(self, fc_dim: int, embed_dim: int, act_fn: str = "mish"):
        super().__init__()
        self.final_block = nn.ModuleList(
            [
                Linear(fc_dim + embed_dim, fc_dim // 2),
                get_activation(act_fn),
                Linear(fc_dim // 2, 1),
            ]
        )

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        hidden_states = torch.cat((hidden_states, temb), dim=-1)
        for layer in self.final_block:
            hidden_states = layer(hidden_states)

        return hidden_states

class SelfAttention1d(nn.Module):
    def __init__(self, in_channels: int, n_head: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        self.channels = in_channels
        self.group_norm = GroupNorm(in_channels, 1, min_channels_per_group=1)
        self.num_heads = max(n_head, 1)

        self.query = Linear(self.channels, self.channels)
        self.key = Linear(self.channels, self.channels)
        self.value = Linear(self.channels, self.channels)

        self.proj_attn = Linear(self.channels, self.channels, bias=True)

        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        batch, channel_dim, seq = hidden_states.shape

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        scale = 1 / math.sqrt(math.sqrt(key_states.shape[-1]))

        attention_scores = torch.matmul(query_states * scale, key_states.transpose(-1, -2) * scale)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # compute attention output
        hidden_states = torch.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual

        return output

class AttentionOp(torch.autograd.Function):
    """
    Attention weight computation, i.e., softmax(Q^T * K).
    Performs all computation using FP32, but uses the original datatype for
    inputs/outputs/gradients to conserve memory.
    """

    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(dtype=torch.float32, device=q.device),
                (k / (k.shape[1]**0.5)).to(dtype=torch.float32, device=k.device),
            )
            .softmax(dim=2)
            .to(dtype=q.dtype, device=q.device)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(dtype=torch.float32, device=dw.device),
            output=w.to(dtype=torch.float32, device=w.device),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(dtype=torch.float32, device=k.device), db).to(
            dtype=q.dtype, device=q.device
        ) / (k.shape[1]**0.5)
        dk = torch.einsum("ncq,nqk->nck", q.to(dtype=torch.float32, device=q.device), db).to(
            dtype=k.dtype, device=k.device
        ) / (k.shape[1]**0.5)
        return dq, dk

class ScriptableAttentionOp(torch.nn.Module):
    def __init__(self):
        super(ScriptableAttentionOp, self).__init__()

    def forward(self, q, k):
        scale_factor = k.shape[1] ** 0.5
        k_scaled = k / scale_factor
        w = torch.einsum("ncq,nck->nqk", q.float(), k_scaled.float()).softmax(dim=2)
        return w.to(dtype=q.dtype)


class ResConvBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, time_emb_dim: Optional[int] = None, is_last: bool = False):
        super().__init__()
        self.is_last = is_last
        self.has_conv_skip = in_channels != out_channels # need to skip if in != out to sync with residual connection

        if self.has_conv_skip:
            self.conv_skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.conv_1 = nn.Conv1d(in_channels, mid_channels, 5, padding=2)
        self.group_norm_1 = GroupNorm(mid_channels, 1, min_channels_per_group=1)
        self.gelu_1 = nn.GELU()
        self.conv_2 = nn.Conv1d(mid_channels, out_channels, 5, padding=2)

        if not self.is_last:
            self.group_norm_2 = GroupNorm(out_channels, 1, min_channels_per_group=1)
            self.gelu_2 = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv_skip(hidden_states) if self.has_conv_skip else hidden_states

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.group_norm_1(hidden_states)
        hidden_states = self.gelu_1(hidden_states)
        hidden_states = self.conv_2(hidden_states)

        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)

        output = hidden_states + residual
        return output

class UNetMidBlock1D(nn.Module):
    def __init__(self, mid_channels: int, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
        self.up = Upsample1d(kernel="cubic")

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.down(hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states




MidBlockType = Union[MidResTemporalBlock1D, ValueFunctionMidBlock1D, UNetMidBlock1D]
OutBlockType = Union[OutConv1DBlock, OutValueFunctionBlock]


def get_mid_block(
    mid_block_type: str,
    num_layers: int,
    in_channels: int,
    mid_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
) -> MidBlockType:
    if mid_block_type == "MidResTemporalBlock1D":
        return MidResTemporalBlock1D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    elif mid_block_type == "ValueFunctionMidBlock1D":
        return ValueFunctionMidBlock1D(in_channels=in_channels, out_channels=out_channels, embed_dim=embed_dim)
    elif mid_block_type == "UNetMidBlock1D":
        return UNetMidBlock1D(in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels)
    raise ValueError(f"{mid_block_type} does not exist.")


def get_out_block(
    *, out_block_type: str, num_groups_out: int, embed_dim: int, out_channels: int, act_fn: str, fc_dim: int
) -> Optional[OutBlockType]:
    if out_block_type == "OutConv1DBlock":
        return OutConv1DBlock(num_groups_out, out_channels, embed_dim, act_fn)
    elif out_block_type == "ValueFunction":
        return OutValueFunctionBlock(fc_dim, embed_dim, act_fn)
    return None


"""
class Downsample2D(nn.Module):
    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    

    def __init__(self,channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = torch.nn.RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = nn.Conv2d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        self.Conv2d_0 = conv
        self.conv = conv

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states

"""