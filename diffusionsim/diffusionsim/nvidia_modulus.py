from re import X
import torch.nn as nn
from dataclasses import dataclass
import modulus
import nvtx
from typing import Any, Dict, List, Optional
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
import numpy as np
import torch
from torch.nn.functional import silu
import einops
import einsum
from collections import OrderedDict


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """
    ACT2CLS = {
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
    }

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


def weight_init(shape: tuple, mode: str, fan_in: int, fan_out: int):
    """
    Unified routine for initializing weights and biases.
    This function provides a unified interface for various weight initialization
    strategies like Xavier (Glorot) and Kaiming (He) initializations.

    Parameters
    ----------
    shape : tuple
        The shape of the tensor to initialize. It could represent weights or biases
        of a layer in a neural network.
    mode : str
        The mode/type of initialization to use. Supported values are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
    fan_in : int
        The number of input units in the weight tensor. For convolutional layers,
        this typically represents the number of input channels times the kernel height
        times the kernel width.
    fan_out : int
        The number of output units in the weight tensor. For convolutional layers,
        this typically represents the number of output channels times the kernel height
        times the kernel width.

    Returns
    -------
    torch.Tensor
        The initialized tensor based on the specified mode.

    Raises
    ------
    ValueError
        If the provided `mode` is not one of the supported initialization modes.
    """
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(torch.nn.Module):
    """
    A fully connected (dense) layer implementation. The layer's weights and biases can
    be initialized using custom initialization strategies like "kaiming_normal",
    and can be further scaled by factors `init_weight` and `init_bias`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an additive
        bias. By default True.
    init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: str = "kaiming_normal",
        init_weight: int = 1,
        init_bias: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(dtype=x.dtype, device=x.device).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(dtype=x.dtype, device=x.device))
        return x

class Conv1d(torch.nn.Module):
    """
    A custom 1D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations. The layer's weights
    and biases canbe initialized using custom initialization strategies like
    "kaiming_normal", and can be further scaled by factors `init_weight` and
    `init_bias`.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel : int
        Size of the convolving kernel.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an
        additive bias. By default True.
    up : bool, optional
        Whether to perform up-sampling. By default False.
    down : bool, optional
        Whether to perform down-sampling. By default False.
    resample_filter : List[int], optional
        Filter to be used for resampling. By default [1, 1].
    fused_resample : bool, optional
        If True, performs fused up-sampling and convolution or fused down-sampling
        and convolution. By default False.
    init_mode : str, optional (default="kaiming_normal")
        init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.0.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.0.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: Optional[List[int]] = None,
        fused_resample: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        resample_filter = resample_filter if resample_filter is not None else [1, 1] 
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel,
            fan_out=out_channels * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        # f = torch.as_tensor(resample_filter, dtype=torch.float32)
        # f = f.unsqueeze(0).unsqueeze(1) / f.sum()
        f = torch.tensor(resample_filter, dtype=torch.float32).unsqueeze(0).unsqueeze(1) / sum(resample_filter)
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(dtype=x.dtype, device=x.device) if self.weight is not None else None
        b = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None

        # f = self.resample_filter if self.resample_filter is not None else torch.tensor([], dtype=x.dtype, device=x.device)
        # w_pad = w.shape[-1] // 2 if w is not None else 0
        # f_pad = (f.size(-1) - 1) // 2 if f.numel() > 0 else 0  # Check for empty tensor

        # Directly use self.resample_filter without creating an empty tensor
        f = self.resample_filter

        w_pad = w.shape[-1] // 2 if w is not None else 0
        # Adjust f_pad calculation based on whether f is None or not
        f_pad = (f.size(-1) - 1) // 2 if f is not None else 0  # Use f directly
        # Adjust convolution operations based on the existence of f
        if f is not None:

            if self.fused_resample and self.up and w is not None:
                x = torch.nn.functional.conv_transpose1d(
                    x,
                    f.repeat(self.in_channels, 1, 1) * 2,
                    groups=self.in_channels,
                    stride=2,
                    padding=max(f_pad - w_pad, 0),
                )
                x = torch.nn.functional.conv1d(x, w, padding=max(w_pad - f_pad, 0))
            elif self.fused_resample and self.down and w is not None:
                x = torch.nn.functional.conv1d(x, w, padding=w_pad + f_pad)
                x = torch.nn.functional.conv1d(
                    x,
                    f.repeat(self.out_channels, 1, 1),
                    groups=self.out_channels,
                    stride=2,
                )
            else:
                if self.up:
                    x = torch.nn.functional.conv_transpose1d(
                        x,
                        f.repeat(self.in_channels, 1, 1) * 2,
                        groups=self.in_channels,
                        stride=2,
                        padding=f_pad,
                    )
                if self.down:
                    x = torch.nn.functional.conv1d(
                        x,
                        f.repeat(self.in_channels, 1, 1),
                        groups=self.in_channels,
                        stride=2,
                        padding=f_pad,
                    )
                if w is not None:
                    x = torch.nn.functional.conv1d(x, w, padding=w_pad)

        else:            
            if w is not None:
                x = torch.nn.functional.conv1d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1))
        return x

class GroupNorm(torch.nn.Module):
    """
    A custom Group Normalization layer implementation.

    Group Normalization (GN) divides the channels of the input tensor into groups and
    normalizes the features within each group independently. It does not require the
    batch size as in Batch Normalization, making itsuitable for batch sizes of any size
    or even for batch-free scenarios.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional
        Desired number of groups to divide the input channels, by default 32.
        This might be adjusted based on the `min_channels_per_group`.
    min_channels_per_group : int, optional
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number. By default 4.
    eps : float, optional
        A small number added to the variance to prevent division by zero, by default
        1e-5.

    Notes
    -----
    If `num_channels` is not divisible by `num_groups`, the actual number of groups
    might be adjusted to satisfy the `min_channels_per_group` condition.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
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


class UNetCondEmbedding(torch.nn.Module):
    def __init__(
        self, 
        num_channels: int, 
        embed_dim: int = 0, 
        embedding_type: str = "positional",
        emb_method: str = "concat", 
        activation: str = "",
        projection_method: str = "linear",
        **kwargs
    ):
        super().__init__()
        self.act_fn = get_activation(activation) if activation else None
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.emb_method = emb_method

        # create embedding
        # create a 385x8 trainable weight embedding for the input
        match embedding_type:
            case "fourier" | "gaussian":
                assert embed_dim % 2 == 0, f"`time_embed_dim` should be divisible by 2, but is {embed_dim}."
                self.emb_proj = GaussianFourierProjection(embedding_size=embed_dim // 2, set_W_to_weight=False, log=False)
            case "positional":
                self.emb_proj = Timesteps(embed_dim, flip_sin_to_cos=False, downscale_freq_shift=0.0)
            case "discrete_embedding":
                self.emb_proj = torch.nn.Embedding(num_embeddings=num_channels, embedding_dim=embed_dim)
            case "sinusoidal":
                raise NotImplementedError("Sinusoidal embedding not implemented")
            case _:
                self.emb_proj = None

        # define how embedding is injected into the sample        
        match emb_method:
            case "concat":
                pass
            case "add":
                if not embed_dim:
                    embed_dim = num_channels
                if projection_method == "linear":
                    self.emb_linear = Linear(embed_dim, num_channels)
                elif projection_method == "mlp":
                    act_fn = self.act_fn if self.act_fn else torch.nn.SiLU()
                    self.emb_linear = torch.nn.Sequential(Linear(embed_dim, embed_dim), act_fn, Linear(embed_dim, num_channels))
                else:
                    assert embed_dim == num_channels, "no projection specified but channel dimension mismatch"
                self.emb_linear = None
            case t if "attention" in t or "attn" in t:
                raise NotImplementedError("Attention embedding not implemented")
            case _:
                raise ValueError(f"Embedding method {emb_method} not supported")
    
    def forward(self, sample, embed):
        assert embed.shape[0] == sample.shape[0], "Batch size of sample and embed must match"
        self.sample_dims = sample.shape
        if self.act_fn:
            embed = self.act_fn(embed)
        
        if self.emb_proj:
            embed = self.emb_proj(embed)
        
        match self.emb_method:
            case "concat":
                pattern = 'b e -> b e ' + ' '.join([f'd{i}' for i in range(len(sample.shape) - 2)])
                embed = einops.repeat(embed, pattern, **{f'd{i}': s for i, s in enumerate(sample.shape[2:])})
                return torch.cat([sample, embed], dim=1)
            case "add":
                if self.emb_linear is not None:
                    embed = self.emb_linear(embed)
                embed = einops.rearrange(embed, "b c -> b c" + " 1" * (len(self.sample_dims) - len(embed.shape)))
                return sample + embed
            case t if "attention" in t or "attn" in t:
                raise NotImplementedError("Attention embedding not implemented")
            case _:
                raise ValueError(f"Embedding method {self.emb_method} not supported")


class UNetBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_blocks: OrderedDict[str, UNetCondEmbedding] = None,
        kernel_size: int = 3,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        use_scriptable_attention: bool = True,
        num_heads: int = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1,1],
        resample_proj: bool = False,
        adaptive_scale: bool = False,
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_blocks = embedding_blocks
        self.num_heads = (
            0
            if not attention
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel_size,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        # self.affine = Linear(in_features=embed_dims, out_features=out_channels * (2 if adaptive_scale else 1),  **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv1d(
            in_channels=out_channels, out_channels=out_channels, kernel=kernel_size, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv1d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )
            if use_scriptable_attention:
                self.attentionop = ScriptableAttentionOp()
    
    def forward(self, x, conditioning_signals={}):
        skip_res = self.skip(x) if self.skip is not None else x
        x = self.conv0(silu(self.norm0(x)))

        # params = self.affine(emb).unsqueeze(2).to(x.dtype)
        # if self.adaptive_scale:
        #     scale, shift = params.chunk(chunks=2, dim=1)
        #     x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        # else:
        #     x = silu(self.norm1(x.add_(params)))

        x = self.norm1(x)
        if self.embedding_blocks is not None:
            for embed_name, embedding in conditioning_signals.items():
                if embed_name in self.embedding_blocks:
                    x = self.embedding_blocks[embed_name](x, embedding)
                else:
                    print(f"Warning: {embed_name} not found in embedding_blocks")
            
        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(skip_res)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            if self.attentionop is not None:
                w = self.attentionop(q, k)
            else:
                w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            # batch_size, channels, length = x.size()
            # x = self.proj(a.reshape(batch_size, channels, length)).add_(x)
            x = x * self.skip_scale
        return x


class UNetBlock_noatten(UNetBlock):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int = 0,
        kernel_size: int = 3,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1,1],
        resample_proj: bool = False,
        adaptive_scale: bool = False,
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
    ):
        super().__init__(
            in_channels, out_channels, emb_channels, kernel_size,
            up, down,
            False, False, None, channels_per_head,
            dropout, skip_scale,
            eps, resample_filter, resample_proj, adaptive_scale,
            init, init_zero, None)


class UNetBlock_atten(UNetBlock):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int = 0,
        kernel_size: int = 3,
        up: bool = False,
        down: bool = False,
        num_heads: int = 1,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1,1],
        resample_proj: bool = False,
        adaptive_scale: bool = False,
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        attention: bool = True,
    ):
        super().__init__(
            in_channels, out_channels, emb_channels, kernel_size,
            up, down,
            True, True, num_heads, channels_per_head,
            dropout, skip_scale, eps, resample_filter, resample_proj, adaptive_scale,
            init, init_zero, init_attn)


"""
Contains the code for the Unet and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class ClimsimUnetMetaData(modulus.ModelMetaData):
    name: str = "ClimsimUnet"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True


class ClimsimUnet(modulus.Module):
    def __init__(
            self,
            num_vars_profile: int,
            num_vars_scalar: int, 
            num_vars_profile_out: int,
            num_vars_scalar_out: int,
            diffusion_mode: bool,
            seq_resolution: int = 64,
            model_channels: int = 128,
            channel_mult: List[int] = [1, 2, 2, 2],
            num_blocks: int = 4,
            attn_resolutions: List[int] = [16],
            conditioning_resolutions: Dict[int, List[str]] = {
                64: ["timesteps"],
                32: ["timesteps"],
                16: ["timesteps"],
                8: ["timesteps"],
                4: ["timesteps"],
                2: ["timesteps"],
                1: ["timesteps"],
            },
            dropout: float = 0.10,
            channel_mult_noise: int = 1,
            encoder_type: str = "standard",
            decoder_type: str = "standard",
            resample_filter: List[int] = [1, 1],
            n_model_levels: int = 60,
            # qinput_prune=False, 
            output_prune=False, 
            strato_lev=12,
            time_embedding_dim: int = 32,
            loc_embedding: bool = False,
            loc_embedding_dim: int = 8,
            skip_conv: bool = False,
            prev_2d: bool = False
        ):
        
        super().__init__(meta=ClimsimUnetMetaData())
        # check if hidden_dims is a list of hidden_dims
        if diffusion_mode:
            self.num_vars_profile = num_vars_profile_out
            self.num_vars_scalar = num_vars_scalar_out
        else:
            self.num_vars_profile = num_vars_profile
            self.num_vars_scalar = num_vars_scalar
        

        
        self.num_vars_profile_out = num_vars_profile_out
        self.num_vars_scalar_out = num_vars_scalar_out
        self.diffusion_mode = diffusion_mode
        self.model_channels = model_channels

        self.in_channels = self.num_vars_profile + self.num_vars_scalar # + 7 # +(8-1)=7 for the location embedding
        self.out_channels = num_vars_profile_out + num_vars_scalar_out
        # print('1: out_channels', self.out_channels)

        # valid_encoder_types = ["standard", "skip", "residual"]
        valid_encoder_types = ["standard"]
        assert encoder_type in valid_encoder_types, f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}."

        # valid_decoder_types = ["standard", "skip"]
        valid_decoder_types = ["standard"]
        assert decoder_type in valid_decoder_types, f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}."
        self.conditioning_resolutions = conditioning_resolutions
        self.time_embedding_dim = time_embedding_dim
        self.use_loc_embedding = loc_embedding
        if loc_embedding:
            assert loc_embedding_dim > 0, f"Location embedding dimension {loc_embedding_dim} is not valid" 
            self.loc_embedding = UNetCondEmbedding(
                    num_channels=384, embed_dim=loc_embedding_dim, embedding_type="discrete_embedding", emb_method="concat"
            )
            self.in_channels += loc_embedding_dim
        self.seq_resolution = seq_resolution
        self.channel_mult = channel_mult
        self.num_blocks = num_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.channel_mult_noise = channel_mult_noise
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.resample_filter = resample_filter
        self.n_model_levels = n_model_levels
        self.input_padding = (seq_resolution-n_model_levels, 0)
        # self.qinput_prune=qinput_prune
        self.output_prune=output_prune
        self.strato_lev=strato_lev
        self.skip_conv = skip_conv
        self.prev_2d = prev_2d

        # emb_channels = model_channels * channel_mult_emb # channel_mult_emb used to be input param
        # self.emb_channels = emb_channels
        # noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=0.2**0.5)
        block_kwargs = dict(
            # emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=0.5**0.5,
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )


        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = self.in_channels
        caux = self.in_channels
        
        for level, mult in enumerate(channel_mult):
            res = seq_resolution >> level # halves the resolution at each level
            if level == 0:
                cin = cout
                cout = model_channels
                # comment out the first conv layer that supposed to be the input embedding
                # because we will have the input embedding manusally for profile vars and scalar vars
                self.enc[f"{res}_conv"] = Conv1d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}_down"] = UNetBlock_noatten(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}_aux_down"] = Conv1d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}_aux_skip"] = Conv1d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}_aux_residual"] = Conv1d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions # attn_resolutions specifies the block levels at which attention is used
                block_select = {True : UNetBlock_atten, False : UNetBlock_noatten}
                embedding_blocks = self.return_embedding_blocks(res, cout) if self.diffusion_mode else None
                self.enc[f"{res}_block{idx}"] = block_select[attn](
                    in_channels=cin, 
                    out_channels=cout, 
                    emb_channels=0,
                    up=False,
                    down=False,
                    embedding_blocks=embedding_blocks,
                    channels_per_head=64,
                    **block_kwargs
                )
        
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        self.skip_conv_layer = [] #torch.nn.ModuleList()
        # for each skip connection, add a 1x1 conv layer initialized as identity connection, with an option to train the weight
        for idx, skip in enumerate(skips):
            conv = Conv1d(in_channels=skip, out_channels=skip, kernel=1)
            torch.nn.init.dirac_(conv.weight)
            torch.nn.init.zeros_(conv.bias)
            if not self.skip_conv:
                conv.weight.requires_grad = False
                conv.bias.requires_grad = False
            self.skip_conv_layer.append(conv)
        
        self.skip_conv_layer = torch.nn.ModuleList(self.skip_conv_layer)
            # XX doulbe check if the above is correct

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        self.dec_aux_norm = torch.nn.ModuleDict()
        self.dec_aux_conv = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = seq_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}_in0"] = UNetBlock_atten(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}_in1"] = UNetBlock_noatten(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}_up"] = UNetBlock_noatten(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                
                embedding_blocks = self.return_embedding_blocks(res, cout) if self.diffusion_mode else None
                if attn:
                    self.dec[f"{res}_block{idx}"] = UNetBlock_atten(
                        in_channels=cin, out_channels=cout, attention=attn, embedding_blocks=embedding_blocks, **block_kwargs
                    )
                else:
                    self.dec[f"{res}_block{idx}"] = UNetBlock_noatten(
                        in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                    )
            
            if decoder_type == "skip" or level == 0:
                # if decoder_type == "skip" and level < len(channel_mult) - 1:
                #     self.dec[f"{res}_aux_up"] = Conv1d(
                #         in_channels=out_channels,
                #         out_channels=out_channels,
                #         kernel=0,
                #         up=True,
                #         resample_filter=resample_filter,
                #     )
                self.dec_aux_norm[f"{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                ## comment out the last conv layer that supposed to recover the output channels
                ## we will manually recover the output channels
                self.dec_aux_conv[f"{res}_aux_conv"] = Conv1d(
                    in_channels=cout, out_channels=self.out_channels, kernel=3, **init_zero
                )
    
    def return_embedding_blocks(self, res, out_channels):
        embedding_blocks = OrderedDict()
        assert res in self.conditioning_resolutions, f"Resolution {res} not found in conditioning_resolutions"
        for cond_type in self.conditioning_resolutions[res]:
            if cond_type == "timesteps":
                assert self.time_embedding_dim > 0 and self.time_embedding_dim % 2 == 0, f"Time embedding dimension {self.time_embedding_dim} is not valid"
                embedding_blocks[cond_type] = UNetCondEmbedding(
                    num_channels=out_channels, embed_dim=self.time_embedding_dim, embedding_type="fourier", emb_method="add"
                )
            elif cond_type == "loc":
                
                embedding_blocks[cond_type] = 
            else:
                raise ValueError(f"Conditioning type {cond_type} not supported")
        return embedding_blocks
    
    def forward(self, x, timestep=None, loc=None):

        # if self.qinput_prune:
        #     x = x.clone()  # Clone the tensor to ensure you're not modifying the original tensor in-place
        #     x[:, 60:60+self.strato_lev] = x[:, 60:60+self.strato_lev].clone().zero_()  # Set stratosphere q1 to 0
        #     x[:, 120:120+self.strato_lev] = x[:, 120:120+self.strato_lev].clone().zero_()  # Set stratosphere q2 to 0
        #     x[:, 180:180+self.strato_lev] = x[:, 180:180+self.strato_lev].clone().zero_()  # Set stratosphere q3 to 0

        
        if self.diffusion_mode:
            assert timestep is not None, "Timestep is required for diffusion model"
            if not torch.is_tensor(timestep):
                timesteps = torch.tensor([timestep], dtype=torch.long, device=x.device)
                if timesteps.shape[0] != x.shape[0]:
                    timesteps = timesteps.repeat(x.shape[0])
            else:
                timesteps = timestep.to(dtype=torch.long, device=x.device)
            assert timesteps.shape[0] == x.shape[0], "Timestep shape mismatch"
        else:
            assert len(x.shape == 2), "Need input shape of (batch, num_vars_profile*levels+num_vars_scalar)"
            if not self.prev_2d:
                x = x.clone()
                x[:,-8:-3] = x[:,-8:-3].clone().zero_()
            # split x into x_profile and x_scalar
            # x_profile: (batch, num_vars_profile, levels)
            # x_scalar: (batch, num_vars_scalar)
            x_profile = x[:,:self.num_vars_profile*self.n_model_levels]
            x_scalar = x[:,self.num_vars_profile*self.n_model_levels:]
    
            # print(x_profile.shape, x_scalar.shape, x_loc.shape)

            # reshape x_profile to (batch, num_vars_profile, levels)
            x_profile = x_profile.reshape(-1, self.num_vars_profile, self.n_model_levels)
            # broadcast x_scalar to (batch, num_vars_scalar, levels)
            x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, self.n_model_levels)
            x = torch.cat((x_profile, x_scalar), dim=1) #concatenate x_profile, x_scalar to (batch, num_vars_profile+num_vars_scalar, levels)
        
        if self.use_loc_embedding:
            assert loc is not None, "Location embedding is enabled, must pass in ncol"
            if not torch.is_tensor(loc):
                loc = torch.tensor(loc, dtype=torch.int16, device=x.device)
            x = self.loc_embedding(x, loc)

        # print('2:', x.shape)
        # x = torch.cat((x_profile, x_scalar), dim=1)
        
        x = torch.nn.functional.pad(x, self.input_padding, "constant", 0.0)
        # print('3:', x.shape)
        # pass the concatenated tensor through the Unet

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux, conditioning_signals={"timesteps":timesteps})
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux, conditioning_signals={"timesteps":timesteps})
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux, conditioning_signals={"timesteps":timesteps})) / 2**0.5
            else:
                # x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                x = block(x, conditioning_signals={"timesteps":timesteps})
                skips.append(x)

        new_skips = []
        # for x_tmp, conv_tmp in zip(skips, self.skip_conv_layer):
        #     x_tmp = conv_tmp(x_tmp)
        #     new_skips.append(x_tmp)
        for idx, conv_tmp in enumerate(self.skip_conv_layer):
            x_tmp = conv_tmp(skips[idx])
            new_skips.append(x_tmp)

        aux = None
        tmp = None
        for name, block in self.dec.items():
#             print(name)
            # if "aux" not in name:
            if x.shape[1] != block.in_channels:
                # skip_ind = len(skips) - 1
                # skip_conv = self.skip_conv_layer[skip_ind]
                x = torch.cat([x, new_skips.pop()], dim=1)
            # x = block(x, emb)
            x = block(x, conditioning_signals={"timesteps":timesteps}))
            # else:
            #     # if "aux_up" in name:
            #     #     aux = block(aux)
            #     if "aux_conv" in name:
            #         tmp = block(silu(tmp))
            #         aux = tmp if aux is None else tmp + aux
            #     elif "aux_norm" in name:
            #         tmp = block(x)
        for name, block in self.dec_aux_norm.items():
            tmp = block(x)
        for name, block in self.dec_aux_conv.items():
            tmp = block(silu(tmp))
            aux = tmp if aux is None else tmp + aux
        # here x should be (batch, output_channels, seq_resolution)
        # remember that self.input_padding = (seq_resolution-n_model_levels,0)
        x = aux
        if self.diffusion_mode:
            return x
        # print('7:', x.shape)
        if self.input_padding[1]==0:
            y_profile = x[:,:self.num_vars_profile_out,self.input_padding[0]:]
            y_scalar = x[:,self.num_vars_profile_out:,self.input_padding[0]:]
        else:
            y_profile = x[:,:self.num_vars_profile_out,self.input_padding[0]:-self.input_padding[1]]
            y_scalar = x[:,self.num_vars_profile_out:,self.input_padding[0]:-self.input_padding[1]]
        #take relu on y_scalar
        y_scalar = torch.nn.functional.relu(y_scalar)
        #reshape y_profile to (batch, num_vars_profile_out*levels)
        y_profile = y_profile.reshape(-1, self.num_vars_profile_out*self.n_model_levels)

        #average y_scalar for the lev dimension to (batch, num_vars_scalar_out)
        y_scalar = y_scalar.mean(dim=2)
        # print('7.5:', y_profile.shape, y_scalar.shape)

        #concatenate y_profile and y_scalar to (batch, num_vars_profile_out*levels+num_vars_scalar_out)
        y = torch.cat((y_profile, y_scalar), dim=1)

        if self.output_prune:
            y = y.clone()
            y[:, 60:60+self.strato_lev] = y[:, 60:60+self.strato_lev].clone().zero_()
            y[:, 120:120+self.strato_lev] = y[:, 120:120+self.strato_lev].clone().zero_()
            y[:, 180:180+self.strato_lev] = y[:, 180:180+self.strato_lev].clone().zero_()
            y[:, 240:240+self.strato_lev] = y[:, 240:240+self.strato_lev].clone().zero_()
            y[:, 300:300+self.strato_lev] = y[:, 300:300+self.strato_lev].clone().zero_()

        return y