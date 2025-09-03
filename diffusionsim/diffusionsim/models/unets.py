import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers
from diffusers import UNet2DModel
import physicsnemo as modulus
import dataclasses
from collections import OrderedDict
import einops
from typing import Any, Dict, List, Optional, Tuple
from .conv_blocks import *

"""
Contains the code for the Unet and its training.
"""

class UNetCondEmbedding(torch.nn.Module):
    def __init__(
        self, 
        num_channels: int, 
        embed_dim: int = 0, 
        embedding_type: str = "",
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
                self.emb_proj = diffusers.models.embeddings.GaussianFourierProjection(embedding_size=embed_dim // 2, set_W_to_weight=False, log=False)
            case "positional":
                self.emb_proj = diffusers.models.embeddings.Timesteps(embed_dim, flip_sin_to_cos=False, downscale_freq_shift=0.0)
            case "discrete_embedding":
                self.emb_proj = torch.nn.Embedding(num_embeddings=num_channels, embedding_dim=embed_dim)
            case "sinusoidal":
                raise NotImplementedError("Sinusoidal embedding not implemented")
            case _:
                self.emb_proj = None
        """
        self.affine = Linear(in_features=embed_dims, out_features=out_channels * (2 if adaptive_scale else 1),  **init)
        # params = self.affine(emb).unsqueeze(2).to(x.dtype)
        # if self.adaptive_scale:
        #     scale, shift = params.chunk(chunks=2, dim=1)
        #     x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        # else:
        #     x = F.silu(self.norm1(x.add_(params)))
        """

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
        self.embedding_blocks = nn.ModuleDict(embedding_blocks) if embedding_blocks is not None else None
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
        x = self.conv0(F.silu(self.norm0(x)))
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
        embedding_blocks: OrderedDict[str, UNetCondEmbedding] = None,
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
            in_channels, out_channels, embedding_blocks, kernel_size,
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
        embedding_blocks: OrderedDict[str, UNetCondEmbedding] = None,
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
            in_channels, out_channels, embedding_blocks, kernel_size,
            up, down,
            True, True, num_heads, channels_per_head,
            dropout, skip_scale, eps, resample_filter, resample_proj, adaptive_scale,
            init, init_zero, init_attn)

class DownBlock1D(nn.Module):
    def __init__(
        self, 
        out_channels: int, 
        in_channels: int, 
        mid_channels: Optional[int] = None, 
        use_down=False,
        use_up=False, 
        concat_temb=False,
        attention=False,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        in_channels = 2 * in_channels if attention else in_channels
        self.down = Downsample1d("cubic") if use_down else None
        self.up = Upsample1d(kernel="cubic") if use_up else None
        
        self.resnets = nn.ModuleList([
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ])
        self.attentions = None
        if attention:
            self.attentions = nn.ModuleList([
                SelfAttention1d(mid_channels, mid_channels // 32),
                SelfAttention1d(mid_channels, mid_channels // 32),
                SelfAttention1d(out_channels, out_channels // 32),
            ])
            
        self.time_embedding = UNetCondEmbedding(in_channels, emb_method="concat") if concat_temb else None

    def forward(
            self, 
            hidden_states: torch.Tensor, 
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        
        hidden_states = self.down(hidden_states) if self.down is not None else hidden_states
        hidden_states = self.time_embedding(hidden_states, temb) if self.time_embedding is not None else hidden_states
        hidden_states = torch.cat([hidden_states, res_hidden_states_tuple[-1]], dim=1) if self.attentions is not None else hidden_states

        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states)
            if self.attentions is not None:
                hidden_states = self.attentions[i](hidden_states)
        
        hidden_states = self.up(hidden_states) if self.up is not None else hidden_states

        return hidden_states, (hidden_states,)

class UpBlock1D(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            mid_channels: Optional[int] = None, 
            use_up=True,
            attention=False
        ):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        self.resnets = nn.ModuleList([
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ])
        self.attentions = None
        if attention:
            self.attentions = nn.ModuleList([
                SelfAttention1d(mid_channels, mid_channels // 32),
                SelfAttention1d(mid_channels, mid_channels // 32),
                SelfAttention1d(out_channels, out_channels // 32),
            ])
        
        self.up = Upsample1d(kernel="cubic") if use_up else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states)
            if self.attentions is not None:
                hidden_states = self.attentions[i](hidden_states)

        hidden_states = self.up(hidden_states) if self.up else hidden_states

        return hidden_states

class DownResnetBlock1D(nn.Module): # composed of ResidualTemporalBlocks
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        conv_shortcut: bool = False,
        temb_channels: int = 32,
        groups: int = 32,
        groups_out: Optional[int] = None,
        non_linearity: Optional[str] = None,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.add_downsample = add_downsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.ModuleList(resnets)

        self.nonlinearity = None if non_linearity is None else get_activation(non_linearity)

        self.downsample = None
        if add_downsample:
            self.downsample = Conv1d(out_channels, out_channels, kernel=3, bias=False, stride=2, padding=1)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        output_states = ()
        hidden_states = self.resnets[0](hidden_states, temb) 
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
        
        output_states += (hidden_states,)

        return hidden_states, output_states

class UpResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        temb_channels: int = 32,
        groups: int = 32,
        groups_out: Optional[int] = None,
        non_linearity: Optional[str] = None,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm
        self.add_upsample = add_upsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [ResidualTemporalBlock1D(2 * in_channels, out_channels, embed_dim=temb_channels)]

        for _ in range(num_layers):
            resnets.append(ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=temb_channels))

        self.resnets = nn.ModuleList(resnets)

        if non_linearity is None:
            self.nonlinearity = None
        else:
            self.nonlinearity = get_activation(non_linearity)

        self.upsample = None
        if add_upsample:
            self.upsample = nn.ConvTranspose1d(out_channels, out_channels, 4, 2, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if res_hidden_states_tuple is not None:
            res_hidden_states = res_hidden_states_tuple[-1]
            hidden_states = torch.cat((hidden_states, res_hidden_states), dim=1)

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states

def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
) -> Union[DownResnetBlock1D, DownBlock1D, UNetBlock]:
    if down_block_type == "DownResnetBlock1D":
        return DownResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
        )
    elif down_block_type == "DownBlock1D":
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels, use_down=True, use_up=False, concat_temb=False, attention=False)
    elif down_block_type == "AttnDownBlock1D":
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels, use_down=False, use_up=True, concat_temb=False, attention=True)
    elif down_block_type == "DownBlock1DNoSkip":
        return DownBlock1D(out_channels=out_channels, in_channels=in_channels, use_down=False, use_up=False, concat_temb=True, attention=False)
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type: str, num_layers: int, in_channels: int, out_channels: int, temb_channels: int, add_upsample: bool
) -> Union[UpResnetBlock1D, UpBlock1D, UNetBlock]:
    if up_block_type == "UpResnetBlock1D":
        return UpResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
        )
    elif up_block_type == "UpBlock1D":
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels, use_up=True, attention=False)
    elif up_block_type == "AttnUpBlock1D":
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels, use_up=True, attention=True)
    elif up_block_type == "UpBlock1DNoSkip":
        return UpBlock1D(in_channels=in_channels, out_channels=out_channels, use_up=False, attention=False)
    raise ValueError(f"{up_block_type} does not exist.")


@dataclasses.dataclass
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
            kernel_size: int = 3,
            output_scale_type: str = '',
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
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.resample_filter = resample_filter
        self.n_model_levels = n_model_levels
        self.input_padding = (seq_resolution-n_model_levels, 0)
        # self.qinput_prune=qinput_prune
        self.output_prune=output_prune
        self.strato_lev=strato_lev
        self.skip_conv = skip_conv

        # emb_channels = model_channels * channel_mult_emb # channel_mult_emb used to be input param
        # self.emb_channels = emb_channels
        # noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=0.2**0.5)
        block_kwargs = dict(
            # emb_channels=emb_channels,
            num_heads=1,
            kernel_size=kernel_size,
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
        
        if output_scale_type == 'affine' or "bias" in output_scale_type:
            self.scale_output = ChannelAffine(self.out_channels, cross_channel=False, bias=True)
        elif output_scale_type == 'scale-only' or output_scale_type == 'linear':
            self.scale_output = ChannelAffine(self.out_channels, cross_channel=False, bias=False)
        elif 'cross' in output_scale_type or 'mixed' in output_scale_type:
            self.scale_output = ChannelAffine(self.out_channels, cross_channel=True, bias=False)
        else:
            self.scale_output = None

    def return_embedding_blocks(self, res, out_channels):
        embedding_blocks = OrderedDict()
        assert res in self.conditioning_resolutions, f"Resolution {res} not found in conditioning_resolutions"
        for cond_type in self.conditioning_resolutions[res]:
            if cond_type == "timesteps":
                assert self.time_embedding_dim > 0 and self.time_embedding_dim % 2 == 0, f"Time embedding dimension {self.time_embedding_dim} is not valid"
                embedding_blocks[cond_type] = UNetCondEmbedding(
                    num_channels=out_channels, embed_dim=self.time_embedding_dim, embedding_type="fourier", emb_method="add"
                )
            else:
                raise ValueError(f"Conditioning type {cond_type} not supported")
        if len(embedding_blocks) == 0:
            return None
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
            #if not self.prev_2d:
            #    x = x.clone()
            #    x[:,-8:-3] = x[:,-8:-3].clone().zero_()
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

        x = torch.nn.functional.pad(x, self.input_padding, "constant", 0.0)
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
            # if "aux" not in name:
            if x.shape[1] != block.in_channels:
                # skip_ind = len(skips) - 1
                # skip_conv = self.skip_conv_layer[skip_ind]
                skip_tensor = new_skips.pop()
                if skip_tensor.shape[-1] != x.shape[-1]: # results from padding
                    min_len = min(skip_tensor.shape[-1], x.shape[-1])
                    skip_tensor, x = skip_tensor[..., -min_len:], x[...,-min_len:]
                x = torch.cat([x, skip_tensor], dim=1)
            # x = block(x, emb)
            x = block(x, conditioning_signals={"timesteps":timesteps})
            # else:
            #     # if "aux_up" in name:
            #     #     aux = block(aux)
            #     if "aux_conv" in name:
            #         tmp = block(F.silu(tmp))
            #         aux = tmp if aux is None else tmp + aux
            #     elif "aux_norm" in name:
            #         tmp = block(x)
        for name, block in self.dec_aux_norm.items():
            tmp = block(x)
        for name, block in self.dec_aux_conv.items():
            tmp = block(F.silu(tmp))
            aux = tmp if aux is None else tmp + aux
        # here x should be (batch, output_channels, seq_resolution)
        # remember that self.input_padding = (seq_resolution-n_model_levels,0)
        x = aux
        if self.scale_output:
            x = self.scale_output(x)
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


@dataclasses.dataclass
class UNet1DOutput(diffusers.utils.BaseOutput):
    sample: torch.Tensor


class DiffusersUNet1D(diffusers.models.ModelMixin, diffusers.configuration_utils.ConfigMixin):
    r"""
    A 1D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.
    Inspired by https://github.com/leap-stc/ClimSim/blob/main/online_testing/baseline_models/Unet_v5/training/climsim_unet.py
    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (`int`, *optional*, defaults to 2): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 2): Number of channels in the output.
        extra_in_channels (`int`, *optional*, defaults to 0):
            Number of additional channels to be added to the input of the first down block. Useful for cases where the
            input data has more channels than what the model was initially designed for.
        time_embedding_type (`str`, *optional*, defaults to `"fourier"`): Type of time embedding to use.
        freq_shift (`float`, *optional*, defaults to 0.0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D")`):
            Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(32, 32, 64)`):
            Tuple of block output channels.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock1D"`): Block type for middle of UNet.
        out_block_type (`str`, *optional*, defaults to `None`): Optional output processing block of UNet.
        act_fn (`str`, *optional*, defaults to `None`): Optional activation function in UNet blocks.
        norm_num_groups (`int`, *optional*, defaults to 8): The number of groups for normalization.
        layers_per_block (`int`, *optional*, defaults to 1): The number of layers per block.
        downsample_each_block (`int`, *optional*, defaults to `False`):
            Experimental feature for using a UNet without upsampling.
    """

    _skip_layerwise_casting_patterns = ["norm"]

    @diffusers.configuration_utils.register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        extra_in_channels: int = 0,
        time_embedding_type: str = "fourier",
        time_embedding_dim: int = 0,
        flip_sin_to_cos: bool = True,
        use_timestep_embedding: bool = False,
        freq_shift: float = 0.0,
        down_block_types: Tuple[str] = ("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
        up_block_types: Tuple[str] = ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
        mid_block_type: Tuple[str] = "UNetMidBlock1D",
        out_block_type: str = None,
        block_out_channels: Tuple[int] = (32, 32, 64),
        act_fn: str = None,
        norm_num_groups: int = 8,
        layers_per_block: int = 1,
        downsample_each_block: bool = False,
        output_scale_type: str = "scale-only",
    ):
        super().__init__()

        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = diffusers.models.embeddings.GaussianFourierProjection(
                embedding_size=time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0]
            self.time_proj = diffusers.models.embeddings.Timesteps(
                time_embed_dim, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift
            )
            timestep_input_dim = time_embed_dim
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        if use_timestep_embedding:
            time_embed_dim = block_out_channels[0] * 4
            self.time_mlp = diffusers.models.embeddings.TimestepEmbedding(
                in_channels=timestep_input_dim,
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
                out_dim=block_out_channels[0],
            )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None

        # down
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if i == 0:
                input_channel += extra_in_channels

            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=timestep_input_dim,
                add_downsample=not is_final_block or downsample_each_block,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            temb_channels=timestep_input_dim,
            num_layers=layers_per_block,
            add_downsample=downsample_each_block,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        if out_block_type is None:
            final_upsample_channels = out_channels
        else:
            final_upsample_channels = block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = (
                reversed_block_out_channels[i + 1] if i < len(up_block_types) - 1 else final_upsample_channels
            )

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=timestep_input_dim,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.out_block = get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=block_out_channels[0],
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=block_out_channels[-1] // 4,
        )
        if output_scale_type == 'affine' or "bias" in output_scale_type:
            self.scale_output = ChannelAffine(out_channels, cross_channel=False, bias=True)
        elif output_scale_type == 'scale-only' or output_scale_type == 'linear':
            self.scale_output = ChannelAffine(out_channels, cross_channel=False, bias=False)
        elif 'cross' in output_scale_type or 'mixed' in output_scale_type:
            self.scale_output = ChannelAffine(out_channels, cross_channel=True, bias=False)
        else:
            self.scale_output = None   

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
        debug=False,
    ) -> Union[UNet1DOutput, Tuple, torch.Tensor]:
        r"""
        The [`UNet1DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_1d.UNet1DOutput`] instead of a plain tuple.
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timestep_embed = self.time_proj(timesteps)
        if self.config.use_timestep_embedding:
            timestep_embed = self.time_mlp(timestep_embed.to(sample.dtype))
        timestep_embed = timestep_embed.broadcast_to((sample.shape[:1] + timestep_embed.shape[1:]))
        if debug:
            print(sample.shape, timestep_embed.shape)
        # 2. down
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=timestep_embed)
            down_block_res_samples += res_samples

        # 3. mid
        if self.mid_block:
            sample = self.mid_block(sample, timestep_embed)

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(sample, res_hidden_states_tuple=res_samples, temb=timestep_embed)

        # 5. post-process
        if self.out_block:
            sample = self.out_block(sample, timestep_embed)
        
        if self.scale_output:
            sample = self.scale_output(sample)
        return sample
