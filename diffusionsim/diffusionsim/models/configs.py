from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional



@dataclass
class VAEParams:
    model_type: str = "vae"
    data_dims: int = 128
    label_dims: int = 128
    latent_dims: int = 16
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    disable_logstd_bias: bool = True


@dataclass
class BaselineModelParams:
    model_type: str = "baseline"
    baseline_model_type: str = "ANN"
    bl_model_dir: str = "/mnt/home/ssa2206/Climsim/climsim-online/storage/shared_e3sm/saved_models/wrapper"
    bl_load_model_name: str = None
    bl_input_size: int = 124
    bl_output_size: int = 128
    bl_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256]) 


@dataclass
class UNetParamsHF:
    model_type: str = "unet1d-hf"
    in_channels: int = 10
    out_channels: int = 10
    extra_in_channels: int = 0
    block_out_channels: Tuple = field(default_factory=lambda: (32, 64, 64, 128))  # num output channel for each UNet block
    down_block_types: Tuple = field(default_factory=lambda: ("DownResnetBlock1D", "AttnDownBlock1D", "DownBlock1D"))
    up_block_types: Tuple = field(default_factory=lambda: ("UpResnetBlock1D", "AttnUpBlock1D", "UpBlock1D"))
    layers_per_block: int = 2
    norm_num_groups: int = 4
    act_fn: str = "silu" # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py#L27
    freq_shift: float = 0.0 # fourier freq shift
    use_timestep_embedding: bool = True
    time_embedding_dim: int = 32
    act_fn: str = None

@dataclass
class UNetParamsNV:
    model_type: str = "unet1d-nv"
    num_vars_profile: int = 2
    num_vars_scalar: int = 4
    num_vars_profile_out: int = 2
    num_vars_scalar_out: int = 8
    diffusion_mode: bool = True
    model_channels: int = 64
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    num_blocks: int = 3
    attn_resolutions: List[int] = field(default_factory=lambda: [16])
    conditioning_resolutions: Dict[int, List[str]] = field(default_factory=lambda: {
        64: ["timesteps"],
        32: ["timesteps"],
        16: ["timesteps"],
        8: ["timesteps"],
    })
    dropout: float = 0.0
    output_scale_type: str = 'scale-only'
    kernel_size: int = 3
    time_embedding_dim: int = 32


# scheduler params
@dataclass
class SchedulerParams:
    model_type: str = 'ddpm-scheduler'
    num_train_timesteps: int = 100
    beta_schedule: str = 'linear'
    clip_sample: bool = False
    clip_sample_range: float = 4.0
    beta_end: float = 0.02
    prediction_type: str = 'v_prediction'
    scheduler_inference_steps: int = 100
