from .configs import VAEParams, BaselineModelParams, UNetParamsHF, UNetParamsNV, SchedulerParams
from .autoencoders import VariationalAutoencoder
from .utils import build_baseline_model, ModelLens, pass_config
from .unets import ClimsimUnet, DiffusersUNet1D
import diffusers
import dataclasses

MODEL_REGISTRY = {
    "vae" : (VAEParams, VariationalAutoencoder),
    "baseline" : (BaselineModelParams, build_baseline_model),
    "unet2d" : (UNetParamsHF, diffusers.UNet2DModel),
    "unet1d-hf" : (UNetParamsHF, DiffusersUNet1D),
    "unet1d-nv" : (UNetParamsNV, ClimsimUnet),
    "ddpm-scheduler" : (SchedulerParams, diffusers.DDPMScheduler),
    "ddim-scheduler" : (SchedulerParams, diffusers.DDIMScheduler),
}

model_table = {
    'best_diffusion_2d' : ('diffusion_hp_search', 'lr-explore', 'lr-explorea', "best"),
    'vintage_diffusion_2d' : ( "full_dataset_testrun" , 'trial_1b', 'trial_1b', ""),
    'diff_1d_v2' : ('diffusion_hp_search', 'diff_1d', 'diff_1da', ""),
    'diff_1d_v1' : ('diffusion_hp_search', 'diff_1d_v1', 'diff_1d_v1a', "best-"),
}


def load_mconfig(config_dict):
    if dataclasses.is_dataclass(config_dict):
        mtype = config_dict.model_type.lower()
    else:
        mtype = config_dict['model_type'].lower()
    if mtype not in MODEL_REGISTRY:
        raise ValueError(f"Model type {mtype} not supported")
    data_class = MODEL_REGISTRY[mtype][0]
    if dataclasses.is_dataclass(config_dict):
        assert isinstance(config_dict, data_class), f"Config must be instance of {data_class}"
        return(config_dict)
    return(data_class(**config_dict))
    


def load_model(config, apply_lens = True, **kwargs):
    config = load_mconfig(config)
    model_builder = MODEL_REGISTRY[config.model_type.lower()][1]
    model = pass_config(model_builder, config)
    if('device' in kwargs):
        model = model.to(kwargs["device"])
    if("distributed" in kwargs and kwargs["distributed"]):
        assert "rank" in kwargs, "rank must be provided if distributed is True"
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[kwargs["rank"]])
    if apply_lens:
        model = ModelLens(model)
    return(model)