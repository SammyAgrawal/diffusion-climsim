from .configs import VAEParams, BaselineModelParams, UNetParamsHF, UNetParamsNV
from .autoencoders import VariationalAutoencoder
from .utils import build_baseline_model, ModelLens, pass_config
from .unets import ClimsimUnet, DiffusersUNet1D, UNet2DModel
import torch

MODEL_REGISTRY = {
    "vae" : (VAEParams, VariationalAutoencoder),
    "baseline" : (BaselineModelParams, build_baseline_model),
    "unet2d" : (UNetParamsHF, UNet2DModel),
    "unet1d-hf" : (UNetParamsHF, DiffusersUNet1D),
    "unet1d-nv" : (UNetParamsNV, ClimsimUnet),
}

def load_model(config, apply_lens = True, **kwargs):
    mtype = config.model_type.lower()
    if mtype not in MODEL_REGISTRY:
        raise ValueError(f"Model type {config.model_type} not supported")
    data_class, model_builder = MODEL_REGISTRY[mtype]
    assert isinstance(config, data_class), f"Config must be instance of {data_class}"
    model = pass_config(model_builder, config)
    if('device' in kwargs):
        model = model.to(kwargs["device"])
    if("distributed" in kwargs and kwargs["distributed"]):
        assert "rank" in kwargs, "rank must be provided if distributed is True"
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[kwargs["rank"]])
    if apply_lens:
        model = ModelLens(model)
    return(model)