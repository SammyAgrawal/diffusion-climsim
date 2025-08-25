import xarray as xr
import numpy as np
import pandas as pd
import numpy as np
import os
import diffusers
import gcsfs
import json
import torch

from .models import load_model, build_baseline_model
from .mydatasets import load_dataset, load_dataloaders, load_scheduler, log_event
from .climsim_utils import imagify
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple




@dataclass
class DataLoaderParams:
    batch_size: int = 128
    shuffle: bool = True
    num_workers: int = 0
    prefetch_factor: int = None
    persistent_workers: bool = False
    multiprocessing_context: str = None
    pin_memory: bool = False

@dataclass
class DataConfig:
    dataset_type: str = "XBatchDataset"
    climsim_type: str = "low-res-expanded" 
    source: str = "local-vzarr"
    data_dir: str = "/mnt/home/ssa2206/Climsim/diffusion-climsim/data/local_manifests"
    num_epochs: int = 10
    phases: List[str] = field(default_factory=lambda: ['train', 'eval'])
    train_test_split: List[int] = field(default_factory=lambda: [1.0, 0.0])
    dataloader_params: DataLoaderParams = field(default_factory=lambda: DataLoaderParams())
    #xarr_subsamples: Tuple[int, int, int] = (36,210240, 144)
    data_vars: str = "v1"
    use_tendencies: bool = False
    norm_info: str = "image"
    chunksize: Dict = field(default_factory=lambda:{})
    shuffle_indices: bool = True
    distributed_training: bool = False
    # logging params
    checkpoint_best_epoch: bool = True
    checkpoint_every_epoch: bool = False
    batch_logging_interval: int = 32
    batch_checkpoint_interval: int = 50 # save checkpoint every 10 batches
    log_gradients: bool = False

    
    def __post_init__(self):
        if isinstance(self.dataloader_params, dict):
            self.dataloader_params = DataLoaderParams(**self.dataloader_params)

def my_dconfig(source="local-vzarr", data_vars='v1', in_notebook=True, shuffle_indices=True,
               dataset_type="climsim", batch_size=128, use_tendencies = True, **kwargs):
    dconfig = DataConfig(source=source, dataset_type=dataset_type, shuffle_indices=shuffle_indices, data_vars=data_vars, use_tendencies=use_tendencies, **kwargs)
    dconfig.phases = ['train', 'eval'] if "climsim" in dataset_type else ['train']
    if "train_test_split" not in kwargs:
        dconfig.train_test_split = [0.45, 0.20] if "climsim" in dataset_type else [1.0]

    dl_params = DataLoaderParams(batch_size=batch_size)
    dconfig.checkpoint_every_epoch = False if "diffusion" in dataset_type else True
    dconfig.batch_logging_interval = 128 if "diffusion" in dataset_type else 16
    dconfig.batch_checkpoint_interval = 100 if "diffusion" in dataset_type else 50
    dconfig.log_gradients = False if "diffusion" in dataset_type else True
    
    if(torch.cuda.is_available() and batch_size > 16 and not in_notebook):
        dl_params.pin_memory = True
    if(not in_notebook):
        dl_params.num_workers = 4
        dl_params.prefetch_factor = 3
        dl_params.persistent_workers = True
        dl_params.multiprocessing_context = "forkserver"
    return(dconfig)
            
@dataclass
class TrainingConfig:
    exp_id: str
    run_id: str
    # learning parameters
    optimizer: str = 'adam'
    learning_rate_params: Dict = field(default_factory=lambda: {
        "learning_rate" : 1e-4, "lr_scheduler" : None, "patience" : 5, "betas" : (0.9, 0.999),
        "lr_warmup_steps" : 20, "step_size" : 100, "gamma" : 0.9, "min_lr" : 1e-6,
    })
    loss_weight_params: Dict = field(default_factory=lambda: {
                    "strategy": "gradnorm", "lr": 0.025, "alpha": 0.5, "gradnorm_layer" : -2, "T" : 3.0, "update_interval": 5,
                    "loss_weights":{'mse': 1.0, 'distribution': 0.0, 'diffusion': 0.0, "kl_div": 0.2},
                    "loss_schedule" : {'mse': [0,100], 'distribution': [0,100], 'diffusion': [0,100], "kl_div": [0,100]}
                    })
    clip_gradients: bool = True
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"
    max_T_sample: int = 100
    push_to_hub: bool = False
    # distribution loss params
    diffusion_strategy: str = "1d-encode-decode"
    diffusion_image_loss: str = "1d-mse"
    diffusion_loss_noise_level: int = 10; 
    diffusion_loss_decoding_stride: int = 1
    distloss_type: str = "ksd"
    num_gaussians: List[int] = field(default_factory=lambda: [3, 2, 3, 2])
    num_distloss_samples: int = 8
    distloss_bs: int = 1152 # 384 * 8
    distloss_var_inds: List[int] = field(default_factory=lambda: [68, 60, 73, 82])
    distloss_var_sel: str = 'uniform'

@dataclass
class UNetParams:
    sample_size: Tuple[int, int] = field(default_factory=lambda: (16, 24))
    in_channels: int = 128
    out_channels: int = 128
    extra_in_channels: int = 0
    block_out_channels: Tuple = field(default_factory=lambda: (32, 64, 64, 128))  # num output channel for each UNet block
    down_block_types: Tuple = field(default_factory=lambda: (
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ))
    up_block_types: Tuple = field(default_factory=lambda: (
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
    ))
    layers_per_block: int = 1
    norm_num_groups: int = 2
    act_fn: str = "silu" # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py#L27
    freq_shift: float = 0.0 # fourier freq shift
    use_timestep_embedding: bool = True

# scheduler params
@dataclass
class SchedulerParams:
    num_train_timesteps: int = 100
    beta_schedule: str = 'linear'
    clip_sample: bool = False
    clip_sample_range: float = 4.0
    beta_end: float = 0.02
    prediction_type: str = 'epsilon'

@dataclass
class ModelConfig:
    model_type: str = "ddpm_diffusion"
    scheduler_type: str = 'ddpm'
    unet: UNetParams = field(default_factory=lambda: UNetParams())
    scheduler: SchedulerParams = field(default_factory=lambda:SchedulerParams())
    scheduler_inference_steps: int = 100
    # VAE params
    num_channels: int = 128
    latent_dims: int = 16
    ae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    disable_enc_logstd_bias: bool = True
    # Baseline Model Params
    bl_model_dir: str = "/mnt/home/ssa2206/Climsim/climsim-online/storage/shared_e3sm/saved_models/wrapper"
    bl_load_model_name: str = None
    bl_input_size: int = 124
    bl_output_size: int = 128
    bl_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256]) 
    def __post_init__(self):
        if isinstance(self.unet, dict):
            self.unet = UNetParams(**self.unet)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerParams(**self.scheduler)  


class ModelLens:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.params = dict(model.named_parameters())
        self.param_names = [p for p in self.params]
    
    def get_param(self, getter):
        if isinstance(getter, str):
            return self.params[getter]
        elif isinstance(getter, int):
            return self.params[self.param_names[getter]]
        raise ValueError(f"Invalid getter: {getter}")
    
    def log_gradients(self, gdict):
        for n, p in self.params.items():
            if p.grad is not None and torch.isfinite(p.grad).all():
                gdict.setdefault(n, []).append((p.grad.mean().item(), p.grad.std().item()))
            else:
                return( dict(param_name=n, grad=p.grad, state_dict={k: v.clone().cpu() for k, v in self.params.items()}) )
        return(0)
    
    def __getattr__(self, name):
        return getattr(self.model, name)
    def __call__(self, *args):
        return self.model(*args)

    def __repr__(self):
        return repr(self.model)
    def __str__(self):
        return str(self.model)

def load_config(fname, expid, base_dir="experiments/"):
    if 'json' not in fname:
        fname += ".json"
    with open(os.path.join(base_dir, expid, fname), 'r') as f:
        cdict = json.load(f)
    try:
        tconfig = TrainingConfig(**cdict['training_config'])
    except:
        print("mismatch between tconfig and class")
        tconfig = cdict['training_config']
    try:
        mconfig = ModelConfig(**cdict['model_config'])
    except:
        print("mismatch between mconfig and class")
        mconfig = cdict['model_config']
    try:
        dconfig = DataConfig(**cdict['data_config'])
    except:
        print("mismatch between dconfig and class")
        dconfig = cdict['data_config']

    return(tconfig, mconfig, dconfig)

def load_model_from_ckpt(ckpt_path, mconfig, baseline=False):
    if(isinstance(mconfig, dict)):
        mconfig = ModelConfig(**mconfig)
    mconfig.model_type = "baseline" if baseline else mconfig.model_type
    model = load_model(mconfig)
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True))
    return(model)


model_table = {
    'best_diffusion_2d' : ('diffusion_hp_search', 'lr-explore', 'lr-explorea', "best"),
    'vintage_diffusion_2d' : ( "full_dataset_testrun" , 'trial_1b', 'trial_1b', ""),
    'diff_1d_v2' : ('diffusion_hp_search', 'diff_1d', 'diff_1da', ""),
    'diff_1d_v1' : ('diffusion_hp_search', 'diff_1d_v1', 'diff_1d_v1a', "best-"),
}

def load_diffusion_model(model_id='best_diffusion_2d', base_dir="/mnt/home/ssa2206/Climsim/experiments"):
    exp_id, log_id, run_id, cid = model_table[model_id]
    log_dict_path = os.path.join(base_dir, exp_id, f"{log_id}.json")
    with open(log_dict_path, 'r') as file:
        diff_logs = json.load(file)
    mconfig = diff_logs[run_id]['model_config']
    ckpt = os.path.join(base_dir, exp_id, "checkpoints", f"{cid}{run_id}-ckpt.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(base_dir, exp_id, "checkpoints", log_id,f"{cid}{run_id}-ckpt.pt")
    model = tru.load_model_from_ckpt(ckpt, mconfig)
    return(model)

def image_loss(tconfig):
    # return loss function L : denoised_image, og_image --> loss
    return torch.nn.MSELoss()

def load_lr_scheduler(tconfig, optim, dataloader, num_epochs=10):
    total_steps = len(dataloader) * num_epochs
    params = tconfig.learning_rate_params
    match params['lr_scheduler']:
        # -------------------- HuggingFace/Diffusers schedulers --------------------
        case "cosine":
            return diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=optim, 
                num_warmup_steps=params.get("lr_warmup_steps", 0), 
                num_training_steps=total_steps,
                num_cycles=params.get("cycles", 0.5)
            )
        case "linear":
            return diffusers.optimization.get_linear_schedule_with_warmup(
                optimizer=optim,
                num_warmup_steps=params.get("lr_warmup_steps", 0),
                num_training_steps=total_steps,
            )

        # -------------------- PyTorch built-in schedulers --------------------
        case "steplr":
            return torch.optim.lr_scheduler.StepLR(
                optimizer=optim,
                step_size=params.get("step_size", 10),
                gamma=params.get("gamma", 0.9)
            )
        case "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optim,
                gamma=params.get("gamma", 0.95)
            )
        case "cosineanneal":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optim,
                T_max=total_steps,
                eta_min=params.get("min_lr", 0)
            )
        case "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optim,
                mode=params.get("mode", "min"),
                factor=params.get("gamma", 0.75),
                patience=params.get("patience", 5),
                min_lr=params.get("min_lr", 0)
            )

        # -------------------- No scheduler --------------------
        case _:
            return None

 
def create_optimizer(model, tconfig):
    match tconfig.optimizer.lower():
        case "adam":
            my_betas = tconfig.learning_rate_params.get("betas", (0.9, 0.999))
            optim = torch.optim.Adam(model.parameters(), lr=tconfig.learning_rate_params.get("learning_rate"), betas=my_betas)

        case _: # defaults to SGD
            optim = torch.optim.SGD(model.parameters(), lr=tconfig.learning_rate_params.get("learning_rate"))
    return(optim)

def unnormalize_npy(X_norm, Y_norm, data_vars='v1'):
    inputs, outputs = load_vars(data_vars)
    input_mean = xr.open_dataset('Climsim_info/input_mean.nc')[inputs].to_stacked_array('mlvar', sample_dims='').values
    input_max = xr.open_dataset('Climsim_info/input_max.nc')[inputs].to_stacked_array('mlvar', sample_dims='').values
    input_min = xr.open_dataset('Climsim_info/input_min.nc')[inputs].to_stacked_array('mlvar', sample_dims='').values
    output_scale = xr.open_dataset('Climsim_info/output_scale.nc')[outputs].to_stacked_array('mlvar', sample_dims='').values

    X = X_norm*(input_max - input_min) + input_mean
    Y = Y_norm / output_scale
    return(X,Y)


def create_sample(data, ds):
    # mimics get_item from loaded xarray subsample
    if not (hasattr(ds, "Y_mean") or hasattr(ds, "Ymean")):
        raise ValueError("ds does not have Y_mean or Ymean")
    ym = ds.Y_mean if hasattr(ds, "Y_mean") else ds.Ymean
    ys = ds.Y_std if hasattr(ds, "Y_std") else ds.Ystd
    if "ncol" in ym.dims:
        ym = ym.mean(dim='ncol')
        ys = ys.mean(dim='ncol')
    data = ((data-ym)/ys).isel(ncol=ds.permute_indices)
    data = data.to_stacked_array(new_dim="mlo", sample_dims=("time", "ncol"))
    data = data.transpose("time", "mlo", "ncol").load()
    data = torch.tensor(data.data.reshape(-1, 128, 16, 24), dtype=torch.float32)
    return data

def recreate_sample(sample, dataset):
    if(isinstance(sample, torch.Tensor)):
        if(sample.device != 'cpu'):
            sample = sample.detach().cpu()
        sample = sample.numpy()
    
    xarr = xr.DataArray(sample.reshape(-1, 128, 384), dims="time mlo ncol".split(), coords=dict(
        time = np.arange(sample.shape[0]),
        mlo = dataset.mlo, 
        ncol = dataset.permute_indices
    )).isel(ncol=np.argsort(dataset.permute_indices))
    if "ncol" in dataset.Ymean.dims:
        mu_stack = dataset.Ymean.to_stacked_array(new_dim="mlo", sample_dims=('ncol',))
        sig_stack = dataset.Ystd.to_stacked_array(new_dim="mlo", sample_dims=('ncol',))
        xrec = (xarr * sig_stack.mean(dim='ncol')) + mu_stack.mean(dim='ncol')
    else:
        mu_stack = dataset.Ymean.to_stacked_array(new_dim="mlo", sample_dims=())
        sig_stack = dataset.Ystd.to_stacked_array(new_dim="mlo", sample_dims=())
        xrec = (xarr * sig_stack) + mu_stack
    
    return(xrec.transpose("time", "ncol", "mlo"))