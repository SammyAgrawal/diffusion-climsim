import xarray as xr
import numpy as np
import pandas as pd
import numpy as np
import os

import gcsfs
import json
import torch

from .models import load_model, build_baseline_model
from .mydatasets import load_dataset, load_dataloaders, load_scheduler, log_event
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple


@dataclass
class TrainLoaderParams:
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
    climsim_type: str = "expanded-low-res"
    source: str = "gcsfs"
    data_dir: str = "/mnt/home/ssa2206/Climsim/diffusion-climsim/data/local_manifests"
    train_test_split: List[int] = field(default_factory=lambda: [1.0, 0.0])
    dataloader_params: TrainLoaderParams = field(default_factory=lambda: TrainLoaderParams())
    xarr_subsamples: Tuple[int, int, int] = (36,210240, 144)
    data_vars: str = "v1"
    use_tendencies: bool = False
    norm_info: str = "image"
    prenormalize: bool = False
    chunksize: Dict = field(default_factory=lambda:{})
    log_batching: bool = True
    def __post_init__(self):
        if isinstance(self.dataloader_params, dict):
            self.dataloader_params = TrainLoaderParams(**self.dataloader_params)

def my_dconfig(data_vars='v1', in_notebook=False, climsim_training=True, batch_size=128):
    dl_params = TrainLoaderParams()
    dl_params.batch_size = batch_size
    if(climsim_training):
        dl_params.batch_size *= 384
    dl_params.shuffle = True
    if(torch.cuda.is_available() and batch_size > 16):
        dl_params.pin_memory = True
    if(not in_notebook):
        dl_params.num_workers = 4
        dl_params.prefetch_factor = 3
        dl_params.persistent_workers = True
        dl_params.multiprocessing_context = "forkserver"
    
    dconfig = DataConfig()

    dconfig.dataloader_params = dl_params
    dconfig.source = "local-vzarr" # specify from raw cloud bucket
    dconfig.climsim_type = "low-res-expanded" 
    dconfig.dataset_type = "climsim" if climsim_training else "xbatch"
    dconfig.data_dir = "/mnt/home/ssa2206/Climsim/diffusion-climsim/data/local_manifests"
    dconfig.train_test_split = [0.35, 0.05] if climsim_training else [1.0]
    dconfig.data_vars = data_vars
    return(dconfig)
            
@dataclass
class TrainingConfig:
    exp_id: str
    run_id: str
    # data params
    num_epochs: int = 5
    phases: List[str] = field(default_factory=lambda: ['train', 'eval'])
    distributed_training: bool = False
    # learning parameters
    optimizer: str = 'adam'
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler: str = None
    learning_rate: float = 1e-4
    loss_weights: Dict = field(default_factory=lambda: {'mse': 1.0, 'distribution': 0.0, 'diffusion': 0.0, "kl_div": 0.2})
    clip_gradients: bool = True
    gradient_accumulation_steps = 1
    lr_warmup_steps = 500
    mixed_precision = "fp16"
    max_T_sample: int = 100
    # logging params
    save_best_epoch: bool = True
    batch_logging_interval: int = 32
    batch_checkpoint_interval: int = 50 # save checkpoint every 50 batches
    log_gradients: bool = False
    #save_image_epochs: int = 2
    push_to_hub: bool = False
    diffusion_loss_noise_level: int = 10; 

    # distribution loss params
    distloss_type: str = "ksd"
    num_gaussians: int = 2
    num_distloss_samples: int = 5
    distloss_bs: int = 3072 # 384 * 8
    distloss_var_ind: int = 68
    def __post_init__(self):
        self.shuffle_data = {'train':False, 'eval':False}

def my_tconfig(climsim_training, batch_size, max_T_sample=51, lr=1e-4):
    tconfig = TrainingConfig()
    tconfig.exp_id = exp_id
    tconfig.num_epochs = 10
    #tconfig.lr_scheduler = 'get_cosine_schedule_with_warmup'
    #tconfig.lr_warmup_steps = 100
    ref_batch_size = 128
    tconfig.learning_rate = lr * batch_size / ref_batch_size
    tconfig.loss_weights = {'mse': 1.0, 'distribution': 0.0, 'diffusion': 0.0}
    tconfig.max_T_sample = max_T_sample

@dataclass
class UNetParams:
    sample_size: Tuple[int, int] = field(default_factory=lambda: (16, 24))
    in_channels: int = 128
    out_channels: int = 128
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

# scheduler params
@dataclass
class SchedulerParams:
    num_train_timesteps: int = 100
    beta_schedule: str = 'linear'
    clip_sample: bool = False
    clip_sample_range: float = 4.0
    beta_end: float = 0.02
    beta_schedule: str = 'linear'

@dataclass
class ModelConfig:
    model_type: str = "ddpm_diffusion"
    data_vars: str = "v1"
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
    bl_model_dir: str = "/mnt/home/ssa2206/Climsim/saved_models/"
    bl_load_model_name: str = None
    bl_input_size: int = 124
    bl_output_size: int = 128
    bl_num_layers: int = 3
    bl_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256]) 
    def __post_init__(self):
        if isinstance(self.unet, dict):
            self.unet = UNetParams(**self.unet)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerParams(**self.scheduler)  


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

def load_model_from_ckpt(ckpt_fname, mconfig, expid, exp_dir, baseline=False):
    if(isinstance(mconfig, dict)):
        mconfig = ModelConfig(**mconfig)
    model_type = "baseline" if baseline else mconfig.model_type
    model = load_model(mconfig, model_type)
    cpath = os.path.join(exp_dir, expid, ckpt_fname)
    model.load_state_dict(torch.load(cpath, map_location=torch.device('cpu')))
    return(model)

#def load_lr_scheduler(config, optim, dataloader):
#    lr = get_cosine_schedule_with_warmup(
#        optimizer=optim, 
#        num_warmup_steps=config.lr_warmup_steps, 
#        num_training_steps=len(dataloader) * config.num_epochs,
#    )
#    return(lr)
 
def create_optimizer(model, tconfig):
    match tconfig.optimizer.lower():
        case "adam":
            try:
                my_betas = tconfig.betas
            except AttributeError:
                my_betas = (0.9, 0.999) # default values
            optim = torch.optim.Adam(model.parameters(), lr=tconfig.learning_rate, betas=my_betas)

        case _: # defaults to SGD
            optim = torch.optim.SGD(model.parameters(), lr=tconfig.learning_rate)
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
