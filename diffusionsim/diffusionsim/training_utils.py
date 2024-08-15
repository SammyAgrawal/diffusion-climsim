import xarray as xr
import numpy as np
import pandas as pd
import numpy as np
import os

import gcsfs
import datetime as dt
import json
import torch

from .models import load_model
from .mydatasets import ClimsimDataset, ClimsimImageDataset

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple

@dataclass
class TrainingConfig:
    # data params
    train_test_split: List[int] = field(default_factory=lambda: [0.75, 0.25])
    phases: List[str] = field(default_factory=lambda: ['train', 'eval'])
    num_epochs: int = 5
    distributed_training: bool = False
    shuffle_data: Dict[str, bool] = None
    batch_size: int = 32
    xarr_subsamples: Tuple[int, int, int] = (36,210240, 144)
    # learning parameters
    optimizer: str = 'adam'
    lr_scheduler: str = None
    learning_rate: float = 1e-4
    beta: int = 0.2
    clip_gradients: bool = True
    gradient_accumulation_steps = 1
    lr_warmup_steps = 500
    mixed_precision = "fp16"
    # logging params
    save_best_epoch: bool = True
    batch_logging_interval: int = 4
    model_checkpoint_interval: int = 2 # save checkpoint every 2 epochs
    log_gradients: bool = True
    #save_image_epochs: int = 2
    push_to_hub: bool = False
    
    def __post_init__(self):
        self.shuffle_data = {'train':False, 'eval':False}

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
    norm_num_groups: int = 4

# scheduler params
@dataclass
class SchedulerParams:
    num_train_timesteps: int = 100
    beta_schedule: str = 'linear'
    clip_sample: bool = False
    clip_sample_range: float = 4.0


@dataclass
class ModelConfig:
    model_type: str = "diffusion"
    scheduler_type: str = 'ddpm'
    unet: UNetParams = field(default_factory=lambda: UNetParams())
    scheduler: SchedulerParams = field(default_factory=lambda:SchedulerParams())
    # VAE params
    num_channels: int = 128
    latent_dims: int = 16
    ae_hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    disable_enc_logstd_bias: bool = True
    def __post_init__(self):
        if isinstance(self.unet, dict):
            self.unet = UNetParams(**self.unet)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerParams(**self.scheduler)  



def fetch_config(fpath):
    with open(fpath, 'r') as f:
        config_dict = json.load(f)
    return(TrainingConfig(**config_dict))


os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'
def train_test_split(Xarr, Yarr, split_frac=[0.75, 0.25]):
    datasets = []
    num_samples = Yarr.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    counter = 0
    for frac in split_frac:
        # Calculate the split index
        split = int(num_samples * frac)
        phase_indices = indices[counter:counter+split]
        datasets.append((Xarr[phase_indices], Yarr[phase_indices]))
        counter += split

    return(datasets)
 

fs = gcsfs.GCSFileSystem()

def unnormalize_npy(X_norm, Y_norm, data_vars='v1'):
    inputs, outputs = load_vars(data_vars)
    input_mean = xr.open_dataset('Climsim_info/input_mean.nc')[inputs].to_stacked_array('mlvar', sample_dims='').values
    input_max = xr.open_dataset('Climsim_info/input_max.nc')[inputs].to_stacked_array('mlvar', sample_dims='').values
    input_min = xr.open_dataset('Climsim_info/input_min.nc')[inputs].to_stacked_array('mlvar', sample_dims='').values
    output_scale = xr.open_dataset('Climsim_info/output_scale.nc')[outputs].to_stacked_array('mlvar', sample_dims='').values

    X = X_norm*(input_max - input_min) + input_mean
    Y = Y_norm / output_scale
    return(X,Y)

def process_climsim_old(ds_input, ds_output, data_vars='v1', downsample=True, chunks=True):
    if(type(data_vars) == str):
        input_vars, output_vars = load_vars(data_vars)
    else:
        input_vars, output_vars = data_vars # (assume tuple)
    
    for var in output_vars:
        if('ptend' in var and var not in ds_output.data_vars): # each timestep is 20 minutes which corresponds to 1200 seconds
            v = var.replace("ptend", "state")
            ds_output[var] = (ds_output[v] - ds_input[v]) / 1200
    
    if downsample: # might as well do first
        N_samples = len(ds_input.sample)
        ds_input = ds_input.isel(sample = np.arange(36,N_samples,72)) #  every 1 day
        ds_output = ds_output.isel(sample = np.arange(36,N_samples,72))

    # reformat, add time dimension
    time = pd.DataFrame({"ymd":ds_input.ymd, "tod":ds_input.tod})
    ds_input = ds_input[input_vars]
    ds_output = ds_output[output_vars]
    f = lambda ymd, tod : cftime.DatetimeNoLeap(ymd//10000, ymd%10000//100, ymd%10000%100, tod // 3600, tod%3600 // 60)
    time = list(time.apply(lambda x: f(x.ymd, x.tod), axis=1))
    print(f"Computed time {len(time)}")
    # Load spatial latlon info
    mapper = fs.get_mapper("gs://leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.grid-info.zarr")
    ds_grid = xr.open_dataset(mapper, engine='zarr')
    lat = ds_grid.lat.values.round(2) 
    lon = ds_grid.lon.values.round(2)  
    lon = ((lon + 180) % 360) - 180 # convert from 0-360 to -180 to 180
    def add_timelatlon(ds):
        ds['sample'] = time
        ds = ds.rename({'sample':'time'})
        ds = ds.assign_coords({'ncol' : ds.ncol})
        ds['lat'] = (('ncol'),lat.T)
        ds['lon'] = (('ncol'),lon.T)
        ds = ds.assign_coords({'lat' : ds.lat, 'lon' : ds.lon})
        return(ds)
    return(add_timelatlon(ds_input), add_timelatlon(ds_output))
