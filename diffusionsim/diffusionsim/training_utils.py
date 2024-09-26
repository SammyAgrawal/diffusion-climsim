import xarray as xr
import numpy as np
import pandas as pd
import numpy as np
import os

import gcsfs
import json
import torch

from .models import load_model
from .mydatasets import load_dataset, load_dataloader, load_scheduler
from .trainers import VAETrainer, DiffusionTrainer, create_optimizer
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple

def setup_trainer(exp_id, run_id, tconfig, mconfig, dconfig, exp_dir="./experiments"):
    from pathlib import Path
    base_dir = os.path.join(exp_dir, exp_id)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    tconfig.exp_id = exp_id
    with open(os.path.join(base_dir, f'{run_id}.json'), "w") as f:
        json.dump(dict(
            training_config=asdict(tconfig),
            model_config=asdict(mconfig),
            data_config=asdict(dconfig),
        ), f)
    model = load_model(mconfig)
    dataloader = load_dataloader(dconfig)
    optimizer = create_optimizer(model, tconfig)
    next(iter(dataloader)) # just to finish setting up
    match mconfig.model_type:
        case mtype if "diffusion" in mtype:
            loss_fn = torch.nn.MSELoss()
            scheduler = load_scheduler(mconfig)
            trainer = DiffusionTrainer(model, scheduler, dataloader, loss_fn, optimizer, tconfig, rank=0, base_dir=base_dir)

    return(trainer)

@dataclass
class TrainLoaderParams:
    batch_size: int = 128
    shuffle: bool = False
    num_workers: int = 0
    prefetch_factor: int = None
    persistent_workers: bool = False
    multiprocessing_context: str = None
    pin_memory: bool = True

@dataclass
class DataConfig:
    dataset_type: str = "XBatchDataset"
    climsim_type: str = "low-res"
    source = "gcsfs"
    train_test_split: List[int] = field(default_factory=lambda: [1.0, 0.0])
    dataloader_params: TrainLoaderParams = field(default_factory=lambda: TrainLoaderParams())
    xarr_subsamples: Tuple[int, int, int] = (36,210240, 144)
    data_vars: str = "v1"
    use_tendencies: bool = False
    norm_info: str = "image"
    prenormalize: bool = False
    chunksize: Dict = field(default_factory=lambda:{})
    def __post_init__(self):
        if isinstance(self.dataloader_params, dict):
            self.dataloader_params = TrainLoaderParams(**self.dataloader_params)
            
@dataclass
class TrainingConfig:
    exp_id: str = "expName"
    # data params
    num_epochs: int = 5
    phases: List[str] = field(default_factory=lambda: ['train', 'eval'])
    distributed_training: bool = False
    # learning parameters
    optimizer: str = 'adam'
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler: str = None
    learning_rate: float = 1e-4
    beta: float = 0.2 # VAE Kl div beta
    clip_gradients: bool = True
    gradient_accumulation_steps = 1
    lr_warmup_steps = 500
    mixed_precision = "fp16"
    # logging params
    save_best_epoch: bool = True
    batch_logging_interval: int = 4
    batch_checkpoint_interval: int = 100 # save checkpoint every 100 batches
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
    data_vars: str = "v1"
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

def load_model_from_ckpt(ckpt_fname, mconfig, expid, base_dir="experiments/"):
    model = load_model(mconfig)
    cpath = os.path.join(base_dir, expid, ckpt_fname)
    model.load_state_dict(torch.load(cpath, map_location=torch.device('cpu')))
    return(model)

#def load_lr_scheduler(config, optim, dataloader):
#    lr = get_cosine_schedule_with_warmup(
#        optimizer=optim, 
#        num_warmup_steps=config.lr_warmup_steps, 
#        num_training_steps=len(dataloader) * config.num_epochs,
#    )
#    return(lr)

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

