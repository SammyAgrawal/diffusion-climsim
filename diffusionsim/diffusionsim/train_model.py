import time
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from climsim_training_utils import fetch_config, load_training_objects
from .mydatasets import load_numpy_arrays, reconstruct_xarr_from_npy
import xarray as xr

def main(rank: int, world_size: int, port: int, config: TrainingConfig, Xarr: xr.core.dataarray.DataArray, Yarr: xr.core.dataarray.DataArray):
    # set up ddp
    device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    training_objs = load_training_objects(config)
    # train_data = prepare_dataloader(dataset, batch_size)
    trainer = DistributedVAETrainer(rank, gpu_id, training_objects, config, exp_id='VAE_distributed'):
    trainer.train()
    dist.destroy_process_group()


if __name__ == '__main__':
    import sys
    
    assert len(sys.argv) == 3, "Run as DistributedTrainer <path_to_config.json> <master port>"
    world_size = torch.cuda.device_count()
    config = fetch_config(sys.argv[1])
    port = int(sys.argv[2])
    
    print("Loading Data once, on main device")
    X, Y = load_numpy_arrays(config.data_paths)
    Xarr, Yarr = reconstruct_xarr_from_npy(X, Y, subsampling=config.xarr_subsamples, data_vars=config.data_vars)
    
    mp.spawn(main, args=(world_size, port, config, Xarr, Yarr), nprocs=world_size)
    