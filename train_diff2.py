import os
import sys
sys.path.append(os.path.abspath(os.path.join('diffusionsim')))
from tqdm import tqdm
from pathlib import Path
import time
import json
import diffusers
import diffusionsim as diff
import diffusionsim.training_utils as tru
import torch
from typing import Optional
import typer
from dataclasses import dataclass, asdict, field
from typing_extensions import Annotated
import pprint
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'

#rank = 0
#device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
#print(f"Using device: {device}")    

def define_configs(exp_id, climsim_training=True, in_notebook=True, data_vars='v1', lr=1e-4):
    dl_params = tru.TrainLoaderParams()
    dl_params.batch_size = 32
    if(climsim_training):
        dl_params.batch_size *= 384
    dl_params.shuffle = True
    dl_params.pin_memory = True
    if(not in_notebook):
        dl_params.num_workers = 4
        dl_params.prefetch_factor = 3
        dl_params.persistent_workers = True
        dl_params.multiprocessing_context = "forkserver"
    
    dconfig = tru.DataConfig()
    dconfig.dataloader_params = dl_params
    dconfig.source = "local-vzarr" # specify from raw cloud bucket
    dconfig.climsim_type = "low-res-expanded" 
    dconfig.dataset_type = "climsim" if climsim_training else "xbatch"
    dconfig.data_dir = "/mnt/home/ssa2206/Climsim/diffusion-climsim/data/local_manifests"
    dconfig.train_test_split = [1.0]
    dconfig.data_vars = data_vars

    tconfig = tru.TrainingConfig()
    tconfig.exp_id = exp_id
    tconfig.num_epochs = 5
    tconfig.phases = ['train', 'eval']
    #tconfig.lr_scheduler = 'get_cosine_schedule_with_warmup'
    #tconfig.lr_warmup_steps = 100
    ref_batch_size = 128*384 if climsim_training else 128
    tconfig.learning_rate = lr * dconfig.dataloader_params.batch_size / ref_batch_size
    tconfig.batch_logging_interval = 32
    tconfig.batch_checkpoint_interval = 50
    tconfig.save_best_epoch = True
    tconfig.log_gradients = False
    tconfig.loss_weights = {'mse': 1.0, 'distribution': 0.0, 'diffusion': 0.0}
    tconfig.max_T_sample = 51

    unet = tru.UNetParams()
    unet.block_out_channels = (128, 256, 512) if data_vars == "v1" else (256, 512, 1024)
    unet.down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D")
    unet.up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D")
    unet.layers_per_block = 1
    unet.norm_num_groups = 2
    unet.in_channels = 128 if data_vars == "v1" else 368
    unet.out_channels = unet.in_channels

    mconfig = tru.ModelConfig()
    mconfig.model_type = "ddpm_diffusion"
    mconfig.unet = unet
    mconfig.scheduler = tru.SchedulerParams()
    # define baseline model
    mconfig.bl_hidden_dims = [256, 256]
    mconfig.bl_num_layers = 2

    return(tconfig, mconfig, dconfig)

def setup_configs(exp_id, run_id, exp_dir, lr=3e-5):
    from pathlib import Path
    base_dir = os.path.join(exp_dir, exp_id)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    tconfig, mconfig, dconfig = define_configs(exp_id, climsim_training=False, in_notebook=False, "v1", lr=lr)
    tconfig.exp_id = exp_id
    with open(os.path.join(base_dir, f'{run_id}.json'), "w") as f:
        json.dump(dict(
            training_config=asdict(tconfig),
            model_config=asdict(mconfig),
            data_config=asdict(dconfig),
        ), f)
    return(base_dir, tconfig, mconfig, dconfig)


def main(rank: int, world_size: int, port: int, config: TrainingConfig, Xarr: xr.core.dataarray.DataArray, Yarr: xr.core.dataarray.DataArray):
    # set up ddp
    device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    model = tru.load_model(mconfig)
    dataloaders, indices = tru.load_dataloaders(dconfig)
    optimizer = tru.create_optimizer(model, tconfig)
    loss_fn = torch.nn.MSELoss()
    scheduler = tru.load_scheduler(mconfig)

    trainer = tru.DiffusionTrainer(model, scheduler, dataloaders, loss_fn, optimizer, tconfig, base_dir=base_dir, rank=0)
    # train_data = prepare_dataloader(dataset, batch_size)
    #trainer = DistributedVAETrainer(rank, gpu_id, training_objects, config, exp_id='VAE_distributed'):
    trainer.train(num_epochs=20, log=True, run_id=run_id)
    dist.destroy_process_group()


if __name__ == '__main__':
    
    assert len(sys.argv) == 3, "Run as DistributedTrainer <path_to_config.json> <master port>"
    world_size = torch.cuda.device_count()
    config = fetch_config(sys.argv[1])
    port = int(sys.argv[2])

    exp_id = "empire_fullrun"
    run_id = "trial_1c"
    exp_dir = "/mnt/home/ssa2206/Climsim/experiments"
    base_dir, tconfig, mconfig, dconfig = setup_configs(exp_id, run_id, exp_dir)
    run_start_time = tru.log_event("run start", 
        data_params = asdict(dconfig.dataloader_params),
    )
    t0 =  tru.log_event("setup start", run_id=run_id)
    pprint.pprint(asdict(tconfig))
    print("\n\n")
    pprint.pprint(asdict(mconfig))
    print("\n\n" )
    pprint.pprint(asdict(dconfig))
    print("\n\n", )
    #model = tru.load_model_from_ckpt("trial_1-ckpt.pt", mconfig, exp_id, exp_dir)

    tru.log_event("setup end", duration=time.time() - t0)
    mp.spawn(main, args=(world_size, port, config, Xarr, Yarr), nprocs=world_size)
    print("Done!")
    tru.log_event("run end", duration = time.time() - run_start_time)
