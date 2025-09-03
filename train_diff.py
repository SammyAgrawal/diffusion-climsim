import os
import sys
sys.path.append(os.path.abspath(os.path.join('diffusionsim')))
from tqdm import tqdm
from pathlib import Path
import re
import time
import diffusionsim as diff
import diffusionsim.training_utils as tru
from diffusionsim.trainers import DiffusionTrainer
import diffusionsim.evaluations as evals
import torch
import pprint

def setup_diffusion_run(
    num_models, exp_id, base_run_id, 
    data_vars, 
    image_dim,
    batch_size=128, 
    shuffle_indices=False, 
    use_tendencies=False
):
    dconfig = tru.my_dconfig("local-vzarr", data_vars, in_notebook, shuffle_indices, dataset_type, batch_size, use_tendencies)
    dconfig.train_test_split = [0.85]
    dconfig.phases = ['train']
    tconfigs, mconfigs = [], []
    dconfig.batch_checkpoint_interval = 32
    dconfig.batch_logging_interval = 128
    learning_rates = [1e-5, 1e-5, 1e-5, 1e-5]
    unet_layers_per_block = [1, 1, 2, 2, 1]
    block_out_channels = [
        (32, 32, 64),
        (16, 32, 48),
        (16, 32, 64),
        (32, 48, 64),
    ]
    

    lettering = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(num_models):
        tconfig = tru.TrainingConfig(exp_id=exp_id, run_id=f"{base_run_id}{lettering[i]}")
        tconfig.phases = ['train']
        tconfig.learning_rate = learning_rates[i] * batch_size / REF_BATCH_SIZE
        tconfig.lr_scheduler = "cosine"
        tconfig.lr_warmup_steps = 200
        tconfig.max_T_sample = 51
        tconfig.clip_gradients = True

        unet = tru.UNetParams()
        if image_dim == 1:
            unet.block_out_channels = block_out_channels[i] #if data_vars == "v1" else None
            unet.down_block_types = ("DownResnetBlock1D", "AttnDownBlock1D", "DownBlock1D") #if data_vars=="v1" else None
            unet.up_block_types = ("UpResnetBlock1D", "AttnUpBlock1D", "UpBlock1D") #if data_vars=="v1" else None
            unet.in_channels = 10 if data_vars == "v1" else 14
        else:
            unet.block_out_channels = (128, 256, 512) if data_vars == "v1" else (256, 512, 1024)
            unet.down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D")
            unet.up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D")
            unet.in_channels = 128 if data_vars == "v1" else 368
        unet.out_channels = unet.in_channels
        unet.layers_per_block = 1
        unet.norm_num_groups = 4

        scheduler = tru.SchedulerParams()
        scheduler.beta_schedule = "linear"
        scheduler.clip_sample = False
        scheduler.clip_sample_range = 4.0
        scheduler.beta_end = 0.02

        mconfig = tru.ModelConfig(unet=unet, scheduler=scheduler)
        mconfig.model_type = 'ddpm_diffusion1d'
        mconfig.data_vars = data_vars

        tconfigs.append(tconfig)
        mconfigs.append(mconfig)
    return(tconfigs, mconfigs, dconfig)

trainer = None
num_epochs = 15
REF_BATCH_SIZE = 128
dataset_type = "climsim_diffusion1d"
image_dim = 1
data_vars = 'v2'
in_notebook = True

NUM_MODELS = 4

RESTART_FROM_CKPT = False
RERUN = False
torch.set_grad_enabled(True)

EXP_DIR = "/mnt/home/ssa2206/Climsim/experiments"
if __name__ == "__main__":
    exp_id = "diffusion_hp_search"
    run_id = "diff_v2_1d"
    run_start_time = tru.log_event("run start", exp_id=exp_id, run_id=run_id)
    t0 =  tru.log_event("setup start", run_id=run_id)
    if(RERUN):
        run = evals.Run(exp_id, run_id, climsim_run=False, cid='')
        run.dconfig.dataloader_params.batch_size = 256
        trainer = run.reconstruct_trainer(apply_checkpoints=True, apply_indices=True)
    else:
        base_dir = os.path.join(EXP_DIR, exp_id)
        tconfigs, mconfigs, dconfig = setup_diffusion_run(
            NUM_MODELS, exp_id, run_id, data_vars, image_dim, batch_size=256, shuffle_indices=True, use_tendencies=True
        )
        dataloaders, indices = tru.load_dataloaders(dconfig, log=True)
        trainer = DiffusionTrainer(dataloaders, indices, mconfigs, tconfigs, base_dir, run_id)
    
    ckpt = "/mnt/home/ssa2206/Climsim/experiments/diffusion_hp_search/checkpoints/diff_1d/diff_1da-ckpt.pt"
    state = torch.load(ckpt, weights_only=True, map_location=trainer.device)
    trainer.models[trainer.run_ids[0]].load_state_dict(state)

    tru.log_event("setup end", duration=time.time() - t0)
    if (RESTART_FROM_CKPT):
        trainer.restart_from_ckpt(num_epochs, log=True, cid='')
    else:
        trainer.train(num_epochs, log=True)
    print("Done!", flush=True)
    tru.log_event("run end", duration = time.time() - run_start_time)
