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
import torch
import pprint
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'

rank = 0
device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

REF_BATCH_SIZE = 128
dataset_type = "xbatch"
in_notebook = True


def setup_diffusion_run(num_models, exp_id, base_run_id, data_vars='v1', image_dim=1, batch_size=128, shuffle_indices=False, use_tendencies=False):
    dataset_type = "diffusion1d"
    dconfig = tru.my_dconfig("local-vzarr", data_vars, in_notebook, shuffle_indices, dataset_type, batch_size, use_tendencies)
    dconfig.train_test_split = [0.8]
    tconfigs, mconfigs = [], []

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
        tconfig.learning_rate = 1e-5 * batch_size / REF_BATCH_SIZE
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


if __name__ == "__main__":
    num_models = 4
    exp_id = "diffusion_hp_search"
    run_id = "diff_1d"
    base_dir = os.path.join("/mnt/home/ssa2206/Climsim/experiments", exp_id)
    tconfigs, mconfigs, dconfig = setup_diffusion_run(num_models, exp_id, run_id, data_vars = 'v2', batch_size = 256, use_tendencies=True)
    run_start_time = tru.log_event("run start", data_params = tru.asdict(dconfig.dataloader_params))
    t0 =  tru.log_event("setup start", run_id=run_id)
    #model = tru.load_model_from_ckpt("trial_1-ckpt.pt", mconfig, exp_id, EXP_DIR)
    loss_fn = torch.nn.MSELoss()
    dataloaders, indices = tru.load_dataloaders(dconfig, log=False)
    trainer = DiffusionTrainer(dataloaders, indices, mconfigs, tconfigs, base_dir, base_run_id=run_id)

    tru.log_event("setup end", duration=time.time() - t0)
    trainer.train(num_epochs=20, log=True)
    print("Done!")
    tru.log_event("run end", duration = time.time() - run_start_time)
