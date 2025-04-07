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
from diffusionsim.trainers import DiffusionTrainer
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

REF_BATCH_SIZE = 128
EXP_DIR = "/mnt/home/ssa2206/Climsim/experiments"
climsim_training = False
in_notebook = False

def setup_run(num_models, exp_id, base_run_id,
              data_vars='v1', batch_size=128):
    dconfig = tru.my_dconfig(data_vars, in_notebook, climsim_training, batch_size)
    dconfig.train_test_split = [1.0]
    tconfigs, mconfigs = [], []
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    unet_channel_dims = []
    unet_down_block_types = []
    lettering = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(num_models):
        tconfig = tru.TrainingConfig(exp_id=exp_id, run_id=f"{base_run_id}{lettering[i]}")
        tconfig.learning_rate = learning_rates[i] * batch_size / REF_BATCH_SIZE
        tconfig.phases = ['train']
        tconfig.loss_weights = {'mse': 1.0, 'distribution': 0.0, 'diffusion': 0.0}
        tconfig.max_T_sample = 51
        
        unet = tru.UNetParams()
        unet.block_out_channels = (128, 256, 512) if data_vars == "v1" else (256, 512, 1024)
        unet.down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D")
        unet.up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D")
        unet.in_channels = 128 if data_vars == "v1" else 368
        unet.out_channels = unet.in_channels

        scheduler = tru.SchedulerParams()
        mconfig = tru.ModelConfig(unet=unet, scheduler=scheduler)
        # define baseline model
        mconfig.bl_hidden_dims = [256, 256, 256] if data_vars == "v1" else [512, 256, 256]
        mconfig.bl_num_layers = len(mconfig.bl_hidden_dims)
        mconfig.bl_input_size = 124 if data_vars == "v1" else 557
        mconfig.bl_output_size = 128 if data_vars == "v1" else 368

        tconfigs.append(tconfig)
        mconfigs.append(mconfig)

    return(tconfigs, mconfigs, dconfig)

if __name__ == "__main__":
    num_models_to_train = 5
    exp_id = "empire_fullrun"
    run_id = "lr-search"
    base_dir = os.path.join(EXP_DIR, exp_id)
    tconfigs, mconfigs, dconfig = setup_run(num_models_to_train, exp_id, run_id)
    run_start_time = tru.log_event("run start", 
        data_params = asdict(dconfig.dataloader_params),
    )
    t0 =  tru.log_event("setup start", run_id=run_id)
    #model = tru.load_model_from_ckpt("trial_1-ckpt.pt", mconfig, exp_id, EXP_DIR)
    loss_fn = torch.nn.MSELoss()
    trainer = DiffusionTrainer(dconfig, mconfigs, tconfigs, loss_fn, base_dir, rank=0)

    tru.log_event("setup end", duration=time.time() - t0)
    trainer.train(num_epochs=20, log=True)
    print("Done!")
    tru.log_event("run end", duration = time.time() - run_start_time)