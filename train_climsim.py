import os
import sys
from pathlib import Path
import time
import json
#import diffusers
import diffusionsim.training_utils as tru
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict, field
from diffusionsim.trainers import ClimsimTrainer
import pprint
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'

REF_BATCH_SIZE = 128
EXP_DIR = "/mnt/home/ssa2206/Climsim/experiments"
climsim_training = True
in_notebook = False


def setup_run(num_models, exp_id, base_run_id, data_vars='v1', batch_size=128):
    dconfig = tru.my_dconfig(data_vars, in_notebook, climsim_training, batch_size)
    dconfig.train_test_split = [0.01, 0.002]
    tconfigs, mconfigs = [], []

    learning_rates = [1e-4, 1e-4, 1e-4, 1e-4]
    distloss_weights = [1.0, 1.0, 2.0, 5.0]
    target_variables_distloss = [68, 60, 73, 82]
    num_gaussians = [3, 2, 3, 2]
    unet_channel_dims = []
    unet_down_block_types = []

    lettering = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(num_models):
        tconfig = tru.TrainingConfig(exp_id=exp_id, run_id=f"{base_run_id}{lettering[i]}")
        tconfig.learning_rate = learning_rates[i] * batch_size / REF_BATCH_SIZE
        tconfig.loss_weights = {'mse': 1.0, 'distribution': distloss_weights[i], 'diffusion': 0.0}
        tconfig.distloss_var_ind = target_variables_distloss[i]
        tconfig.num_gaussians = num_gaussians[i]

        tconfig.max_T_sample = 51
        tconfig.phases = ['train', 'eval']
        
        unet = tru.UNetParams()
        unet.block_out_channels = (128, 256, 512) if data_vars == "v1" else (256, 512, 1024)
        unet.down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D")
        unet.up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D")
        unet.in_channels = 128 if data_vars == "v1" else 368
        unet.out_channels = unet.in_channels

        scheduler = tru.SchedulerParams()
        mconfig = tru.ModelConfig(unet=unet, scheduler=scheduler)
        if(climsim_training):
            mconfig.model_type = "baseline"
        # define baseline model
        mconfig.bl_hidden_dims = [256, 256, 256] if data_vars == "v1" else [512, 256, 256]
        mconfig.bl_num_layers = len(mconfig.bl_hidden_dims)
        mconfig.bl_input_size = 124 if data_vars == "v1" else 557
        mconfig.bl_output_size = 128 if data_vars == "v1" else 368

        tconfigs.append(tconfig)
        mconfigs.append(mconfig)

    return(tconfigs, mconfigs, dconfig)



if __name__ == "__main__":
    #typer.run(main)
    #typer.run(test_args)
    exp_id = "DistLossTesting"
    run_id = "trial1"
    base_dir = os.path.join(EXP_DIR, exp_id)
    tconfigs, mconfigs, dconfig = setup_run(4, exp_id, run_id, data_vars='v1', batch_size=64)
    run_start_time = tru.log_event("run start", data_params = asdict(dconfig.dataloader_params))
    t0 = tru.log_event("setup start", run_id=run_id)
    
    #unet = tru.load_model(mconfigs[0], "diffusion")
    #scheduler = tru.load_scheduler(mconfig)
    trainer = ClimsimTrainer(dconfig, mconfigs, tconfigs, torch.nn.MSELoss(), base_dir, rank=0)


    tru.log_event("setup end", duration=time.time() - t0)
    trainer.train(num_epochs=10, log=True)
    print("Done!")
    tru.log_event("run end", duration = time.time() - run_start_time)