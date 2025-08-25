import os
import sys
import signal
from pathlib import Path
import time
import json
#import diffusers
sys.path.append(os.path.abspath("diffusionsim"))
import diffusionsim.training_utils as tru
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict, field
from diffusionsim.trainers import ClimsimTrainer
import diffusionsim.evaluations as evals
import pprint

os.environ['WANDB_NAME'] = ""
os.environ['WANDB_NOTES'] = ""

num_epochs = 10
REF_BATCH_SIZE = 128
dataset_type="climsim"
in_notebook = True
torch.set_grad_enabled(True)


def setup_climsim_run(num_models, exp_id, base_run_id, data_vars='v1', batch_size=128, use_tendencies=True, shuffle_indices=False):
    dataset_type = "climsim"
    #dconfig = tru.my_dconfig("local_vzarr", "v1", in_notebook, False, "climsim")
    dconfig = tru.my_dconfig("local-vzarr", data_vars, in_notebook, shuffle_indices,
                                dataset_type, batch_size, use_tendencies,
                             )
    dconfig.batch_logging_interval = 1
    dconfig.train_test_split = [0.002, 0.001]
    tconfigs, mconfigs = [], []

    learning_rates = [1e-3, 1e-3, 1e-3, 1e-3] + [1e-3] * max(0, num_models - 4)
    mse_weights = [1.0,] * num_models
    distloss_weights = [0.1,] * num_models
    diffloss_weights = [20.0,] * num_models

    image_dim = 1
    target_variables_distloss = [68, 73, 82] # previously included 60 asw
    num_gaussians = [3, 2, 3, 2]

    lettering = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(num_models):
        tconfig = tru.TrainingConfig(exp_id=exp_id, run_id=f"{base_run_id}{lettering[i]}")
        tconfig.learning_rate_params.update(dict(
            learning_rate = learning_rates[i] * batch_size / REF_BATCH_SIZE,
            lr_scheduler="steplr",
            step_size=500,
            gamma=0.95,
            lr_warmup_steps=20,
        ))

        loss_weights = {'mse': mse_weights[i], 'distribution': distloss_weights[i], 'diffusion': diffloss_weights[i]}
        loss_schedule = {'mse': [0,100], 'distribution': [0,100], 'diffusion': [0,100]}
        tconfig.loss_weight_params.update(dict(
            loss_weights=loss_weights,
            loss_schedule=loss_schedule,
            alpha=0.5, 
            strategy="gradnorm",
            update_interval=10,
        ))

        tconfig.clip_gradients = True
        tconfig.distloss_var_inds = target_variables_distloss
        tconfig.num_gaussians = num_gaussians[:len(target_variables_distloss)]
        
        tconfig.diffusion_image_dim = image_dim
        tconfig.diffusion_loss_noise_level = 10
        tconfig.diffusion_strategy = "1d-encode-decode"
        #tconfig.diffusion_image_loss = "1d-mse"
        tconfigs.append(tconfig)
        
        mconfig = tru.ModelConfig(scheduler=tru.SchedulerParams(
            prediction_type="epsilon",
            beta_schedule="linear",
        ))
        if("climsim" in dataset_type):
            mconfig.model_type = "baseline"
        # define baseline model
        mconfig.bl_hidden_dims = [256, 256, 256] if data_vars == "v1" else [512, 256, 256]
        mconfig.bl_input_size = 124 if data_vars == "v1" else 557
        mconfig.bl_output_size = 128 if data_vars == "v1" else 368
        mconfigs.append(mconfig)

    return(tconfigs, mconfigs, dconfig)


trainer = None

NUM_MODELS = 4
RESTART_FROM_CKPT = False
RERUN = False


EXP_DIR = "/mnt/home/ssa2206/Climsim/experiments"
if __name__ == "__main__":
    #typer.run(main)
    #typer.run(test_args)
    exp_id = "MultiTask"
    run_id = "lr-sweep2"
    run_start_time = tru.log_event("run start", exp_id=exp_id, run_id=run_id)
    t0 = tru.log_event("setup start")
    if(RERUN):
        run = evals.Run(exp_id, run_id)
        trainer = run.reconstruct_trainer(apply_checkpoints=False, apply_indices=False)
    else:
        base_dir = os.path.join(EXP_DIR, exp_id)
        tconfigs, mconfigs, dconfig = setup_climsim_run(NUM_MODELS, exp_id, run_id, data_vars='v1', batch_size=256)
        dataloaders, indices = tru.load_dataloaders(dconfig, log=True)
        trainer = ClimsimTrainer(dataloaders, indices, mconfigs, tconfigs, base_dir, run_id)
    
    tru.log_event("setup end", duration=time.time() - t0)
    if (RESTART_FROM_CKPT):
        trainer.restart_from_ckpt(num_epochs, log=True, cid='')
    else:
        trainer.train(num_epochs, log=True)
    
    print("Done!", flush=True)
    tru.log_event("run end", duration = time.time() - run_start_time)
