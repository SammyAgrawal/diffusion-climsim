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

trainer = None
num_epochs = 10

REF_BATCH_SIZE = 128
dataset_type="climsim"
in_notebook = True

NUM_MODELS = 5
os.environ['WANDB_NAME'] = ""
os.environ['WANDB_NOTES'] = ""

RESTART_FROM_CKPT = False
RERUN = False
torch.set_grad_enabled(True)


def setup_climsim_run(num_models, exp_id, base_run_id, data_vars='v1', batch_size=128, shuffle_indices = True):
    dconfig = tru.my_dconfig("local-vzarr", data_vars, in_notebook, shuffle_indices, 
                             dataset_type, batch_size, use_tendencies=False)
    dconfig.train_test_split = [0.25, 0.10]
    dconfig.log_gradients = True
    dconfig.batch_checkpoint_interval = 50
    dconfig.batch_logging_interval = 16
    tconfigs, mconfigs = [], []

    learning_rates = [5e-5, 1e-4, 5e-4, 1e-3, 2.5e-3]
    #diffloss_weights = [5.0, 5.0, 5.0, 5.0]
    #distloss_weights = [0.1, 0.1, 0.1, 0.1]
    target_variables_distloss = [73, 59, 123]
    num_gaussians = [2, 2, 2]
    
    #run_ids = f"{base_run_id}-mse {base_run_id}-diff {base_run_id}-dist {base_run_id}-joint".split()
    lettering = "abcdefghijklmnopqrstuvwxyz"
    run_ids = [f"{base_run_id}-{lettering[i]}" for i in range(num_models)]
    for i in range(num_models):
        tconfig = tru.TrainingConfig(exp_id=exp_id, run_id=run_ids[i])
        tconfig.phases = ['train', 'eval']
        tconfig.learning_rate_params.update({
            "base_learning_rate" : learning_rates[i] * batch_size / REF_BATCH_SIZE,
            "step_size" : 500,
            "gamma" : 0.95,
            "scheduler_type" : "steplr",
        })
        tconfig.loss_weight_params.update({
            "strategy" : "gradnorm",
            "update_interval" : 10,
            "alpha" : 0.5,
            "weights" : {'mse': 1.0, 'distribution': 0.1, 'diffusion': 5.0},
            "schedule" : {'mse': [0,100], 'distribution': 2 , 'diffusion': 2}
        })
        tconfig.clip_gradients = False
        
        tconfig.distloss_var_inds = target_variables_distloss
        tconfig.num_gaussians = num_gaussians
        tconfig.distloss_var_sel = "uniform"

        tconfig.diffusion_loss_noise_level = 20
        tconfigs.append(tconfig)
        
        mconfig = tru.ModelConfig(unet=tru.UNetParams(), scheduler=tru.SchedulerParams())
        mconfig.data_vars = data_vars
        if("climsim" in dataset_type):
            mconfig.model_type = "baseline"
        # define baseline model
        mconfig.bl_hidden_dims = [256, 256, 256] if data_vars == "v1" else [512, 256, 256]
        mconfig.bl_input_size = 124 if data_vars == "v1" else 557
        mconfig.bl_output_size = 128 if data_vars == "v1" else 368
        mconfigs.append(mconfig)

    return(tconfigs, mconfigs, dconfig)


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
