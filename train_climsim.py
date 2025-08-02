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
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'

trainer = None
num_epochs = 10

REF_BATCH_SIZE = 128
dataset_type="climsim"
in_notebook = True

NUM_MODELS = 7
os.environ['WANDB_NAME'] = ""
os.environ['WANDB_NOTES'] = ""


def setup_climsim_run(num_models, exp_id, base_run_id, data_vars='v1', batch_size=128):
    dconfig = tru.my_dconfig("local-vzarr", data_vars, in_notebook, dataset_type, batch_size=batch_size, use_tendencies=False)
    dconfig.shuffle_indices = True
    dconfig.train_test_split = [0.15, 0.05]
    tconfigs, mconfigs = [], []

    learning_rates = [1e-5, 5e-5, 1e-4, 2e-4, 6e-4, 0.001, 0.005]
    diffloss_weights = [0.0, 0.0, 0.0, 0.0]
    distloss_weights = [0.0, 0.0, 0.0, 0.0]
    target_variables_distloss = [68, 73, 82]
    num_gaussians = [2, 2, 2]
    
    #run_ids = f"{base_run_id}-mse {base_run_id}-diff {base_run_id}-dist {base_run_id}-joint".split()
    lettering = "abcdefghijklmnopqrstuvwxyz"
    run_ids = [f"{base_run_id}-{lettering[i]}" for i in range(num_models)]
    for i in range(num_models):
        tconfig = tru.TrainingConfig(exp_id=exp_id, run_id=run_ids[i])
        tconfig.phases = ['train', 'eval']
        tconfig.learning_rate = learning_rates[i] * batch_size / REF_BATCH_SIZE
        tconfig.loss_weights = {'mse': 1.0, 'distribution': distloss_weights[0], 'diffusion': diffloss_weights[0]}
        tconfig.loss_schedule = {'mse': [0,100], 'distribution': 2 , 'diffusion': 2}
        tconfig.clip_gradients = True
        tconfig.lr_scheduler = "cosine"
        
        tconfig.distloss_var_inds = target_variables_distloss
        tconfig.num_gaussians = num_gaussians
        tconfig.distloss_var_sel = "uniform"

        tconfig.diffusion_loss_noise_level = 20

        tconfig.log_gradients = True
        tconfig.batch_checkpoint_interval = 50
        tconfig.batch_logging_interval = 16
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


RESTART_FROM_CKPT = False
RERUN = True
torch.set_grad_enabled(True)

EXP_DIR = "/mnt/home/ssa2206/Climsim/experiments"
if __name__ == "__main__":
    #typer.run(main)
    #typer.run(test_args)
    exp_id = "LossWeightTesting"
    run_id = "JointTest"
    run_start_time = tru.log_event("run start", exp_id=exp_id, run_id=run_id)
    t0 = tru.log_event("setup start")
    if(RERUN):
        run = evals.Run(exp_id, run_id)
        trainer = run.reconstruct_trainer(apply_checkpoints=False, apply_indices=False)
    else:
        base_dir = os.path.join(EXP_DIR, exp_id)
        tconfigs, mconfigs, dconfig = setup_climsim_run(NUM_MODELS, exp_id, run_id, data_vars='v1', batch_size=256)
        trainer = ClimsimTrainer(dconfig, mconfigs, tconfigs, torch.nn.MSELoss(), base_dir, run_id, rank=0)
    
    tru.log_event("setup end", duration=time.time() - t0)
    if (RESTART_FROM_CKPT):
        trainer.restart_from_ckpt(num_epochs, log=True, cid='')
    else:
        trainer.train(num_epochs, log=True)
    
    print("Done!", flush=True)
    tru.log_event("run end", duration = time.time() - run_start_time)
