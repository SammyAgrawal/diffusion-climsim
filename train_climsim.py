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
import pprint
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'

trainer = None
num_epochs=5

REF_BATCH_SIZE = 128
EXP_DIR = "/mnt/home/ssa2206/Climsim/experiments"
dataset_type="climsim"
in_notebook = True

print(f"Starting run with cuda: {torch.cuda.is_available()}", flush=True)

def cleanup_and_exit(signum, frame):
    print(f"Received signal {signum}, cleaning up before exiting...", flush=True)
    sys.stdout.flush()
    if trainer is not None:
        trainer.finish_training(num_epochs)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


signal.signal(signal.SIGTERM, cleanup_and_exit)
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGUSR1, cleanup_and_exit)
signal.signal(signal.SIGUSR2, cleanup_and_exit)

def setup_climsim_run(num_models, exp_id, base_run_id, data_vars='v1', batch_size=128):
    dconfig = tru.my_dconfig("local-vzarr", data_vars, in_notebook, dataset_type, batch_size=batch_size, use_tendencies=False)
    dconfig.shuffle_indices = False
    dconfig.train_test_split = [0.25, 0.10]
    tconfigs, mconfigs = [], []

    learning_rates = [1e-3, 1e-3, 1e-3, 1e-3]
    mse_weights = [1.0, 1.0, 1.0, 1.0]
    distloss_weights = [0.0, 0.0, 1.0, 1.0] # [1.0, 1.0, 2.0, 5.0]
    diffloss_weights = [0.0, 10.0, 0.0, 10.0]
    target_variables_distloss = [68, 60, 73, 82]
    num_gaussians = [3, 2, 3, 2]

    run_ids = "trial3-mse trial3-diff trial3-dist trial3-joint".split()
    for i in range(num_models):
        tconfig = tru.TrainingConfig(exp_id=exp_id, run_id=run_ids[i])
        tconfig.phases = ['train', 'eval']  
        tconfig.learning_rate = learning_rates[i] * batch_size / REF_BATCH_SIZE
        tconfig.loss_weights = {'mse': mse_weights[i], 'distribution': distloss_weights[i], 'diffusion': diffloss_weights[i]}
        tconfig.clip_gradients = True
        tconfig.lr_scheduler = "cosine"
        
        tconfig.distloss_var_inds = target_variables_distloss
        tconfig.num_gaussians = num_gaussians
        tconfig.diffusion_loss_noise_level = 20
        tconfig.distloss_var_sel = "uniform"

        tconfig.log_gradients = True
        tconfig.batch_checkpoint_interval = 50
        tconfig.batch_logging_interval = 32
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

if __name__ == "__main__":
    #typer.run(main)
    #typer.run(test_args)
    exp_id = "DiffLossTesting"
    run_id = "trial3"
    base_dir = os.path.join(EXP_DIR, exp_id)
    tconfigs, mconfigs, dconfig = setup_climsim_run(4, exp_id, run_id, data_vars='v1', batch_size=256)
    run_start_time = tru.log_event("run start", data_params = asdict(dconfig.dataloader_params))
    t0 = tru.log_event("setup start", run_id=run_id)
    
    #unet = tru.load_model(mconfigs[0], "diffusion")
    #scheduler = tru.load_scheduler(mconfig)
    trainer = ClimsimTrainer(dconfig, mconfigs, tconfigs, torch.nn.MSELoss(), base_dir, run_id, rank=0)
    tru.log_event("setup end", duration=time.time() - t0)
    if (RESTART_FROM_CKPT):
        trainer.restart_from_ckpt(num_epochs, log=True, cid='')
    else:
        trainer.train(num_epochs, log=True)
    print("Done!", flush=True)
    tru.log_event("run end", duration = time.time() - run_start_time)
