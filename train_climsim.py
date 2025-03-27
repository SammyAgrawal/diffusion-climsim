import os
import sys
from tqdm import tqdm
from pathlib import Path
import time
import json
#import diffusers
import diffusionsim.training_utils as tru
import torch
import torch.nn as nn
from dataclasses import dataclass, asdict, field
import pprint
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'

rank = 0
device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")



def define_configs():
    dl_params = tru.TrainLoaderParams()
    dl_params.batch_size = 128
    dl_params.shuffle = False
    dl_params.num_workers = 4
    dl_params.prefetch_factor = 3
    dl_params.persistent_workers = True
    dl_params.multiprocessing_context = "forkserver"

    dconfig = tru.DataConfig()
    dconfig.dataloader_params = dl_params
    dconfig.source = "gcsfs" # specify from raw cloud bucket
    dconfig.climsim_type = "low-res-expanded"
    dconfig.dataset_type = "climsim"
    dconfig.data_dir = "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/hf_manifests/"
    dconfig.train_test_split = [0.70, 0.30]

    tconfig = tru.TrainingConfig()
    tconfig.exp_id = 'climsim_training'
    tconfig.num_epochs = 5
    tconfig.phases = ['train']

    #tconfig.lr_scheduler = 'get_cosine_schedule_with_warmup'
    #tconfig.lr_warmup_steps = 100
    tconfig.learning_rate = 3e-5
    tconfig.batch_logging_interval = 32
    tconfig.batch_checkpoint_interval = 50
    tconfig.save_best_epoch = True
    tconfig.log_gradients = False
    tconfig.loss_weights = {'mse': 1.0, 'distribution': 0.0, 'diffusion': 0.0}
    tconfig.max_T_sample = 50

    unet = tru.UNetParams()
    unet.block_out_channels = (128, 256, 512)
    unet.down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D")
    unet.up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D")
    unet.layers_per_block = 1
    unet.norm_num_groups = 2

    mconfig = tru.ModelConfig()
    mconfig.model_type = "ddpm_diffusion"
    mconfig.unet = unet
    mconfig.scheduler =  tru.SchedulerParams()
    # define baseline model
    mconfig.bl_hidden_dims = [256, 256]
    mconfig.bl_num_layers = 2

    return(tconfig, mconfig, dconfig)

def setup_configs(exp_id, run_id, exp_dir, use_distribution_loss, use_diffusion_loss):
    from pathlib import Path
    base_dir = os.path.join(exp_dir, exp_id)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    tconfig, mconfig, dconfig = define_configs()
    tconfig.exp_id = exp_id
    with open(os.path.join(base_dir, f'{run_id}.json'), "w") as f:
        json.dump(dict(
            training_config=asdict(tconfig),
            model_config=asdict(mconfig),
            data_config=asdict(dconfig),
            use_distribution_loss=use_distribution_loss,
            use_diffusion_loss=use_diffusion_loss,
        ), f)
    return(base_dir, tconfig, mconfig, dconfig)


if __name__ == "__main__":
    #typer.run(main)
    #typer.run(test_args)
    exp_dir = "/home/jovyan/Samarth/ClimsimProjectWork/diffusion-climsim/experiments"
    exp_id = "climsim_training"
    run_id = "trial_1_just_mse"

    use_distribution_loss = False
    use_diffusion_loss = False

    base_dir, tconfig, mconfig, dconfig = setup_configs(exp_id, run_id, exp_dir, use_distribution_loss, use_diffusion_loss)
    run_start_time = tru.log_event("run start", 
        data_params = asdict(dconfig.dataloader_params),
    )
    t0 = tru.log_event("setup start", run_id=run_id)
    pprint.pprint(asdict(tconfig))
    print("\n\n")
    pprint.pprint(asdict(mconfig))
    print("\n\n" )
    pprint.pprint(asdict(dconfig))
    print("\n\n", )
    
    #unet = tru.load_model(mconfig)
    #scheduler = tru.load_scheduler(mconfig)
    dataloaders, indices = tru.load_dataloaders(dconfig, log=True)
    with open(os.path.join(base_dir, f'{run_id}.json'), "r") as f:
        log = json.load(f)
    log['test_indices'] = indices[1].tolist()
    with open(os.path.join(base_dir, f'{run_id}.json'), "w") as f:
        json.dump(log, f)
    
    model = tru.build_baseline_model(mconfig)

    loss_fn = nn.MSELoss()
    optimizer = tru.create_optimizer(model, tconfig)
    #print("Testing batch fetch")
    #next(iter(dataloaders[0])) # just to finish setting up

    trainer = tru.ClimsimTrainer(model, dataloaders, loss_fn, optimizer, 
                   tconfig, base_dir, use_distribution_loss, use_diffusion_loss, rank=0)


    tru.log_event("setup end", duration=time.time() - t0)
    trainer.train(num_epochs=5, log=True, run_id=run_id)
    print("Done!")
    tru.log_event("run end", duration = time.time() - run_start_time)