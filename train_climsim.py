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



def define_configs(exp_id, climsim_training=True, in_notebook=False, lr=3e-5):
    dl_params = tru.TrainLoaderParams()
    dl_params.batch_size = 128
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
    dconfig.data_vars = "v1"

    tconfig = tru.TrainingConfig()
    tconfig.exp_id = exp_id
    tconfig.num_epochs = 5
    tconfig.phases = ['train', 'eval']
    #tconfig.lr_scheduler = 'get_cosine_schedule_with_warmup'
    #tconfig.lr_warmup_steps = 100
    tconfig.learning_rate = lr
    tconfig.batch_logging_interval = 32
    tconfig.batch_checkpoint_interval = 50
    tconfig.save_best_epoch = True
    tconfig.log_gradients = False
    tconfig.loss_weights = {'mse': 1.0, 'distribution': 0.0, 'diffusion': 0.0}
    tconfig.max_T_sample = 51

    unet = tru.UNetParams()
    unet.block_out_channels = (128, 256, 512) if dconfig.data_vars == "v1" else (256, 512, 1024)
    unet.down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D")
    unet.up_block_types = ("UpBlock2D", "UpBlock2D", "UpBlock2D")
    unet.layers_per_block = 1
    unet.norm_num_groups = 2
    unet.in_channels = 128 if dconfig.data_vars == "v1" else 368
    unet.out_channels = unet.in_channels

    mconfig = tru.ModelConfig()
    mconfig.model_type = "ddpm_diffusion"
    mconfig.unet = unet
    mconfig.scheduler = tru.SchedulerParams()
    # define baseline model
    mconfig.bl_hidden_dims = [256, 256]
    mconfig.bl_num_layers = 2

    return(tconfig, mconfig, dconfig)


def setup_configs(exp_id, run_id, exp_dir, 
                  tconfig, mconfig, dconfig, 
                  use_distribution_loss, use_diffusion_loss, **kwargs):
    from pathlib import Path
    base_dir = os.path.join(exp_dir, exp_id)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    tconfig.exp_id = exp_id
    with open(os.path.join(base_dir, f'{run_id}.json'), "w") as f:
        json.dump(dict(
            training_config=asdict(tconfig),
            model_config=asdict(mconfig),
            data_config=asdict(dconfig),
            use_distribution_loss=use_distribution_loss,
            use_diffusion_loss=use_diffusion_loss,
            **kwargs
        ), f)
    return(base_dir, tconfig, mconfig, dconfig)


if __name__ == "__main__":
    #typer.run(main)
    #typer.run(test_args)
    exp_dir = "/mnt/home/ssa2206/Climsim/experiments"
    exp_id = "JointTraining"
    run_id = "trial_1_just_mse"
    lr = 1e-4
    tconfig, mconfig, dconfig = define_configs(exp_id, climsim_training=True, in_notebook=False, lr=lr)
    use_distribution_loss = False
    use_diffusion_loss = False
    t0 = tru.log_event("setup start", run_id=run_id)
    dataloaders, indices = tru.load_dataloaders(dconfig, log=True, shuffle_indices=False)
    base_dir, tconfig, mconfig, dconfig = setup_configs(exp_id, run_id, exp_dir, 
                                                        tconfig, mconfig, dconfig, 
                                                        use_distribution_loss, use_diffusion_loss, 
                                                        test_indices=indices[1].tolist())
    run_start_time = tru.log_event("run start", 
        data_params = asdict(dconfig.dataloader_params),
    )
    pprint.pprint(asdict(tconfig))
    print("\n\n")
    pprint.pprint(asdict(mconfig))
    print("\n\n" )
    pprint.pprint(asdict(dconfig))
    print("\n\n", )
    
    #unet = tru.load_model(mconfig)
    #scheduler = tru.load_scheduler(mconfig)
    
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