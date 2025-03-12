import os
import sys
from tqdm import tqdm
from pathlib import Path
import time
import json
#import diffusers
import diffusionsim as diff
import diffusionsim.training_utils as tru
from diffusionsim import mydatasets as data
import torch
import torch.nn as nn
from typing import Optional
import typer
from dataclasses import dataclass, asdict, field
from typing_extensions import Annotated
import pprint
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'

rank = 0
device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def setup_configs(exp_id, run_id, exp_dir, use_distribution_loss, use_diffusion_loss):
    from pathlib import Path
    base_dir = os.path.join(exp_dir, exp_id)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    tconfig, mconfig, dconfig = tru.load_config(run_id, exp_id, exp_dir)
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
    t0 = data.log_event("setup start", run_id=run_id)
    pprint.pprint(asdict(tconfig))
    print("\n\n")
    pprint.pprint(asdict(mconfig))
    print("\n\n" )
    pprint.pprint(asdict(dconfig))
    print("\n\n", )
    
    #unet = tru.load_model(mconfig)
    #scheduler = tru.load_scheduler(mconfig)
    dataloaders, indices = tru.load_dataloaders(dconfig)
    with open(os.path.join(base_dir, f'{run_id}.json'), "r") as f:
        log = json.load(f)
    log['test_indices'] = indices[1].tolist()
    with open(os.path.join(base_dir, f'{run_id}.json'), "w") as f:
        json.dump(log, f)
    
    model = tru.build_baseline_model(mconfig)

    loss_fn = nn.MSELoss()
    optimizer = tru.create_optimizer(model, tconfig)
    next(iter(dataloaders[0])) # just to finish setting up

    trainer = tru.ClimsimTrainer(model, dataloaders, loss_fn, optimizer, 
                   tconfig, base_dir, use_distribution_loss, use_diffusion_loss, rank=0)


    data.log_event("setup end", duration=time.time() - t0)
    trainer.train(num_epochs=5, log=True, run_id=run_id)
    print("Done!")
    data.log_event("run end", duration = time.time() - run_start_time)