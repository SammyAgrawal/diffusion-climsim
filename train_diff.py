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
if __name__ == "__main__":
    #typer.run(main)
    #typer.run(test_args)
    exp_id = "empire_testrun"
    run_id = "trial_2"
    exp_dir = "/mnt/home/ssa2206/diffusion-climsim/experiments"
    tconfig, mconfig, dconfig = tru.load_config(run_id, exp_id, exp_dir)
    run_start_time = tru.log_event("run start", 
        data_params = asdict(dconfig.dataloader_params),
    )
    t0 =  tru.log_event("setup start", run_id=run_id)
    pprint.pprint(asdict(tconfig))
    print("\n\n")
    pprint.pprint(asdict(mconfig))
    print("\n\n" )
    pprint.pprint(asdict(dconfig))
    print("\n\n", )
    
    unet = tru.load_model(mconfig)
    dataloaders = tru.load_dataloaders(dconfig)

    model = 




    tru.log_event("setup end", duration=time.time() - t0)
    trainer.train(num_epochs=20, log=True, run_id=run_id)
    print("Done!")
    tru.log_event("run end", duration = time.time() - run_start_time)