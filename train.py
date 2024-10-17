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
from diffusionsim import mydatasets as data
import torch
from typing import Optional
import typer
from dataclasses import dataclass, asdict, field
from typing_extensions import Annotated
import pprint
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

rank = 0
device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
if __name__ == "__main__":
    #typer.run(main)
    #typer.run(test_args)
    exp_id = "full_dataset_testrun"
    run_id = "trial_1b"
    tconfig, mconfig, dconfig = tru.load_config(run_id, exp_id)
    run_start_time = data.log_event("run start", 
        data_params = asdict(dconfig.dataloader_params),
    )
    t0 = data.log_event("setup start", run_id=run_id)
    pprint.pprint(asdict(tconfig))
    print("\n\n")
    pprint.pprint(asdict(mconfig))
    print("\n\n" )
    pprint.pprint(asdict(dconfig))
    print("\n\n", )
    trainer = tru.setup_trainer(exp_id, run_id, tconfig, mconfig, dconfig)
    data.log_event("setup end", duration=time.time() - t0)
    trainer.train(num_epochs=20, log=True, run_id='trial_1b')
    print("Done!")
    data.log_event("run end", duration = time.time() - run_start_time)