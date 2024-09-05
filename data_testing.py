import os
import sys
from tqdm import tqdm
import xarray as xr
import numpy as np
import time
import json
import dask
import matplotlib.pyplot as plt
import gcsfs
sys.path.append(os.path.abspath(os.path.join('diffusionsim')))
from diffusionsim import mydatasets as data
import diffusionsim as diff
import xbatcher
import torch
from torch import multiprocessing
import diffusers
from typing import Optional
import typer
from typing_extensions import Annotated

fs = gcsfs.GCSFileSystem()
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/srv/conda/envs/notebook'


def log_event(event_name, **kwargs):
    t = time.time()
    log = {
        "event": event_name,
        "time": t,
        "pid": multiprocessing.current_process().pid,
    }
    for key in kwargs:
        log[key] = kwargs[key]
    print(json.dumps(log))
    return(t)

class XBatchDataset(torch.utils.data.Dataset):
    def __init__(self, dso, batch_size, normalize=False, log=True):
        self.Xmean, self.Xstd, self.Ymean, self.Ystd = data.get_norm_info("image")
        # snowfall has some zeros, so just take global mean to avoid dividing by zero
        self.Ystd['cam_out_PRECSC'].data = self.Ystd.cam_out_PRECSC.mean().item() * np.ones_like(self.Ystd.cam_out_PRECSC.data) 
        self.height, self.width = (16, 24)
        self.normalize = normalize
        self.log = log
        self.permute_indices = data.image_regridding(dso)
        if(normalize):
            #print("Normalizing")
            self.data = (dso - self.Ymean) / self.Ystd
        else:
            self.data = dso

        self.bgen = xbatcher.BatchGenerator(self.data,
                input_dims=dict(time=batch_size, lev=60, ncol=384),
                preload_batch=False,
        )
    def __len__(self):
        return(len(self.bgen))
    
    def __getitem__(self, idx):
        if(self.log):
            t0 = log_event("get-batch start", batch_idx=idx)
        data = self.bgen[idx].load()
        if(not self.normalize): # wasn't normalized from beginning, must do now
            data = (data - self.Ymean.mean(dim='ncol')) / self.Ystd.mean(dim='ncol')
        data = data.isel(ncol=self.permute_indices)
        stacked = data.to_stacked_array(new_dim="mlo", sample_dims=("time", "ncol"))
        stacked = stacked.transpose("time", "mlo", "ncol")
        item = torch.tensor(stacked.data.reshape(-1, 128, self.height, self.width), dtype=torch.float32)
        if(self.log):
            log_event("get-batch end", batch_idx=idx, duration=time.time() - t0)

        return item


def setup(batch_size):
    input_vars, output_vars = data.load_vars('v1', tendencies=False)
    mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr')
    ds_out = xr.open_dataset(mapper, engine='zarr', chunks={})
    ds_out = ds_out[output_vars].rename({'sample':'time'})
    ds_out = data.add_space(ds_out)
    dataset = XBatchDataset(ds_out, batch_size, normalize=False, log=True)
    return(dataset)

def train(training_generator, num_epochs, train_step_time, num_batches):
    for epoch in range(num_epochs):
        e0 = log_event("epoch start", epoch=epoch)
        for i, sample in enumerate(training_generator):
            tt0 = log_event("training start", batch=i)
            time.sleep(train_step_time)  # simulate model training
            log_event("training end", batch=i, duration= time.time() - tt0, batch_shape=sample.shape)
            if i == num_batches - 1:
                break

        log_event("epoch end", epoch=epoch, duration=time.time() - e0)

def collate_test_fn(batches):
    return(batches[0])

def main(
    num_epochs: Annotated[int, typer.Option(min=0, max=1000)] = 2,
    num_batches: Annotated[int, typer.Option(min=0, max=1000)] = 3,
    batch_size: Annotated[int, typer.Option(min=0, max=1000)] = 1,
    shuffle: Annotated[Optional[bool], typer.Option()] = False,
    num_workers: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    prefetch_factor: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    persistent_workers: Annotated[Optional[bool], typer.Option()] = None,
    pin_memory: Annotated[Optional[bool], typer.Option()] = None,
    train_step_time: Annotated[Optional[float], typer.Option()] = 0.6,
    dask_threads: Annotated[Optional[int], typer.Option()] = None,
):
    data_params = dict(batch_size=1, shuffle=shuffle)
    if num_workers is not None:
        data_params["num_workers"] = num_workers
        data_params["multiprocessing_context"] = "forkserver"
    if prefetch_factor is not None:
        data_params["prefetch_factor"] = prefetch_factor
    if persistent_workers is not None:
        data_params["persistent_workers"] = persistent_workers
    if pin_memory is not None:
        data_params["pin_memory"] = pin_memory
    if dask_threads is None or dask_threads <= 1:
        dask.config.set(scheduler="single-threaded")
    else:
        dask.config.set(scheduler="threads", num_workers=dask_threads)
    _locals = {k: v for k, v in locals().items() if not k.startswith("_")}
    run_start_time = log_event("run start", **{"data_params": str(data_params),"locals": _locals,})
    t0 = log_event("setup start", data_params=str(data_params))
    dataset = setup(batch_size)
    training_generator = torch.utils.data.DataLoader(dataset, collate_fn=collate_test_fn, **data_params)
    _ = next(iter(training_generator))  # wait until dataloader is ready
    log_event("setup end", duration=time.time() - t0)
    train(training_generator, num_epochs, train_step_time, num_batches)
    log_event("run end", duration = time.time() - run_start_time)


def test_args(
    num_epochs: Annotated[int, typer.Option(min=0, max=1000)] = 2,
    num_batches: Annotated[int, typer.Option(min=0, max=1000)] = 3,
    batch_size: Annotated[int, typer.Option(min=0, max=1000)] = 1,
    shuffle: Annotated[Optional[bool], typer.Option()] = False,
    num_workers: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    prefetch_factor: Annotated[Optional[int], typer.Option(min=0, max=64)] = None,
    persistent_workers: Annotated[Optional[bool], typer.Option()] = None,
    pin_memory: Annotated[Optional[bool], typer.Option()] = None,
    train_step_time: Annotated[Optional[float], typer.Option()] = 0.6,
    dask_threads: Annotated[Optional[int], typer.Option()] = None,
):
    print(f"num_epochs: {num_epochs}")
    print(f"num_batches: {num_batches}")
    print(f"batch_size and shuffle: {batch_size}, {shuffle}")
    print(f"num_workers and prefetch factor: {num_workers}, {prefetch_factor}")
    print(f"persistent_workers and pin_memory: {persistent_workers}, {pin_memory}")
    print(f"train_step_time and dask_threads: {train_step_time}, {dask_threads}")

rank = 0
device = f"cuda:{rank}" if torch.cuda.is_available() else 'cpu'
if __name__ == "__main__":
    typer.run(main)
    #typer.run(test_args)
    