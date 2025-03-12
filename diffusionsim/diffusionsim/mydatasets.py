import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
import xarray as xr
import numpy as np
import pandas as pd
import gcsfs
import dask
import xbatcher
from dataclasses import dataclass, asdict, field
import inspect
try:
    import diffusers
    diffusers_available = True
except Exception as e:
    diffusers_available = False
from . import climsim_utils as cut
import time
fs = gcsfs.GCSFileSystem()

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_path(file):
    return os.path.join(_ROOT, 'climsim_data', file)

#print(os.path.dirname(__file__))

def log_event(event_name, **kwargs):
    t = time.time()
    log = dict(event=event_name, time=t, pid=torch.multiprocessing.current_process().pid)
    for key in kwargs:
        log[key] = kwargs[key]
    print(json.dumps(log), file=sys.stderr)
    return(t)

def load_dataset(dconfig, log=False):
    dsi, dso = cut.load_raw_dataset(dconfig)
    dsets, indices = train_test_split(dsi, dso, dconfig.train_test_split)
    datasets = []
    for (dsi, dso) in dsets:
        match dconfig.dataset_type.lower():
            case ds if "xbatch" in ds:
                datasets.append(XBatchDataset(dso, dconfig, log=log))
            case ds if "image" in ds:
                datasets.append(ClimsimImageDataset(dsi, dso, dconfig, log))
            case ds if "climsim" in ds:
                datasets.append(ClimsimDataset(dsi, dso, dconfig, log))
            case _:
                return(dsets, indices)
    return(datasets, indices)

def load_dataloader(dconfig, log=False):
    datasets, indices = load_dataset(dconfig, log)
    params = asdict(dconfig.dataloader_params)
    dataloaders = []
    for dataset in datasets:
        match dconfig.dataset_type.lower():
            case ds if "xbatch" in ds or "image" in ds:
                # batch size is already set via xbatcher in dataset sample; dataloader should just return one item
                params['batch_size'] = 1
                dataloaders.append(DataLoader(dataset, collate_fn=collate_test_fn, **params))
            case _:
                dataloaders.append(DataLoader(dataset, **params))
    return(dataloaders)

def train_test_split(dsi, dso, split_frac=[0.75, 0.25], typ='xr'):
    datasets, indices = [], []
    if(typ == 'np'):
        num_timesteps = dsi.sizes['state']
    else:
        num_timesteps = dsi.sizes['time']
    times = np.arange(num_timesteps)
    np.random.shuffle(times)
    counter = 0
    for frac in split_frac:
        # Calculate the split index
        split = int(num_timesteps * frac)
        if(split > 0):
            phase_indices = np.sort(times[counter:counter+split])
            indices.append(phase_indices)
            if(typ == 'np'):
                datasets.append((dsi.isel(state=phase_indices), dso.isel(state=phase_indices)))
            else:
                datasets.append((dsi.isel(time=phase_indices), dso.isel(time=phase_indices)))
            counter += split
    return(datasets, indices)

def get_norm_info(style='image'):
    if(style=='image'):    
        X_mean = xr.open_dataset(get_path("image_xmean.nc"))
        X_std = xr.open_dataset(get_path("image_xstd.nc"))
        Y_mean = xr.open_dataset(get_path("image_ymean.nc"))
        Y_std = xr.open_dataset(get_path("image_ystd.nc"))
        Y_std['cam_out_PRECSC'].data = Y_std.cam_out_PRECSC.mean().item() * np.ones_like(Y_std.cam_out_PRECSC.data) 
        return(X_mean, X_std, Y_mean, Y_std)
    elif(style=='nc'):
        input_mean = xr.open_dataset(get_path('input_mean.nc'))
        input_max = xr.open_dataset(get_path('input_max.nc'))
        input_min = xr.open_dataset(get_path('input_min.nc'))
        output_scale = xr.open_dataset(get_path('output_scale.nc'))
        #output_std['cam_out_PRECSC'] = output_std.cam_out_PRECSC.mean()
        return(input_mean, input_max, input_min, output_scale)

def load_scheduler(mconfig):
    def pass_config(func, data_class):
        accepted_params = inspect.signature(func).parameters
        filtered_kwargs = {k: v for k, v in asdict(data_class).items() if k in accepted_params}
        return func(**filtered_kwargs)
    match mconfig.scheduler_type.lower():
        case sched if 'ddpm' in sched:
            # in charge of betas
            return(pass_config(diffusers.DDPMScheduler, mconfig.scheduler))
        case sched if 'ddim' in sched:
            return(pass_config(diffusers.DDIMScheduler, mconfig.scheduler))
        case other if True:
            print(f"scheduler '{other}' not yet supported")
        
def noise_batch(scheduler, clean_images, device):
    num_timesteps = scheduler.config.num_train_timesteps
    # Given a batch from a dataloader on the dataset, return a noised sample
    noise = torch.randn(clean_images.shape, device=device)
    timesteps = torch.randint(0, num_timesteps, size=(clean_images.shape[0],), device=device, dtype=torch.int64)
    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
    return(noisy_images, timesteps, noise)
    

class ClimsimDataset(Dataset):
    def __init__(self, dsi, dso, dconfig, log=False):
        self.log = log
        self.dsi, self.dso = dsi, dso
        #assert self.dsi.sizes['time'] == self.dso.sizes['time'], "dsi and dso must have the same number of timesteps"

        input_vars, target_vars = list(dsi.data_vars), list(dso.data_vars) 
        self.X_mean, self.X_std, self.Y_mean, self.Y_std = get_norm_info(style='image')
        self.xm = self.X_mean[input_vars].mean(dim=['ncol']).to_stacked_array(new_dim='mli', sample_dims=())
        self.xs = self.X_std[input_vars].mean(dim=['ncol']).to_stacked_array(new_dim='mli', sample_dims=())
        self.ym = self.Y_mean[target_vars].mean(dim=['ncol']).to_stacked_array(new_dim='mlo', sample_dims=())
        self.ys = self.Y_std[target_vars].mean(dim=['ncol']).to_stacked_array(new_dim='mlo', sample_dims=())

        self.mli, self.mlo = self.xm.mli, self.ym.mlo
        
        #self.xgen = xbatcher.BatchGenerator(self.dsi, input_dims=dict(time=dconfig.dataloader_params.batch_size, lev=60, ncol=384), preload_batch=False,)
        #self.ygen = xbatcher.BatchGenerator(self.dso, input_dims=dict(time=dconfig.dataloader_params.batch_size, lev=60, ncol=384), preload_batch=False,)

        dsi = dsi.to_stacked_array(new_dim="mli", sample_dims=("time", "ncol"))
        self.X = dsi.stack(sample=("time", "ncol")).transpose("sample", "mli")
        dso = dso.to_stacked_array(new_dim="mlo", sample_dims=("time", "ncol"))
        self.Y = dso.stack(sample=("time", "ncol")).transpose("sample", "mlo")
        self.length = min(self.X.shape[0], self.Y.shape[1])

    def __getitem__(self, idx):
        if(self.log):
            t0 = log_event("get-batch start", batch_idx=idx)
        x, y = self.X[idx].load(), self.Y[idx].load()
        x = (x - self.xm) / self.xs
        y = (y - self.ym) / self.ys
        x, y = torch.tensor(x.data, dtype=torch.float32), torch.tensor(y.data, dtype=torch.float32)
        if(self.log):
            log_event("get-batch end", batch_idx=idx, duration=time.time() - t0)

        return(x, y)
    
    def __len__(self):
        return(self.length)
    def index_var(self, var, level):
        mli, mlo = list(self.mli.values), list(self.mlo.values)
        if((var, level) in mli):
            return(mli.index((var, level)))
        elif((var, level) in mlo):
            return(mlo.index((var, level)))
        return(-1)

    def reconstruct_X(self, X_norm):
        X_rec = (X_norm * self.X_std) + self.X_mean
        return(X_rec)
    
    def reconstruct_Y(self, Y_norm):
        Y_rec = (Y_norm * self.Y_std) + self.Y_mean
        return(Y_rec)
    


class ClimsimDatasetOld(Dataset):
    def __init__(self, X, Y, normalize=True):
        self.device = device
        self.mli, self.mlo = X.mli, Y.mlo
        self.Xarr, self.Yarr = X, Y
        if(normalize):
            X, Y = self.normalize(X, Y, load_norm=False)
        self.X = torch.tensor(X.values, dtype=torch.float32) # each row is datapoint
        self.Y = torch.tensor(Y.values, dtype=torch.float32)

    def normalize(self, X, Y, load_norm=True):
        print("Normalizing data")
        if(load_norm):
            self.X_mean = xr.DataArray(np.load(get_path("X_mean.npy")), coords={'mli' : self.mli})
            self.X_std = xr.DataArray(np.load(get_path("X_std.npy")), coords={'mli' : self.mli})
            self.Y_mean = xr.DataArray(np.load(get_path("Y_mean.npy")), coords={'mlo' : self.mlo})
            self.Y_std = xr.DataArray(np.load(get_path("Y_std.npy")), coords={'mlo' : self.mlo})
        else:
            self.X_mean, self.X_std = X.mean(dim=['sample', 'ncol']), X.std(dim=['sample', 'ncol'])
            self.Y_mean, self.Y_std = Y.mean(dim=['sample', 'ncol']), Y.std(dim=['sample', 'ncol'])
        
        X_norm = (X - self.X_mean) / self.X_std
        Y_norm = (Y - self.Y_mean) / self.Y_std
        return(X_norm, Y_norm)
        
    def __len__(self):
        return(self.Y.shape[0])

    def __getitem__(self, idx):
        return(self.X[idx], self.Y[idx])



class ClimsimImageDataset(Dataset):
    def __init__(self, dsi, dso, dconfig, log=False):
        self.X_mean, self.X_std, self.Y_mean, self.Y_std = get_norm_info(style='nc')
        self.normalize = dconfig.prenormalize
        self.log = log
        self.permute_indices = image_regridding(dsi)
        if(self.normalize):
            self.X = (dsi - self.X_mean) / self.X_std
            self.Y = (dso - self.Y_mean) / self.Y_std
        else:
            self.X, self.Y = dsi, dso
        self.xgen = xbatcher.BatchGenerator(self.X, input_dims=dict(time=dconfig.dataloader_params.batch_size, lev=60, ncol=384), preload_batch=False,)
        self.ygen = xbatcher.BatchGenerator(self.Y, input_dims=dict(time=dconfig.dataloader_params.batch_size, lev=60, ncol=384), preload_batch=False,)
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            self.mli = dsi.to_stacked_array(new_dim='mli', sample_dims=("time", "ncol")).mli
            self.mlo = dso.to_stacked_array(new_dim='mlo', sample_dims=("time", "ncol")).mlo
    
    def __getitem__(self, idx):
        if(self.log):
            t0 = log_event("get-batch start", batch_idx=idx)
        x, y = self.xgen[idx].load(), self.ygen[idx].load()
        if(not self.normalize): # wasn't normalized from beginning, must do now
            x, y = (x - self.X_mean) / self.X_std, (y - self.Y_mean) / self.Y_std
        x, y = x.isel(ncol=self.permute_indices), y.isel(ncol=self.permute_indices)
        x, y = x.to_stacked_array(new_dim="mli", sample_dims=("time", "ncol")), y.to_stacked_array(new_dim="mlo", sample_dims=("time", "ncol"))
        x, y = torch.tensor(x.data, dtype=torch.float32), torch.tensor(y.data, dtype=torch.float32)
        if(self.log):
            log_event("get-batch end", batch_idx=idx, duration=time.time() - t0)
        return x, y
    
    def __len__(self):
        return(len(self.xgen))

class XBatchDataset(torch.utils.data.Dataset):
    def __init__(self, dso, dconfig, log=False):
        self.Xmean, self.Xstd, self.Ymean, self.Ystd = get_norm_info("image")
        # snowfall has some zeros, so just take global mean to avoid dividing by zero
        self.height, self.width = (16, 24)
        self.normalize = dconfig.prenormalize
        self.log = log
        self.permute_indices = image_regridding(dso)
        if(self.normalize):
            self.data = (dso - self.Ymean) / self.Ystd
        else:
            self.data = dso
        self.bgen = xbatcher.BatchGenerator(self.data, input_dims=dict(time=dconfig.dataloader_params.batch_size, lev=60, ncol=384),
                preload_batch=False,
        )
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            self.mlo = dso.to_stacked_array(new_dim='mlo', sample_dims=("time", "ncol")).mlo
    
    def __getitem__(self, idx):
        if(self.log):
            t0 = log_event("get-batch start", batch_idx=idx)
        data = self.bgen[idx].load()
        if(not self.normalize): # wasn't normalized from beginning, must do now
            data = (data - self.Ymean.mean(dim='ncol')) / self.Ystd.mean(dim='ncol')
        data = data.isel(ncol=self.permute_indices)
        data = data.to_stacked_array(new_dim="mlo", sample_dims=("time", "ncol"))
        data = data.transpose("time", "mlo", "ncol")
        data = torch.tensor(data.data.reshape(-1, 128, self.height, self.width), dtype=torch.float32)
        if(self.log):
            log_event("get-batch end", batch_idx=idx, duration=time.time() - t0)
        return data
        
    def __len__(self):
        return(len(self.bgen))

    def reconstruct_X(self, X_norm):
        X_rec = (X_norm * self.X_std) + self.X_mean
        return(X_rec)
        
    def reconstruct_Y(self, Y_norm):
        mean, std = torch.tensor(self.Y_mean.values).view(128, 1, 1), torch.tensor(self.Y_std.values).view(128, 1, 1)
        Y_rec = (Y_norm * std) + mean
        return(Y_rec)

def add_time(ds_in, ds_out):
    def compute_time(ymd, tod):
        year, month, day = (ymd // 10000)+2000, (ymd % 10000) // 100, ymd % 100
        hour, minute = tod // 3600, (tod % 3600) // 60
        dt_str = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
        return pd.to_datetime(dt_str)
    
    time = np.vectorize(compute_time)(ds_in.ymd.data, ds_in.tod.data)
    ds_in['sample'] = time
    ds_out['sample'] = time
    return(ds_in.rename({'sample':'time'}), ds_out.rename({'sample':'time'}))


def add_tendencies(ds_out, output_vars):
    print("Converting state deltas to tendencies")
    for var in output_vars:
        if('ptend' in var and var not in ds_out.data_vars): # each timestep is 20 minutes which corresponds to 1200 seconds
            v = var.replace("ptend", "state")
            ds_out[var] = (ds_out[v] - ds_out[v]) / 1200

    return(ds_out[output_vars])

def image_regridding(ds):
    lat, lon = np.round(ds.lat.data), np.round(ds.lon.data)
    array = np.column_stack([lon, lat])
    # first sort by longitude, then by latitude (top is area of high longitude)
    sorted_indices = np.lexsort((array[:, 0], -1*array[:, 1]))
    arr = array[sorted_indices]
    indices = np.array([], dtype=int)
    for i in range(16):
        start = i*24
        indices = np.concatenate([indices, start + np.argsort(arr[start:start+24, 0])])

    return(sorted_indices[indices])



def collate_test_fn(batches):
    return(batches[0])