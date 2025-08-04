import torch
from torch.utils.data import DataLoader
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

def load_dataset(dconfig, log=False, indices=None):
    dsi, dso, dutils = cut.load_raw_dataset(dconfig, return_dutils=True)
    dsets, indices = train_test_split(dsi, dso, dconfig.train_test_split, indices, shuffle=dconfig.shuffle_indices)
    datasets = []
    for (dsi, dso) in dsets:
        match dconfig.dataset_type.lower():
            case ds if "1d" in ds:
                datasets.append(Diffusion2DDataset(dso.unify_chunks(), dutils, dconfig, log=log))
            case ds if "diffusion" in ds:
                datasets.append(Diffusion1DDataset(dso.unify_chunks(), dutils, dconfig, log))
            case ds if "climsim" in ds:
                datasets.append(ClimsimDataset(dsi.unify_chunks(), dso.unify_chunks(), dutils, dconfig, log))
            case _:
                return(dsets, indices)
    return(datasets, indices)

def load_dataloaders(dconfig, log=False, indices=None):
    datasets, indices = load_dataset(dconfig, log, indices)
    params = asdict(dconfig.dataloader_params)
    dataloaders = []
    for dataset in datasets:
        match dconfig.dataset_type.lower():
            case ds if "xbatch" in ds or "image" in ds or "climsim" in ds:
                # batch size is already set via xbatcher in dataset sample; dataloader should just return one item
                params['batch_size'] = 1
                dataloaders.append(DataLoader(dataset, collate_fn=collate_test_fn, **params))
            case _:
                dataloaders.append(DataLoader(dataset, **params))
    return(dataloaders, indices)

def train_test_split(dsi, dso, split_frac=[0.75, 0.25], indices=None, typ='xr', shuffle=True):
    if(indices is not None):
        # want to load up same indices as before; preserve prior run
        datasets = []
        assert isinstance(indices, list), "expect indices to be list of indices for each phase"
        for phase_index in indices:
            datasets.append((dsi.isel(time=phase_index), dso.isel(time=phase_index)))
        return(datasets, indices)
    datasets, indices = [], []
    if(typ == 'np'):
        num_timesteps = dsi.sizes['state']
    else:
        num_timesteps = dsi.sizes['time']
    times = np.arange(num_timesteps)
    if(shuffle):
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

def load_scheduler(mconfig):
    def pass_config(func, data_class):
        accepted_params = inspect.signature(func).parameters
        filtered_kwargs = {k: v for k, v in asdict(data_class).items() if k in accepted_params}
        return func(**filtered_kwargs)
    match mconfig.scheduler_type.lower():
        case sched if 'ddpm' in sched:
            # in charge of betas
            scheduler = pass_config(diffusers.DDPMScheduler, mconfig.scheduler)
        case sched if 'ddim' in sched:
            scheduler = pass_config(diffusers.DDIMScheduler, mconfig.scheduler)
        case other if True:
            print(f"scheduler '{other}' not yet supported")
            return(None)
    return(scheduler)   
        
def noise_batch(scheduler, clean_images, device):
    num_timesteps = scheduler.config.num_train_timesteps
    # Given a batch from a dataloader on the dataset, return a noised sample
    noise = torch.randn(clean_images.shape, device=device)
    timesteps = torch.randint(0, num_timesteps, size=(clean_images.shape[0],), device=device, dtype=torch.int64)
    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
    return(noisy_images, timesteps, noise)
    

class ClimsimDataset(torch.utils.data.Dataset):
    def __init__(self, dsi, dso, dutils, dconfig, log=False):
        self.dataset_type = dconfig.dataset_type
        self.output_only = "diff" in self.dataset_type
        self.image_dim = None
        for dim in ["1", "2", "3"]:
            if dim in self.dataset_type:
                self.image_dim = int(dim) 
                break

        self.log = log
        self.dutils = dutils
        self.dsi, self.dso = dsi, dso
        self.permute_indices = cut.image_regridding(dsi)
        #assert self.dsi.sizes['time'] == self.dso.sizes['time'], "dsi and dso must have the same number of timesteps"

        self.input_vars, self.target_vars = dutils.input_vars, dutils.target_vars
        self.input_len, self.target_len = dutils.input_feature_len, dutils.target_feature_len
        set_ds_norm_info(self, dconfig.use_tendencies)
        #self.xgen = xbatcher.BatchGenerator(self.dsi, input_dims=dict(time=dconfig.dataloader_params.batch_size, lev=60, ncol=384), preload_batch=False,)
        #self.ygen = xbatcher.BatchGenerator(self.dso, input_dims=dict(time=dconfig.dataloader_params.batch_size, lev=60, ncol=384), preload_batch=False,)

        dsi = dsi.to_stacked_array(new_dim="mli", sample_dims=("time", "ncol"))
        self.X = dsi.stack(sample=("time", "ncol")).transpose("sample", "mli")
        dso = dso.to_stacked_array(new_dim="mlo", sample_dims=("time", "ncol"))
        self.Y = dso.stack(sample=("time", "ncol")).transpose("sample", "mlo")

        self.xgen = xbatcher.BatchGenerator(self.X, input_dims=dict(sample=dconfig.dataloader_params.batch_size, mli=dutils.input_feature_len), preload_batch=False,)
        self.ygen = xbatcher.BatchGenerator(self.Y, input_dims=dict(sample=dconfig.dataloader_params.batch_size, mlo=dutils.target_feature_len), preload_batch=False,)

    def __getitem__(self, idx):
        if(self.log):
            t0 = log_event("get-item start", batch_idx=idx)
        y = self.ygen[idx].load()
        x = 0 if (self.output_only and not self.dutils.use_tendencies) else self.xgen[idx].load()
        if(self.dutils.use_tendencies):
            x_idx, y_idx = self.dutils.input_var_idx, self.dutils.target_var_idx
            filter_xvar = lambda var: x[:, x_idx[var][0]:x_idx[var][1]].data
            filter_yvar = lambda var: y[:, y_idx[var][0]:y_idx[var][1]].data

            var_pairs = [('state_t', 'ptend_t'), ('state_q0001', 'ptend_q0001')]
            if(self.dutils.full_vars):
                var_pairs += [('state_q0002', 'ptend_q0002'), ('state_q0003', 'ptend_q0003'), ('state_u', 'ptend_u'), ('state_v', 'ptend_v')]
            for (vx, vy) in var_pairs:
                y[:, slice(*y_idx[vy])] = (filter_yvar(vy) - filter_xvar(vx))/1200
        
        y = torch.tensor(y.data, dtype=torch.float32)
        x = 0 if self.output_only else torch.tensor(x.data, dtype=torch.float32)
        x, y = self.normalize(x, y)
        if(self.log):
            log_event("get-item end", batch_idx=idx, duration=time.time() - t0)
        if self.output_only:
            print(y.shape)
            return(cut.imagify(y, self.dutils, 'y', self.image_dim))
        return(self.make_image(x,y))
    

    def normalize(self, x, y):
        if(self.dutils.use_tendencies):
            x = (x - self.xm) / (self.xmax - self.xmin)
            y = y * self.ys
        else:  
            x = (x - self.xm) / self.xs
            y = (y - self.ym) / self.ys
        return(x, y)

    def denormalize(self, x, y):
        if(not torch.is_tensor(x)):
            x = torch.tensor(x)
        if(not torch.is_tensor(y)):
            y = torch.tensor(y)
        if(self.dutils.use_tendencies):
            x = x * (self.xmax - self.xmin) + self.xm
            y = y / self.ys
        else:  
            x = x * self.xs + self.xm
            y = y * self.ys + self.ym
        return(x, y)        
    
    def __len__(self):
        return(len(self.xgen))

    def index_var(self, var, level):
        mli, mlo = list(self.mli.values), list(self.mlo.values)
        if((var, level) in mli):
            return(mli.index((var, level)))
        elif((var, level) in mlo):
            return(mlo.index((var, level)))
        return(-1)
    
    def make_image(self, x, y, denormalize=False, ):
        # IMAGE : (BS, C, H, W)
        # Denormalize using tensors
        if(denormalize):
            x, y = self.denormalize(x, y)
        ximg = cut.imagify(x, self.dutils, 'x', self.image_dim)
        yimg = cut.imagify(y, self.dutils, 'y', self.image_dim)
        return ximg, yimg

class Diffusion1DDataset(torch.utils.data.Dataset):
    def __init__(self, dso, dutils, dconfig, log=False):
        self.dutils = dutils
        self.log = log
        self.input_vars, self.target_vars = dutils.input_vars, dutils.target_vars
        self.input_len, self.target_len = dutils.input_feature_len, dutils.target_feature_len
        self.data = (
            self.expand_levels(dso, self.target_vars, "mlo")
            .stack(sample=("time", "ncol"))
            .transpose("sample", "lev", "mlo")
        )

        set_ds_norm_info(self, self.dutils.use_tendencies)
        print(self.ys.shape)

        self.bgen = xbatcher.BatchGenerator(self.data, input_dims=dict(
            sample=dconfig.dataloader_params.batch_size, lev=60, mlo=self.data.mlo.size
        ), preload_batch=False,)
    
    def expand_levels(self, ds, vars, dim_name):
        out = [ds[var].expand_dims({'lev': ds.lev}) if var in self.dutils.normal_variables else ds[var] for var in vars]
        out =  xr.concat(out, dim=dim_name)
        out.assign_coords({dim_name : vars})
        return(out)
    
    def __getitem__(self, idx):
        mlo = self.bgen[idx].load()
        mlo = torch.tensor(mlo.data, dtype=torch.float32)
        mlo = (mlo - self.ym) / self.ys
        zero_pad = torch.zeros(mlo.size(0), 64-mlo.size(1), mlo.size(2))
        mlo = torch.cat([zero_pad, mlo], dim=1)
        return(mlo)

    def __len__(self):
        return(len(self.bgen))

class Diffusion2DDataset(torch.utils.data.Dataset):
    def __init__(self, dso, dutils, dconfig, log=False):
        self.Xmean, self.Xstd, self.Ymean, self.Ystd = cut.get_norm_info("state")
        # snowfall has some zeros, so just take global mean to avoid dividing by zero
        self.dutils = dutils
        self.height, self.width = (16, 24)
        self.log = log
        self.permute_indices = cut.image_regridding(dso)
        self.target_feature_len = dutils.target_feature_len
        self.data = dso.unify_chunks()
        self.bgen = xbatcher.BatchGenerator(self.data, input_dims=dict(
            time=dconfig.dataloader_params.batch_size, lev=60, ncol=384
        ), preload_batch=False,)
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            self.mlo = dso.to_stacked_array(new_dim='mlo', sample_dims=("time", "ncol")).mlo
    
    def __getitem__(self, idx):
        if(self.log):
            t0 = log_event("get-item start", batch_idx=idx)
        data = self.bgen[idx].load()
        data = (data - self.Ymean) / self.Ystd
        data = data.isel(ncol=self.permute_indices)
        data = data.to_stacked_array(new_dim="mlo", sample_dims=("time", "ncol"))
        data = data.transpose("time", "mlo", "ncol")
        data = torch.tensor(data.data.reshape(-1, self.target_feature_len, self.height, self.width), dtype=torch.float32)
        if(self.log):
            log_event("get-item end", batch_idx=idx, duration=time.time() - t0)
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
    

def set_ds_norm_info(ds, use_tendencies):
    IO_map = dict(x=(ds.input_vars, "mli"), y=(ds.target_vars, "mlo"))
    if(use_tendencies):
        A, B, C, D = cut.get_norm_info("scale")
        var_order = "xxxy"
    else:
        A, B, C, D = cut.get_norm_info(style='state')
        var_order = "xxyy"
    
    if(isinstance(ds, Diffusion1DDataset)):
        a = ds.expand_levels(A, IO_map[var_order[0]][0], IO_map[var_order[0]][1]).transpose("lev", IO_map[var_order[0]][1])
        b = ds.expand_levels(B, IO_map[var_order[1]][0], IO_map[var_order[1]][1]).transpose("lev", IO_map[var_order[1]][1])
        c = ds.expand_levels(C, IO_map[var_order[2]][0], IO_map[var_order[2]][1]).transpose("lev", IO_map[var_order[2]][1])
        d = ds.expand_levels(D, IO_map[var_order[3]][0], IO_map[var_order[3]][1]).transpose("lev", IO_map[var_order[3]][1])
    else:
        a = A[IO_map[var_order[0]][0]].to_stacked_array(IO_map[var_order[0]][1], sample_dims=())
        b = B[IO_map[var_order[1]][0]].to_stacked_array(IO_map[var_order[1]][1], sample_dims=())
        c = C[IO_map[var_order[2]][0]].to_stacked_array(IO_map[var_order[2]][1], sample_dims=())
        d = D[IO_map[var_order[3]][0]].to_stacked_array(IO_map[var_order[3]][1], sample_dims=())
    ds.mli, ds.mlo = a.mli, d.mlo
    to_tensor = lambda x : torch.tensor(x.data, dtype=torch.float32)
    if(use_tendencies):
        ds.X_mean, ds.X_max, ds.X_min, ds.Y_scale = A, B, C, D
        ds.xm, ds.xmax, ds.xmin, ds.ys = to_tensor(a), to_tensor(b), to_tensor(c), to_tensor(d)
    else:
        ds.X_mean, ds.X_std, ds.Y_mean, ds.Y_std = A, B, C, D
        ds.xm, ds.xs, ds.ym, ds.ys =  to_tensor(a), to_tensor(b), to_tensor(c), to_tensor(d)





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



def collate_test_fn(batches):
    return(batches[0])






class ClimsimDatasetOld(torch.utils.data.Dataset):
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
