import torch
from torch.utils.data import Dataset, DataLoader
import os
import xarray as xr
import numpy as np
import pandas as pd
import gcsfs 
from dataclasses import dataclass, asdict, field
import inspect
try:
    import diffusers
    diffusers_available = True
except:
    diffusers_available = False


fs = gcsfs.GCSFileSystem()

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_path(file):
    return os.path.join(_ROOT, 'climsim_data', file)

#print(os.path.dirname(__file__))

def get_norm_info(style='npy'):
    if(style=='npy'):    
        X_mean = xr.DataArray(np.load(get_path("X_mean.npy")), coords={'mli' : self.mli})
        X_std = xr.DataArray(np.load(get_path("X_std.npy")), coords={'mli' : self.mli})
        Y_mean = xr.DataArray(np.load(get_path("Y_mean.npy")), coords={'mlo' : self.mlo})
        Y_std = xr.DataArray(np.load(get_path("Y_std.npy")), coords={'mlo' : self.mlo})
        return(X_mean, X_std, Y_mean, Y_std)
    elif(style=='nc'):
        input_mean = xr.open_dataset(get_path('input_mean.nc'))
        input_max = xr.open_dataset(get_path('input_max.nc'))
        input_min = xr.open_dataset(get_path('input_min.nc'))
        output_scale = xr.open_dataset(get_path('output_scale.nc'))
        return(input_mean, input_max, input_min, output_scale)
    


def load_numpy_arrays(bucket='persist', fprefix='climsim'):
    if('scratch' in bucket):
        bucket = "leap-scratch"
    elif('persist'):
        bucket = "leap-persistent"
    xpath = f"gs://{bucket}/sammyagrawal/input_{fprefix}.npy"
    ypath = f"gs://{bucket}/sammyagrawal/output_{fprefix}.npy" 

    with fs.open(xpath, 'rb') as f:
        X = np.load(f)
    print(f"Finished Loading X from {xpath}")
    with fs.open(ypath, 'rb') as f:
        Y = np.load(f)
    print(f"Finished Loading Y from {ypath}")

    return(X, Y)


def save_arrays(X, Y, bucket='scratch', fprefix='climsim'):
    if(bucket== 'scratch'):
        bucket = "leap-scratch"
    elif(bucket == 'persist'):
        bucket = "leap-persistent"
    with fsspec.open(f"gs://{bucket}/sammyagrawal/input_{fprefix}.npy", 'wb') as f:
        np.save(f, X)
    with fsspec.open(f"gs://{bucket}/sammyagrawal/output_{fprefix}.npy", 'wb') as f:
        np.save(f, Y)

def load_scheduler(config):
    def pass_config(func, data_class):
        accepted_params = inspect.signature(func).parameters
        filtered_kwargs = {k: v for k, v in asdict(data_class).items() if k in accepted_params}
        return func(**filtered_kwargs)
    match config.scheduler_type.lower():
        case sched if 'ddpm' in sched:
            # in charge of betas
            return(pass_config(diffusers.DDPMScheduler, config.scheduler))
        case sched if 'ddim' in sched:
            return(pass_config(diffusers.DDIMScheduler, config.scheduler))
        case other if True:
            print(f"scheduler '{other}' not yet supported")
    

def reconstruct_xarr_from_npy(X: np.ndarray, Y: np.ndarray, subsampling=(36,210240, 144), data_vars='v1'):
    ds_in, ds_out = load_raw_dataset(chunks=True)
    input_vars, output_vars = load_vars(data_vars, tendencies=False)
    start, stop, stride = subsampling
    ds_in = ds_in.isel(sample=slice(start, stop, stride))[input_vars]
    ds_out = ds_out.isel(sample=slice(start, stop, stride))[output_vars]

    mli = ds_in.to_stacked_array('mli', sample_dims=['sample', 'ncol']).mli
    mlo = ds_out.to_stacked_array('mlo', sample_dims=['sample', 'ncol']).mlo
    state = ds_in.stack({'state' : ['sample', 'ncol']}).state
    
    Xarr = xr.DataArray(X, dims=['state', 'mli'], coords={'state' : state, 'mli':mli})
    Yarr = xr.DataArray(Y, dims=['state', 'mlo'], coords={'state' : state, 'mlo':mlo})

    #Xarr, Yarr = add_space(Xarr.unstack('sample'), Yarr.unstack('sample'))
    return(Xarr, Yarr)

class ClimsimDataset(Dataset):
    def __init__(self, X, Y, device, normalize=True):
        self.device = device
        self.mli, self.mlo = X.mli, Y.mlo
        self.Xarr, self.Yarr = X, Y
        if(normalize):
            X, Y = self.normalize(X, Y)
        self.X = torch.tensor(X.values, dtype=torch.float32, device=device) # each row is datapoint
        self.Y = torch.tensor(Y.values, dtype=torch.float32, device=device)

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

    def index_var(self, var, level):
        mli, mli = list(self.mli.values), list(self.mlo.values)
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
        
    def __len__(self):
        return(self.Y.shape[0])

    def __getitem__(self, idx):
        return(self.Y[idx])

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


def add_space(ds_in, ds_out, ds_grid=False, lat=False, lon=False):
    if not ds_grid:
        mapper = fs.get_mapper("gs://leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.grid-info.zarr")
        ds_grid = xr.open_dataset(mapper, engine='zarr')
    if not lat or not lon:
        lat = ds_grid.lat.values.round(2) 
        lon = ds_grid.lon.values.round(2)  
        lon = ((lon + 180) % 360) - 180 # convert from 0-360 to -180 to 180
    def add_ll(ds):
        ds = ds.assign_coords({'ncol' : ds.ncol})
        ds['lat'] = (('ncol'),lat.T)
        ds['lon'] = (('ncol'),lon.T)
        ds = ds.assign_coords({'lat' : ds.lat, 'lon' : ds.lon})
        return(ds)
    return(add_ll(ds_in), add_ll(ds_out))


class ClimsimImageDataset(Dataset):
    def __init__(self, X, Y, mconfig, device, channel_first = True, normalize=True):
        self.device = device
        self.mli, self.mlo = X.mli, Y.mlo
        X, Y = add_space(X.unstack('state'), Y.unstack('state'))
        permute_indices = image_regridding(Y)
        X = X[:, :, permute_indices]
        Y = Y[:, :, permute_indices]
        if(normalize):
            self.X_rec = X
            self.Y_rec = Y
            X, Y = self.normalize(X, Y)
        
        self.X = torch.tensor(X.values, device=device, dtype=torch.float32)
        self.Y = torch.tensor(Y.values, device=device, dtype=torch.float32)
        self.height, self.width = mconfig.unet.sample_size
        if(channel_first):
            self.X = torch.permute(self.X, (1,0,2)).reshape(-1, self.mli.shape[0], self.height, self.width) # Channels Height Width
            self.Y = torch.permute(self.Y, (1,0,2)).reshape(-1, mconfig.unet.in_channels, self.height, self.width) # Channels Height Width
        else:
            self.X = torch.permute(self.X, (1,2,0)).reshape(-1, self.height, self.width, self.mli.shape[0]) # Channels Height Width
            self.Y = torch.permute(self.Y, (1,2,0)).reshape(-1, self.height, self.width, mconfig.unet.in_channels) # Channels Height Width
    
        #assert self.X.shape[0] == self.Y.shape[0], "Number of samples does not match"
        if(diffusers_available):
            self.scheduler = load_scheduler(mconfig)
            self.num_timesteps = self.scheduler.config.num_train_timesteps
        else:
            print("diffusers not available, scheduler not imported")
        #self.sample_shape = image_dataset[0].shape
        #self.static_noise = torch.randn(self.sample_shape, device=self.device)
        

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

    def reconstruct_X(self, X_norm):
        X_rec = (X_norm * self.X_std) + self.X_mean
        return(X_rec)
    def reconstruct_Y(self, Y_norm):
        mean, std = torch.tensor(self.Y_mean.values).view(128, 1, 1), torch.tensor(self.Y_std.values).view(128, 1, 1)
        Y_rec = (Y_norm * std) + mean
        return(Y_rec)
    
    def __len__(self):
        return(self.Y.shape[0])

    def __getitem__(self, idx):
        return(self.Y[idx])

    def noise_batch(self, clean_images):
        """
        Given a batch from a dataloader on the dataset, return a noised sample
        """
        # TODO : do performance comparison between this and collate_fn __getitem__
        noise = torch.randn(clean_images.shape, device=self.device)
        timesteps = torch.randint(0, self.num_timesteps, size=(clean_images.shape[0],), device=self.device, dtype=torch.int64)
        noisy_images = self.scheduler.add_noise(clean_images, noise, timesteps)
        return(noisy_images, timesteps, noise)


def load_vars(s, tendencies=True):
    v1_inputs = ['state_t', 'state_q0001', 'state_ps', 'pbuf_SOLIN','pbuf_LHFLX', 'pbuf_SHFLX']

    v1_outputs = ['ptend_t','ptend_q0001','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD','cam_out_SOLLD']

    v2_inputs = ['state_t', 'state_q0001','state_q0002', 'state_q0003', 'state_u', 'state_v',
             'state_ps','pbuf_SOLIN','pbuf_LHFLX', 'pbuf_SHFLX', 'pbuf_TAUX','pbuf_TAUY', 'pbuf_COSZRS',
             'cam_in_ALDIF', 'cam_in_ALDIR', 'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP', 'cam_in_ICEFRAC', 
             'cam_in_LANDFRAC', 'cam_in_OCNFRAC', 'cam_in_SNOWHICE', 'cam_in_SNOWHLAND',
             'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O'] # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 

    v2_outputs = ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v', 'cam_out_NETSW',
              'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS', 'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']

    if(s=='v1'):
        v1_outputs = [var.replace("ptend", "state") if 'ptend' in var else var for var in v1_outputs]
        return(v1_inputs, v1_outputs)
    elif(s=='v2'):
        v2_outputs = [var.replace("ptend", "state") if 'ptend' in var else var for var in v1_outputs]
        return(v2_inputs, v2_outputs)
    print("Input should be v1 or v2")

def load_raw_dataset(ds_type='', data_vars = 'v1', chunks=False, chunksizes={}):
    # change once re-ingested/ virtualized pipeline works
    # eventually want ds_type to specify aquaplanet / res    
    if(chunks):
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.input.zarr')
        ds_input = xr.open_dataset(mapper, engine='zarr', chunks=chunksizes)
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr')
        ds_output = xr.open_dataset(mapper, engine='zarr', chunks=chunksizes)
    else:
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.input.zarr')
        ds_input = xr.open_dataset(mapper, engine='zarr')
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr')
        ds_output = xr.open_dataset(mapper, engine='zarr')
    return(ds_input, ds_output)


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



