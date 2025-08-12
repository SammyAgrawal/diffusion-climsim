import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import glob
import os
import re
#import netCDF4
import h5py
from tqdm import tqdm
from typing import Literal
import gcsfs
import datetime as dt
import cftime
import json
import fsspec
import time
import torch

_ROOT = os.path.abspath(os.path.dirname(__file__))
def get_path(file):
    return os.path.join(_ROOT, 'climsim_data', file)

def load_raw_dataset(dconfig, return_dutils=False, **kwargs):
    dutils = setup_data_utils(dconfig.climsim_type, dconfig.source, dconfig.data_vars, 
                              use_tendencies=dconfig.use_tendencies, data_dir=dconfig.data_dir, **kwargs)
    #ds_type = expand_ds_name(dconfig.climsim_type)
    target_vars = [var.replace("ptend", "state") if 'ptend' in var else var for var in dutils.target_vars]
    if(dconfig.source == "gcsfs"):
        input_path = 'gs://leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.input.zarr'
        output_path = 'gs://leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr'
        dsi = xr.open_dataset(input_path, engine='zarr', chunks=dconfig.chunksize)[dutils.input_vars].rename({"sample" : "time"})
        dso = xr.open_dataset(output_path, engine='zarr', chunks=dconfig.chunksize)[target_vars].rename({"sample" : "time"})

    elif("vzarr" in dconfig.source):
        import icechunk
        storage = icechunk.local_filesystem_storage(os.path.join(dconfig.data_dir, dutils.mlivar))
        repo = icechunk.Repository.open(storage)
        session = repo.writable_session("main")
        with session.allow_pickling():
            dsi = xr.open_zarr(session.store, zarr_format=3, consolidated=False, chunks={})[dutils.input_vars]
        
        storage = icechunk.local_filesystem_storage(os.path.join(dconfig.data_dir, "mlo"))
        repo = icechunk.Repository.open(storage)
        session = repo.writable_session("main")
        with session.allow_pickling():
            dso = xr.open_zarr(session.store, zarr_format=3, consolidated=False, chunks={})[target_vars]
        
    elif(dconfig.source == "huggingface" or dconfig.source == "local"):
        year = int(input("Input year: "))
        month = int(input("Input month: "))
        stride = int(input("Enter stride: "))
        dutils.set_filelist_using_hfhub('train', year, month, stride_sample=stride)
        dsi, dso = dutils.aggregate_file("train")
    
    elif(dconfig.source == "numpy"):
        X, Y = load_numpy_arrays(dconfig)
        fs = gcsfs.GCSFileSystem()
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.input.zarr')
        ds_in = xr.open_dataset(mapper, engine='zarr', chunks=dconfig.chunksize)
        mapper = fs.get_mapper('leap-persistent-ro/sungdukyu/E3SM-MMF_ne4.train.output.zarr')
        ds_out = xr.open_dataset(mapper, engine='zarr', chunks=dconfig.chunksize)
        
        start, stop, stride = dconfig.xarr_subsamples
        ds_in = ds_in.isel(sample=slice(start, stop, stride))[dutils.input_vars]
        ds_out = ds_out.isel(sample=slice(start, stop, stride))[target_vars]
    
        mli = ds_in.to_stacked_array('mli', sample_dims=['sample', 'ncol']).mli
        mlo = ds_out.to_stacked_array('mlo', sample_dims=['sample', 'ncol']).mlo
        state = ds_in.stack({'state' : ['sample', 'ncol']}).state
        
        dsi = xr.DataArray(X, dims=['state', 'mli'], coords={'state' : state, 'mli':mli})
        dso = xr.DataArray(Y, dims=['state', 'mlo'], coords={'state' : state, 'mlo':mlo})
    
    dsi = add_space(dsi, ds_grid=dutils.grid_info)
    dso = add_space(dso, ds_grid=dutils.grid_info)
    if(return_dutils):
        return(dsi, dso, dutils)
    return(dsi, dso)


def dutils_from_config(dconfig):
    dutils = setup_data_utils(dconfig.climsim_type, dconfig.source, dconfig.data_vars, 
                              use_tendencies=dconfig.use_tendencies, data_dir=dconfig.data_dir)
    return(dutils)

def setup_data_utils(ds_type, data_source, data_vars, use_tendencies, **kwargs):
    # data source is either a google cloud bucket, local file path, or tries to load directly from Huggingface
    ds_type = expand_ds_name(ds_type)
    if('grid_info' in kwargs):
        grid_info = kwargs['grid_info']
    else:
        n = expand_ds_name("highres") if "high" in ds_type else expand_ds_name("lowres") 
        grid_url = f"https://huggingface.co/datasets/LEAP/{n}/resolve/main/{n}_grid-info.nc"
        grid_info = read_url(grid_url, copy_to_local=True)
    data = data_utils(data_source, ds_type, grid_info.compute(), use_tendencies)
    if(data_source == 'huggingface'):
        data.data_path = f"https://huggingface.co/datasets/LEAP/{ds_type}/resolve/main/train/"
    elif(data_source == "gcsfs" or "vzarr" in data_source):
        pass
    elif('local' in data_source):
        assert 'base_dir' in kwargs, "Need to provide base path via base_dir arg"
        data.data_path = kwargs['base_dir']
    else:
        print("Invalid data source, must be huggingface, local, or gcsfs")
    if(data_vars == 'v1'):
        data.set_to_v1_vars()
    elif(data_vars == 'v2'):
        data.set_to_v2_vars()
    elif(data_vars == 'all'):
        data.set_to_all_vars()
    return(data)

#print(os.path.dirname(__file__))

def tocft(year=1, month=1, day=1):
    return(cftime.DatetimeNoLeap(year, month, day, has_year_zero=True))

def expand_ds_name(ds_type=''):
    match ds_type.lower():
        case t if "aqua" in t:
            return("ClimSim_low-res_aqua-planet")
        case t if "expand" in t:
            return("ClimSim_low-res-expanded")
        case t if ("low" in t): 
            return("ClimSim_low-res")
        case t if ("high" in t): 
            return("ClimSim_high-res")
        case _:
            print("Unrecognized type")

def read_url(url, copy_to_local=False):
    fs_local = fsspec.filesystem('file')
    fname = "file.nc"
    with fsspec.open(url, mode='rb') as file:
        if(copy_to_local):            
            with open(fname, 'wb') as f:
                f.write(file.read())
            ds = xr.open_dataset(fname, use_cftime=True).load()
            fs_local.rm(fname)
        else:
            ds = xr.open_dataset(file, use_cftime=True).load()   
    
    return(ds)

def add_space(ds, ds_grid=False, lat=False, lon=False, res='low'):
    if not ds_grid:
        n = expand_ds_name("high-res") if "high" in res else expand_ds_name("low-res")
        grid_url = f"https://huggingface.co/datasets/LEAP/{n}/resolve/main/{n}_grid-info.nc"
        ds_grid = read_url(grid_url, copy_to_local=True)
    if not lat or not lon:
        lat = ds_grid.lat.values.round(2) 
        lon = ds_grid.lon.values.round(2)  
        lon = ((lon + 180) % 360) - 180 # convert from 0-360 to -180 to 180
    ds = ds.assign_coords({'ncol' : ds.ncol})
    ds['lat'] = (('ncol'),lat.T)
    ds['lon'] = (('ncol'),lon.T)
    ds = ds.assign_coords({'lat' : ds.lat, 'lon' : ds.lon})
    return(ds)


def image_regridding(ds):
    lat, lon = np.round(ds.lat.data), np.round(ds.lon.data)
    if lon.max() > 180:
        lon = ((lon + 180) % 360) - 180 # convert from 0-360 to -180 to 180
    array = np.column_stack([lon, lat])
    # first sort by longitude, then by latitude (top is area of high longitude)
    sorted_indices = np.lexsort((array[:, 0], -1*array[:, 1]))
    arr = array[sorted_indices]
    indices = np.array([], dtype=int)
    for i in range(16):
        start = i*24
        indices = np.concatenate([indices, start + np.argsort(arr[start:start+24, 0])])

    return(sorted_indices[indices])

def get_norm_info(style='image', sanitize=True):

    if(style == 'scale' or style == 'tendencies'):
        input_mean = xr.open_dataset(get_path('input_mean.nc'))
        input_max = xr.open_dataset(get_path('input_max.nc'))
        input_min = xr.open_dataset(get_path('input_min.nc'))
        output_scale = xr.open_dataset(get_path('output_scale.nc'))
        if(sanitize):
            input_max['pbuf_N2O'].data = input_max.pbuf_N2O.mean().item() * np.ones_like(input_max['pbuf_N2O'].data)
            input_min['pbuf_N2O'].data = input_min.pbuf_N2O.mean().item() * np.ones_like(input_min['pbuf_N2O'].data)
            input_max['pbuf_CH4'].data = input_max.pbuf_CH4.mean().item() * np.ones_like(input_max['pbuf_CH4'].data)
            input_min['pbuf_CH4'].data = input_min.pbuf_CH4.mean().item() * np.ones_like(input_min['pbuf_CH4'].data)
        return(input_mean, input_max, input_min, output_scale)
    
    else:
        X_mean = xr.open_dataset(get_path("image_xmean.nc"))
        X_std = xr.open_dataset(get_path("image_xstd.nc"))
        Y_mean = xr.open_dataset(get_path("image_ymean.nc"))
        Y_std = xr.open_dataset(get_path("image_ystd.nc"))

        if(sanitize):
            X_mean['state_q0002'].data = X_mean['state_q0002'].mean().item() * np.ones_like(X_mean['state_q0002'].data)
            X_std['state_q0002'].data = X_std['state_q0002'].mean().item() * np.ones_like(X_std['state_q0002'].data)
            Y_std['state_q0002'].data = Y_std['state_q0002'].mean().item() * np.ones_like(Y_std['state_q0002'].data)
            Y_std['cam_out_PRECSC'].data = Y_std.cam_out_PRECSC.mean().item() * np.ones_like(Y_std.cam_out_PRECSC.data) 
        if("image" in style):
            return(X_mean, X_std, Y_mean, Y_std)
            
        return(X_mean.mean(dim='ncol'), X_std.mean(dim='ncol'), Y_mean.mean(dim='ncol'), Y_std.mean(dim='ncol'))


def expand_levels(self, ds, vars, dim_name):
    out = [ds[var].expand_dims({'lev': ds.lev}) if var in self.dutils.normal_variables else ds[var] for var in vars]
    out =  xr.concat(out, dim=dim_name)
    out.assign_coords({dim_name : vars})
    return(out)


def imagify(x, dutils, variable='y', image_dim=2):
    if image_dim is None:
        return(x)
    # X is tensor of shape (BS'=BS*ncol, feature_len) where batch size is multipled by 384
    assert variable in ['x', 'y'], "Variable must be either x or y"
    image_dim = int(image_dim)
    if(image_dim == 1):
        # desired output is (BS', C_var, lev) where lev is 64 
        var_map = dutils.input_var_idx if variable == 'x' else dutils.target_var_idx
        ximg = torch.zeros(x.size(0), len(var_map), 64)
        i = 0
        for var, (start, stop) in var_map.items():
            ximg[:,i, -60:] = x[:, start:stop] if stop-start>1 else x[:, start:stop].expand(-1, 60)
            i += 1
    elif(image_dim == 2):
        # desired output is (BS, C_mlv, H, W)
        feature_len = dutils.input_feature_len if variable == 'x' else dutils.target_feature_len
        ximg = x.reshape(-1, 384, feature_len)  # assuming this is dutils.input_feature_len
        ximg = ximg[:, dutils.permute_indices, :].reshape(-1, 16, 24, feature_len).permute(0, 3, 1, 2) 
    elif(image_dim == 3):
        # desired output is (BS, C_var, lev, H, W)
        var_map = dutils.input_var_idx if variable == 'x' else dutils.target_var_idx
        feature_len = dutils.input_feature_len if variable == 'x' else dutils.target_feature_len
        x = x.reshape(-1, 384, feature_len)[:, dutils.permute_indices, :].reshape(-1, 16, 24, feature_len)
        ximg = torch.zeros(x.size(0), len(var_map), 64, 16, 24) # N, C, 64, H, W
        i = 0
        for var, (start, stop) in var_map.items():
            row = x[:, :, :, start:stop] if stop-start>1 else x[:, :, :, start:stop].expand(-1, -1, -1, 60) # N, H, W, 60
            ximg[:, i, -60:, :, :] = row.permute(0, 3, 1, 2)
            i += 1
    else:
        print("invalid dim argument", image_dim, "must be 1, 2, or 3")
    return(ximg)



MLBackendType = Literal["tensorflow", "pytorch"]


fs = gcsfs.GCSFileSystem()

class data_utils:
    ## modified from https://github.com/leap-stc/ClimSim/blob/main/climsim_utils/data_utils.py
    
    def __init__(self, source_type, ds_type, grid_info='', use_tendencies=True, ml_backend: MLBackendType = "pytorch"):
        self.source_type = source_type
        self.ds_type = ds_type
        self.use_tendencies = use_tendencies
        if("expand" in ds_type):
            self.mlivar = "mlexpand"
            self.copy_to_local = False
        else:
            self.mlivar = "mli"
            self.copy_to_local = True
        self.data_path = None
        self.input_vars = []
        self.target_vars = []
        self.input_feature_len = None
        self.target_feature_len = None
        self.level_name = 'lev'
        self.sample_name = 'sample'
        self.num_levels = 60
        self.var_lens = {}
        self.ml_backend = ml_backend
        self.tf = None
        self.torch = None

        if self.ml_backend == "tensorflow":
            self.successful_backend_import = False
            try:
                import tensorflow as tf

                self.tf = tf
                self.successful_backend_import = True
            except ImportError:
                raise ImportError("Tensorflow is not installed.")
        elif self.ml_backend == "pytorch":
            self.successful_backend_import = False

            try:
                import torch
                self.torch = torch
                self.successful_backend_import = True
            except ImportError:
                raise ImportError("PyTorch is not installed.")
    
                self.hyam = self.grid_info['hyam'].values
        
        self.p0 = 1e5 # code assumes this will always be a scalar
        self.ps_index = None

        self.pressure_grid_train = None
        self.pressure_grid_val = None
        self.pressure_grid_scoring = None
        self.pressure_grid_test = None

        self.dp_train = None
        self.dp_val = None
        self.dp_scoring = None
        self.dp_test = None

        self.train_regexps = None
        self.train_stride_sample = None
        self.train_filelist = None
        self.val_regexps = None
        self.val_stride_sample = None
        self.val_filelist = None
        self.scoring_regexps = None
        self.scoring_stride_sample = None
        self.scoring_filelist = None
        self.test_regexps = None
        self.test_stride_sample = None
        self.test_filelist = None

        self.full_vars = False

        # physical constants from E3SM_ROOT/share/util/shr_const_mod.F90
        self.grav    = 9.80616    # acceleration of gravity ~ m/s^2
        self.cp      = 1.00464e3  # specific heat of dry air   ~ J/kg/K
        self.lv      = 2.501e6    # latent heat of evaporation ~ J/kg
        self.lf      = 3.337e5    # latent heat of fusion      ~ J/kg
        self.lsub    = self.lv + self.lf    # latent heat of sublimation ~ J/kg
        self.rho_air = 101325/(6.02214e26*1.38065e-23/28.966)/273.15 # density of dry air at STP  ~ kg/m^3
                                                                    # ~ 1.2923182846924677
                                                                    # SHR_CONST_PSTD/(SHR_CONST_RDAIR*SHR_CONST_TKFRZ)
                                                                    # SHR_CONST_RDAIR   = SHR_CONST_RGAS/SHR_CONST_MWDAIR
                                                                    # SHR_CONST_RGAS    = SHR_CONST_AVOGAD*SHR_CONST_BOLTZ
        self.rho_h20 = 1000       # density of fresh water     ~ kg/m^ 3
        
        self.v1_inputs = ['state_t',
                          'state_q0001',
                          'state_ps',
                          'pbuf_SOLIN',
                          'pbuf_LHFLX',
                          'pbuf_SHFLX']
        
        self.v1_outputs = ['state_t',
                           'state_q0001',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD']

        self.v2_inputs = ['state_t',
                          'state_q0001',
                          'state_q0002',
                          'state_q0003',
                          'state_u',
                          'state_v',
                          'state_ps',
                          'pbuf_SOLIN',
                          'pbuf_LHFLX',
                          'pbuf_SHFLX',
                          'pbuf_TAUX',
                          'pbuf_TAUY',
                          'pbuf_COSZRS',
                          'cam_in_ALDIF',
                          'cam_in_ALDIR',
                          'cam_in_ASDIF',
                          'cam_in_ASDIR',
                          'cam_in_LWUP',
                          'cam_in_ICEFRAC',
                          'cam_in_LANDFRAC',
                          'cam_in_OCNFRAC',
                          'cam_in_SNOWHICE',
                          'cam_in_SNOWHLAND',
                          'pbuf_ozone',
                          'pbuf_CH4',
                          'pbuf_N2O']  # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3

        self.prev_timestep_vars = [f'tm_{var}' for var in self.v2_inputs[:10]] + ['tm_pbuf_COSZRS']
        self.forcing_vars = []
        self.convective_mem_vars = []

        for var in ['state_t', 'state_q0001', 'state_q0002', 'state_q0003', 'state_u']:
            self.convective_mem_vars += [f'{var}_prvphy', f'tm_{var}_prvphy']
            self.var_lens[f'{var}_prvphy'] = self.num_levels
            self.var_lens[f'tm_{var}_prvphy'] = self.num_levels
        
        for var in ['state_t', 'state_q0', 'state_u']:
            self.forcing_vars += [f'{var}_dyn', f'tm_{var}_dyn']
            self.var_lens[f'{var}_dyn'] = self.num_levels
            self.var_lens[f'tm_{var}_dyn'] = self.num_levels

        self.other_expanded_vars = ['clat', 'icol', 'lat', 'lon', 'slat', 'state_pmid', 'tod', 'ymd']


        self.v2_outputs = ['state_t',
                           'state_q0001',
                           'state_q0002',
                           'state_q0003',
                           'state_u',
                           'state_v',
                           'cam_out_NETSW',
                           'cam_out_FLWDS',
                           'cam_out_PRECSC',
                           'cam_out_PRECC',
                           'cam_out_SOLS',
                           'cam_out_SOLL',
                           'cam_out_SOLSD',
                           'cam_out_SOLLD']

        self.all_inputs  = [v for v in self.v2_inputs]
        self.all_outputs = [v for v in self.v2_outputs]
        
        if(self.use_tendencies):
            self.v1_outputs = [var.replace("state", "ptend") if 'state' in var else var for var in self.v1_outputs]
            self.v2_outputs = [var.replace("state", "ptend") if 'state' in var else var for var in self.v2_outputs]
        

        self.var_short_names = {'ptend_t':'$dT/dt$',
                                'ptend_q0001':'$dq/dt$',
                                'cam_out_NETSW':'NETSW',
                                'cam_out_FLWDS':'FLWDS',
                                'cam_out_PRECSC':'PRECSC',
                                'cam_out_PRECC':'PRECC',
                                'cam_out_SOLS':'SOLS',
                                'cam_out_SOLL':'SOLL',
                                'cam_out_SOLSD':'SOLSD',
                                'cam_out_SOLLD':'SOLLD'}
        
        self.target_energy_conv = {'ptend_t':self.cp,
                                   'ptend_q0001':self.lv,
                                   'ptend_q0002':self.lv,
                                   'ptend_q0003':self.lv,
                                   'ptend_wind': None,
                                   'cam_out_NETSW':1.,
                                   'cam_out_FLWDS':1.,
                                   'cam_out_PRECSC':self.lv*self.rho_h20,
                                   'cam_out_PRECC':self.lv*self.rho_h20,
                                   'cam_out_SOLS':1.,
                                   'cam_out_SOLL':1.,
                                   'cam_out_SOLSD':1.,
                                   'cam_out_SOLLD':1.
                                  }
        
        self.variable_metadata = {
            # TODO: Add additional CF metadata, e.g. `standard_name`, to this dict.
            "pbuf_SOLIN": dict(long_name="Solar insolation", units="W/m2"),
            "pbuf_COSZRS": dict(long_name="Cosine of solar zenith angle", units=""),
            "pbuf_LHFLX": dict(long_name="Surface latent heat flux", units="W/m2"),
            "pbuf_SHFLX": dict(long_name="Surface sensible heat flux", units="W/m2"),
            "pbuf_TAUX": dict(long_name="Zonal surface stress", units="N/m2"),
            "pbuf_TAUY": dict(long_name="Meridional surface stress", units="N/m2"),
            "pbuf_ozone": dict(long_name="Ozone volume mixing ratio", units="mol/mol"),
            "pbuf_N2O": dict(long_name="N2O volume mixing ratio", units="mol/mol"),
            "pbuf_CH4": dict(long_name="CH4 volume mixing ratio", units="mol/mol"),
            "state_ps": dict(long_name="Surface pressure", units="Pa"),
            "state_q0001": dict(long_name="Specific humidity", units="kg/kg"),
            "state_q0002": dict(long_name="Cloud liquid mixing ratio", units="kg/kg"),
            "state_q0003": dict(long_name="Cloud ice mixing ratio", units="kg/kg"),
            "state_t": dict(long_name="Air temperature", units="K"),
            "state_u": dict(long_name="Zonal wind speed", units="m/s"),
            "state_v": dict(long_name="Meridional wind speed", units="m/s"),
            "state_pmid": dict(long_name="Mid-level pressure", units="Pa"),
            "cam_in_ASDIR": dict(
                long_name="Albedo for direct shortwave radiation", units=""
            ),
            "cam_in_ASDIF": dict(
                long_name="Albedo for diffuse shortwave radiation", units=""
            ),
            "cam_in_ALDIR": dict(
                long_name="Albedo for direct longwave radiation", units=""
            ),
            "cam_in_ALDIF": dict(
                long_name="Albedo for diffuse longwave radiation", units=""
            ),
            "cam_in_LWUP": dict(long_name="Upward longwave flux", units="W/m2"),
            "cam_in_SNOWHLAND": dict(
                long_name="Snow depth over land (liquid water equivalent)", units="m"
            ),
            "cam_in_SNOWHICE": dict(long_name="Snow depth over ice", units="m"),
            "cam_in_LANDFRAC": dict(long_name="Land areal fraction", units=""),
            "cam_in_ICEFRAC": dict(long_name="Sea-ice areal fraction", units=""),
            "cam_out_NETSW": dict(
                long_name="Net shortwave flux at surface", units="W/m2"
            ),
            "cam_out_FLWDS": dict(
                long_name="Downward longwave flux at surface", units="W/m2"
            ),
            "cam_out_PRECSC": dict(
                long_name="Snow rate (liquid water equivalent)", units="m/s"
            ),
            "cam_out_PRECC": dict(long_name="Rain rate", units="m/s"),
            "cam_out_SOLS": dict(
                long_name="Downward visible direct solar flux to surface", units="W/m2"
            ),
            "cam_out_SOLL": dict(
                long_name="Downward near-infrared direct solar flux to surface",
                units="W/m2",
            ),
            "cam_out_SOLSD": dict(
                long_name="Downward visible diffuse solar flux to surface", units="W/m2"
            ),
            "cam_out_SOLLD": dict(
                long_name="Downward near-infrared diffuse solar flux to surface",
                units="W/m2",
            ),
        }
        
        self.setup_metrics()
        if(grid_info):
            self.setup_grid_info(grid_info)
        
        self.input_mean, self.input_max, self.input_min, self.output_scale = get_norm_info(style='tendencies')
        self.X_mean, self.X_std, self.Y_mean, self.Y_std = get_norm_info(style='states')


    def setup_metrics(self):
        # for metrics
        self.input_train = None
        self.target_train = None
        self.preds_train = None
        self.samplepreds_train = None
        self.target_weighted_train = {}
        self.preds_weighted_train = {}
        self.samplepreds_weighted_train = {}
        self.metrics_train = []
        self.metrics_idx_train = {}
        self.metrics_var_train = {}
        self.input_val = None
        self.target_val = None
        self.preds_val = None
        self.samplepreds_val = None
        self.target_weighted_val = {}
        self.preds_weighted_val = {}
        self.samplepreds_weighted_val = {}
        self.metrics_val = []
        self.metrics_idx_val = {}
        self.metrics_var_val = {}
        
        self.input_scoring = None
        self.target_scoring = None
        self.preds_scoring = None
        self.samplepreds_scoring = None
        self.target_weighted_scoring = {}
        self.preds_weighted_scoring = {}
        self.samplepreds_weighted_scoring = {}
        self.metrics_scoring = []
        self.metrics_idx_scoring = {}
        self.metrics_var_scoring = {}

        self.input_test = None
        self.target_test = None
        self.preds_test = None
        self.samplepreds_test = None
        self.target_weighted_test = {}
        self.preds_weighted_test = {}
        self.samplepreds_weighted_test = {}
        self.metrics_test = []
        self.metrics_idx_test = {}
        self.metrics_var_test = {}

        self.model_names = []
        self.metrics_names = []
        self.metrics_dict = {'MAE': self.calc_MAE,
                             'RMSE': self.calc_RMSE,
                             'R2': self.calc_R2,
                             'CRPS': self.calc_CRPS,
                             'bias': self.calc_bias
                            }
        self.num_CRPS = 32
        self.linecolors = ['#0072B2', 
                           '#E69F00', 
                           '#882255', 
                           '#009E73', 
                           '#D55E00'
                           ]    
    
    
    def normalize(self, x, y):
        assert isinstance(x, type(y)), "x and y must be the same type"
        if(isinstance(x, xr.Dataset)):
            if(self.use_tendencies):
                x = (x - self.input_mean) / (self.input_max - self.input_min)
                y = y * self.output_scale
            else:
                x = (x - self.X_mean) / self.X_std
                y = (y - self.Y_mean) / self.Y_std
        elif(torch.is_tensor(x)):
            if(self.use_tendencies):
                mu = torch.tensor(self.input_mean[self.input_vars].to_stacked_array("mli", sample_dims=()).data, device=x.device)
                imax = torch.tensor(self.input_max[self.input_vars].to_stacked_array("mli", sample_dims=()).data, device=x.device)
                imin = torch.tensor(self.input_min[self.input_vars].to_stacked_array("mli", sample_dims=()).data, device=x.device)
                scale = torch.tensor(self.output_scale[self.target_vars].to_stacked_array("mlo", sample_dims=()).data, device=y.device)
                x = (x - mu) / (imax - imin)
                y = y * self.output_scale
            else:
                xm = torch.tensor(self.X_mean[self.input_vars].to_stacked_array("mli", sample_dims=()).data, device=x.device)
                xs = torch.tensor(self.X_std[self.input_vars].to_stacked_array("mli", sample_dims=()).data, device=x.device)
                ym = torch.tensor(self.Y_mean[self.target_vars].to_stacked_array("mlo", sample_dims=()).data, device=y.device)
                ys = torch.tensor(self.Y_std[self.target_vars].to_stacked_array("mlo", sample_dims=()).data, device=y.device)
                x = (x - xm) / xs
                y = (y - ym) / ys
        else:
            raise ValueError("x and y must be xarray.Dataset or torch.Tensor")

        return(x, y)
    
    def setup_grid_info(self, grid_info):
        self.grid_info = grid_info
        self.num_levels = len(self.grid_info['lev'])
        self.num_latlon = len(self.grid_info['ncol']) # number of unique lat/lon grid points
        # make area-weights
        self.grid_info['area_wgt'] = self.grid_info['area']/self.grid_info['area'].mean(dim = 'ncol')
        self.area_wgt = self.grid_info['area_wgt'].values
        # map ncol to nsamples dimension
        # to_xarray = {'area_wgt':(self.sample_name,np.tile(self.grid_info['area_wgt'], int(n_samples/len(self.grid_info['ncol']))))}
        # to_xarray = xr.Dataset(to_xarray)
        self.lats, self.lats_indices = np.unique(self.grid_info['lat'].values, return_index=True)
        self.lons, self.lons_indices = np.unique(self.grid_info['lon'].values, return_index=True)
        self.sort_lat_key = np.argsort(self.grid_info['lat'].values[np.sort(self.lats_indices)])
        self.sort_lon_key = np.argsort(self.grid_info['lon'].values[np.sort(self.lons_indices)])
        self.indextolatlon = {i: (self.grid_info['lat'].values[i%self.num_latlon], self.grid_info['lon'].values[i%self.num_latlon]) for i in range(self.num_latlon)}
        self.permute_indices = image_regridding(grid_info)
        
        indices_list = []
        for lat in self.lats:
            indices = self.find_keys(self.indextolatlon, lat)
            indices_list.append(indices)
        indices_list.sort(key = lambda x: x[0])
        self.lat_indices_list = indices_list
        self.hybm = self.grid_info['hybm'].values
        self.var_lens.update({#inputs
                 'state_t':self.num_levels,
                 'state_q0001':self.num_levels,
                 'state_q0002':self.num_levels,
                 'state_q0003':self.num_levels,
                 'state_u':self.num_levels,
                 'state_v':self.num_levels,
                 'state_ps':1,
                 'pbuf_SOLIN':1,
                 'pbuf_LHFLX':1,
                 'pbuf_SHFLX':1,
                 'pbuf_TAUX':1,
                 'pbuf_TAUY':1,
                 'pbuf_COSZRS':1,
                 'cam_in_ALDIF':1,
                 'cam_in_ALDIR':1,
                 'cam_in_ASDIF':1,
                 'cam_in_ASDIR':1,
                 'cam_in_LWUP':1,
                 'cam_in_ICEFRAC':1,
                 'cam_in_LANDFRAC':1,
                 'cam_in_OCNFRAC':1,
                 'cam_in_SNOWHICE':1,
                 'cam_in_SNOWHLAND':1,
                 'pbuf_ozone':self.num_levels,
                 'pbuf_CH4':self.num_levels,
                 'pbuf_N2O':self.num_levels,
                 # expanded inputs
                 'clat':1,
                 'icol':1,
                 'lat':1,
                 'lon':1,
                 'slat':1,
                 'state_pmid':self.num_levels,
                 'tod':1,
                 'ymd':1,
                 'tm_state_t':self.num_levels,
                 'tm_state_q0001':self.num_levels,
                 'tm_state_q0002':self.num_levels,
                 'tm_state_q0003':self.num_levels,
                 'tm_state_u':self.num_levels,
                 'tm_state_v':self.num_levels,
                 'tm_state_ps':1,
                 'tm_pbuf_SOLIN':1,
                 'tm_pbuf_LHFLX':1,
                 'tm_pbuf_SHFLX':1,
                 'tm_pbuf_COSZRS':1,
                 #outputs
                 'ptend_t':self.num_levels,
                 'ptend_q0001':self.num_levels,
                 'ptend_q0002':self.num_levels,
                 'ptend_q0003':self.num_levels,
                 'ptend_u':self.num_levels,
                 'ptend_v':self.num_levels,
                 'cam_out_NETSW':1,
                 'cam_out_FLWDS':1,
                 'cam_out_PRECSC':1,
                 'cam_out_PRECC':1,
                 'cam_out_SOLS':1,
                 'cam_out_SOLL':1,
                 'cam_out_SOLSD':1,
                 'cam_out_SOLLD':1
                })
    
    def find_keys(self, dictionary, value):
        keys = []
        for key, val in dictionary.items():
            if val[0] == value:
                keys.append(key)
        return keys
    
    def _make_index_map(self, var_list):
        """
        Given an ordered list of variable names and self.var_lens,
        returns a dict mapping each var → (start_idx, end_idx)
        in the flattened feature vector (end exclusive).
        """
        idx_map = {}
        offset = 0
        for v in var_list:
            length = self.var_lens[v]
            idx_map[v] = (offset, offset + length)
            offset += length
        return idx_map
    
    def set_to_v1_vars(self):
        '''
        This function sets the inputs and outputs to the V1 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v1_inputs
        self.target_vars = self.v1_outputs
        self.ps_index = 120
        self.input_feature_len = 124
        self.target_feature_len = 128
        self.full_vars = False
        self.input_var_idx  = self._make_index_map(self.input_vars)
        self.target_var_idx = self._make_index_map(self.target_vars)
        self.level_variables = [v for v in self.input_vars + self.target_vars if self.var_lens[v] > 1]
        self.normal_variables = [v for v in self.input_vars + self.target_vars if self.var_lens[v] == 1]
        
    def set_to_v2_vars(self):
        '''
        This function sets the inputs and outputs to the V2 subset.
        It also indicates the index of the surface pressure variable.
        '''
        self.input_vars = self.v2_inputs
        self.target_vars = self.v2_outputs
        self.ps_index = 360
        self.input_feature_len = 557
        self.target_feature_len = 368
        self.full_vars = True
        self.input_var_idx  = self._make_index_map(self.input_vars)
        self.target_var_idx = self._make_index_map(self.target_vars)
        self.level_variables = [v for v in self.input_vars + self.target_vars if self.var_lens[v] > 1]
        self.normal_variables = [v for v in self.input_vars + self.target_vars if self.var_lens[v] == 1]
    
    def set_to_all_vars(self):
        self.input_vars = self.v2_inputs + self.prev_timestep_vars + self.forcing_vars + self.convective_mem_vars
        self.target_vars = self.v2_outputs
        self.ps_index = 360
        #self.input_feature_len = 557
        #self.target_feature_len = 368
        self.full_vars = True
        self.input_var_idx  = self._make_index_map(self.input_vars)
        self.target_var_idx = self._make_index_map(self.target_vars)
    
    def get_xrdata(self, file_name, virtual=False, file_vars = None):
        '''
        This function reads in a file and returns an xarray dataset with the variables specified.
        file_vars must be a list of strings.
        '''
        path = os.path.join(self.data_path, file_name)
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        if(self.source_type == 'gcsfs'):
            mapper = fs.get_mapper(path)
            ds = xr.open_dataset(mapper, engine='zarr', chunks={})
        elif(virtual):
            from virtualizarr import open_virtual_dataset
            ds = open_virtual_dataset(path)
        elif(self.source_type == 'huggingface'):
            with fsspec.open(path, mode='rb') as file: 
                if(self.copy_to_local): # non expanded data somehow needs local copy
                    file_name = os.path.split(file_name)[-1]
                    with open(file_name, 'wb') as f:
                        f.write(file.read())
                    ds = xr.open_dataset(file_name, decode_times=time_coder)
                    #fs_local.rm(fname) # if don't wanna save to disk
                else:
                    ds = xr.open_dataset(file, decode_times=time_coder).load()
            #xr.open_dataset(file, engine="h5netcdf", chunks={}, use_cftime=True)  does not work  
        else: # local
            ds = xr.open_dataset(path, engine = 'netcdf4')
        time = self.parse_time(file_name)
        ds = self.add_time(ds, time)
        if(file_vars):
            return(ds[file_vars])
        return ds

    def get_input(self, input_file, virtual=False):
        '''
        This function reads in a file and returns an xarray dataset with the input variables for the emulator.
        '''
        # read inputs
        return self.get_xrdata(input_file, virtual, self.input_vars)

    def get_target(self, input_file, virtual=False):
        '''
        This function reads in a file and returns an xarray dataset with the target variables for the emulator.
        '''
        # read inputs
        if(self.use_tendencies):
            ds_input = self.get_input(input_file)
            ds_target = self.get_xrdata(input_file.replace(f'.{self.mlivar}.','.mlo.'), virtual)
            # each timestep is 20 minutes which corresponds to 1200 seconds
            ds_target['ptend_t'] = (ds_target['state_t'] - ds_input['state_t'])/1200 # T tendency [K/s]
            ds_target['ptend_q0001'] = (ds_target['state_q0001'] - ds_input['state_q0001'])/1200 # Q tendency [kg/kg/s]
            if self.full_vars:
                ds_target['ptend_q0002'] = (ds_target['state_q0002'] - ds_input['state_q0002'])/1200 # Q tendency [kg/kg/s]
                ds_target['ptend_q0003'] = (ds_target['state_q0003'] - ds_input['state_q0003'])/1200 # Q tendency [kg/kg/s]
                ds_target['ptend_u'] = (ds_target['state_u'] - ds_input['state_u'])/1200 # U tendency [m/s/s]
                ds_target['ptend_v'] = (ds_target['state_v'] - ds_input['state_v'])/1200 # V tendency [m/s/s] 
            ds_target = ds_target[self.target_vars]
        else:
            ds_target = self.get_xrdata(input_file.replace(f'.{self.mlivar}.','.mlo.'), virtual, self.target_vars)
        return ds_target
    
    def parse_time(self, filename):
        '''
        if('ymd' in ds.data_vars and 'tod' in ds.data_vars):
            ymd = str(ds.ymd.values)  # e.g., '10201'
            year, month, day = int(ymd[:-4]), int(ymd[-4:-2]), int(ymd[-2:])  # e.g., '10201' -> '1', '02', '01'
            tod_as_minutes = (int(ds.tod.values) // 60)  # e.g., 37200 (sec) // 60 (sec/min) -> 620 min
            hour = tod_as_minutes // 60  # e.g., 620 min // 60 (min/hr) -> 10 hrs
            minute = tod_as_minutes % 60  # e.g., 620 min % 60 (min/hr) -> 20 min
            time = cftime.DatetimeNoLeap(year=year, month=month, day=day, hour=hour, minute=minute)
        elif('time' in ds.dims and not time):
            time = ds.time[0]
        elif(not time or not isinstance(time, cftime._cftime.DatetimeNoLeap)):
            assert False, "Unknown time or incorrect type to add"
        '''
        year_month, file_part = filename.split('/')
        year, month, day, seconds_into_day = map(int, file_part.split('.')[2].split('-'))
        hour, minute = seconds_into_day // 3600, seconds_into_day % 3600 // 60
        return cftime.DatetimeNoLeap(year, month, day, hour, minute)
    
    def add_time(self, ds, time):
        ds = ds.expand_dims(time=xr.CFTimeIndex([time]))
        ds.time.encoding = {
            # xref: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#time-coordinate
            "units": "minutes since 0001-02-01 00:00:00",
            "calendar": "noleap",
            "dtype": "float64",  # Specify numeric dtype for time
            "_FillValue": None,  # Prevent fill values for coordinate
        }
        return(ds)

    def process_ds(self, ds, data_vars = None):
        # ds_type is which Climsim dataset, default aquaplanet
        #try:
        #    assert 'time' in ds.dims
        #except AssertionError as e:
        #    ds = add_time(ds)
        if data_vars is not None:
            ds = ds[data_vars]
        for vname in self.variable_metadata:
            if vname in ds:
                ds[vname].attrs = self.variable_metadata[vname]
        
        lat = self.grid_info.lat.values.round(2) 
        lon = self.grid_info.lon.values.round(2)
        lon = ((lon + 180) % 360) - 180 # convert from 0-360 to -180 to 180
        
        ds['lat'] = (('ncol'),lat.T)
        ds['lon'] = (('ncol'),lon.T)
        ds = ds.assign_coords({'lat' : ds.lat, 'lon' : ds.lon, 'time' : ds.time})
        #ds = ds.merge(self.grid_info[['lat','lon']])
        ds = ds.where((ds['lat']>-999)*(ds['lat']<999), drop=True)
        ds = ds.where((ds['lon']>-999)*(ds['lon']<999), drop=True)
        return(ds)    
    
    def set_filelist_using_regexps(self, data_split, regexps, stride_sample):
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        
        filelist = []
        for regexp in regexps:
            if(self.source_type == 'local'):
                filelist = filelist + glob.glob(self.data_path + "*/" + regexp)
            elif(self.source_type == 'gcsfs'):
                file_names = fs.ls(self.data_path)
                pattern = re.compile(regexp)
                filelist = filelist + [os.path.join(self.data_path, file) for file in file_names if pattern.search(file)]
            else:
                print(f"Only local and gcsfs source types supported for regex. Current {self.source_type=}")
        
        if data_split == 'train':
            self.train_regexps = regexps
            self.train_stride_sample = stride_sample
            self.train_filelist = sorted(filelist)[::stride_sample]
        elif data_split == 'val':
            self.val_regexps = regexps
            self.val_stride_sample = stride_sample
            self.val_filelist = sorted(filelist)[::stride_sample]
        elif data_split == 'scoring':
            self.scoring_regexps = regexps
            self.scoring_stride_sample = stride_sample
            self.scoring_filelist = sorted(filelist)[::stride_sample]
        elif data_split == 'test':
            self.test_regexps = regexps
            self.test_stride_sample = stride_sample
            self.test_filelist = sorted(filelist)[::stride_sample]
    
    def set_filelist_using_intervals(self, data_split, start, stop, interval):
        """
        takes in cftime of start and stop date as well as time interval as datatime timedelta. 
        """
        if(isinstance(interval,int)):
            interval = dt.timedelta(minutes=interval)

        assert interval.total_seconds()%1200==0, "Interval not multiple of 20 minutes"
        num_deltas = (stop - start).total_seconds() // interval.total_seconds()
        filepaths = []
        for i in range(int(num_deltas)):
            time = start + (interval * i)
            seconds = (time.hour * 3600) + (time.minute * 60)
            fname = (
                f"{time.year:04}-{time.month:02}/E3SM-MMF.{self.mlivar}."
                f"{time.year:04}-{time.month:02}-{time.day:02}-{seconds:05}.nc"
            )
            filepaths.append(fname)
        
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            self.train_filelist = filepaths
        elif data_split == 'val':
            self.val_filelist = filepaths
        elif data_split == 'scoring':
            self.scoring_filelist = filepaths
        elif data_split == 'test':
            self.test_filelist = filepaths

    def set_filelist_using_hfhub(self, data_split, year, month, stride_sample=1):
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        filepaths = fs.glob(f"datasets/LEAP/{self.ds_type}/train/000{year}-{month:02d}/*.nc")
        filepaths = [f.split("train/")[1] for f in filepaths if self.mlivar in f]
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            self.train_filelist = filepaths[::stride_sample]
        elif data_split == 'val':
            self.val_filelist = filepaths[::stride_sample]
        elif data_split == 'scoring':
            self.scoring_filelist = filepaths[::stride_sample]
        elif data_split == 'test':
            self.test_filelist = filepaths[::stride_sample]

    def get_filelist(self, data_split):
        '''
        This function returns the filelist corresponding to data splits for train, val, scoring, and test.
        '''
        assert data_split in ['train', 'val', 'scoring', 'test'], 'Provided data_split is not valid. Available options are train, val, scoring, and test.'
        if data_split == 'train':
            assert self.train_filelist is not None, 'filelist for train is not set.'
            return self.train_filelist
        elif data_split == 'val':
            assert self.val_filelist is not None, 'filelist for val is not set.'
            return self.val_filelist
        elif data_split == 'scoring':
            assert self.scoring_filelist is not None, 'filelist for scoring is not set.'
            return self.scoring_filelist
        elif data_split == 'test':
            assert self.test_filelist is not None, 'filelist for test is not set.'
            return self.test_filelist

    def aggregate_file(self, data_split, virtual=False, normalize=True):
        filelist = self.get_filelist(data_split)
        print(f"Aggregating {len(filelist)} files")
        ds_inputs, ds_targets = [], []
        start = time.time()
        for i, file in enumerate(filelist):
            if(i%5==0):
                t = time.time() - start
                print(f"Processed {i} files in {t:.2f}")
            # read inputs
            ds_input = self.get_input(file, virtual)
            # read targets
            ds_target = self.get_target(file, virtual)
            
            # normalization, scaling
            if normalize:
                ds_input, ds_target = self.normalize(ds_input, ds_target)
                
            ds_inputs.append(ds_input)
            ds_targets.append(ds_target)
        try:
            ds_inputs = xr.concat(ds_inputs, dim='time')
            ds_targets = xr.concat(ds_targets, dim='time')
        finally:
            return(ds_inputs, ds_targets)        
    
    def load_generator(self, data_split):
        filelist = self.get_filelist(data_split)
        def gen():
            for file in filelist:
                # read inputs
                ds_input = self.get_input(file)
                # read targets
                ds_target = self.get_target(file)
                yield (ds_input, ds_target)
        return(gen)
    
    def load_ncdata_with_generator(self, data_split, normalize=True):
        '''
        This function works as a dataloader when training the emulator with raw netCDF files.
        This can be used as a dataloader during training or it can be used to create entire datasets.
        When used as a dataloader for training, I/O can slow down training considerably.
        This function also normalizes the data.
        mli corresponds to input
        mlo corresponds to target
        '''
        gen = self.load_generator(data_split)
        if self.ml_backend == "tensorflow":

            # Removed output_shapes and output_types, converting to output_signature as is
            # recommended in the latest version of TensorFlow.
            return self.tf.data.Dataset.from_generator(
                gen, 
                output_signature=(
                    self.tf.TensorSpec(shape=(None, self.input_feature_len), dtype=self.tf.float64),
                    self.tf.TensorSpec(shape=(None, self.target_feature_len), dtype=self.tf.float64)
                )
            )

        elif self.ml_backend == "pytorch":
            if self.successful_backend_import:

                class IterableTorchDataset(self.torch.utils.data.IterableDataset):
                    def __init__(this_self, data_generator, output_types, output_shapes):
                        this_self.data_generator = data_generator
                        this_self.output_types = output_types
                        this_self.output_shapes = output_shapes

                    def __iter__(this_self):
                        for (inp, out) in this_self.data_generator:
                            inp = inp.drop(['lat','lon'])
                            if normalize:
                                inp, out = self.normalize(inp, out)
                            inp = inp.stack({'batch':{'ncol'}})
                            inp = inp.to_stacked_array('mlvar', sample_dims=['batch'], name='mli')
                            # dso = dso.stack({'batch':{'sample','ncol'}})
                            out = out.stack({'batch':{'ncol'}})
                            out = out.to_stacked_array('mlvar', sample_dims=['batch'], name='mlo')
                            input_array = self.torch.tensor(inp.values, dtype=this_self.output_types[0])
                            target_array = self.torch.tensor(out.values, dtype=this_self.output_types[1])

                            # Assert final dimensions are correct.
                            assert (
                                input_array.shape[-1] == this_self.output_shapes[0][-1]
                            )
                            assert (
                                target_array.shape[-1] == this_self.output_shapes[1][-1]
                            )

                            yield (input_array, target_array)

                    def as_numpy_iterator(this_self):
                        for item in this_self.data_generator:

                            # Convert item to numpy array
                            input_array = np.array(item[0])
                            target_array = np.array(item[1])

                            # Assert final dimensions are correct.
                            assert input_array.shape[-1] == this_self.output_shapes[0][-1]
                            assert target_array.shape[-1] == this_self.output_shapes[1][-1]

                            yield (input_array, target_array)

                dataset = IterableTorchDataset(
                    gen(),
                    (self.torch.float64, self.torch.float64),
                    ((None, self.input_feature_len), (None, self.target_feature_len)),
                )

                return dataset
    
    def save_as_npy(self, data_split, save_path = '', save_latlontime_dict = False):
        '''
        This function saves the training data as a .npy file.
        '''
        data_loader = self.load_ncdata_with_generator(data_split)
        npy_iterator = list(data_loader.as_numpy_iterator())
        npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
        npy_target = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
        with open(save_path + data_split + '_input.npy', 'wb') as f:
            np.save(f, np.float32(npy_input))
        with open(save_path + data_split + '_target.npy', 'wb') as f:
            np.save(f, np.float32(npy_target))
        if data_split == 'train':
            data_files = self.train_filelist
        elif data_split == 'val':
            data_files = self.val_filelist
        elif data_split == 'scoring':
            data_files = self.scoring_filelist
        elif data_split == 'test':
            data_files = self.test_filelist
        if save_latlontime_dict:
            dates = [re.sub(r'^.*mli\.', '', x) for x in data_files]
            dates = [re.sub(r'\.nc$', '', x) for x in dates]
            repeat_dates = []
            for date in dates:
                for i in range(self.num_latlon):
                    repeat_dates.append(date)
            latlontime = {i: [(self.grid_info['lat'].values[i%self.num_latlon], self.grid_info['lon'].values[i%self.num_latlon]), repeat_dates[i]] for i in range(npy_input.shape[0])}
            with open(save_path + data_split + '_indextolatlontime.pkl', 'wb') as f:
                pickle.dump(latlontime, f)
    
    def reshape_npy(self, var_arr, var_arr_dim):
        '''
        This function reshapes the a variable in numpy such that time gets its own axis (instead of being num_samples x num_levels).
        Shape of target would be (timestep, lat/lon combo, num_levels)
        '''
        var_arr = var_arr.reshape((int(var_arr.shape[0]/self.num_latlon), self.num_latlon, var_arr_dim))
        return var_arr

    @staticmethod
    def ls(dir_path = ''):
        '''
        You can treat this as a Python wrapper for the bash command "ls".
        '''
        return os.popen(' '.join(['ls', dir_path])).read().splitlines()
    
    @staticmethod
    def set_plot_params():
        '''
        This function sets the plot parameters for matplotlib.
        '''
        plt.close('all')
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rc('font', family='sans')
        plt.rcParams.update({'font.size': 32,
                            'lines.linewidth': 2,
                            'axes.labelsize': 32,
                            'axes.titlesize': 32,
                            'xtick.labelsize': 32,
                            'ytick.labelsize': 32,
                            'legend.fontsize': 32,
                            'axes.linewidth': 2,
                            "pgf.texsystem": "pdflatex"
                            })
        # %config InlineBackend.figure_format = 'retina'
        # use the above line when working in a jupyter notebook

    @staticmethod
    def load_npy_file(load_path = ''):
        '''
        This function loads the prediction .npy file.
        '''
        with open(load_path, 'rb') as f:
            pred = np.load(f)
        return pred
    
    @staticmethod
    def load_h5_file(load_path = ''):
        '''
        This function loads the prediction .h5 file.
        '''
        hf = h5py.File(load_path, 'r')
        pred = np.array(hf.get('pred'))
        return pred

    def compute_dp(self, inp_data, undo_norm=False):
        '''
        This function sets the pressure weighting for metrics.
        '''
        assert inp_data.shape[0] % self.num_latlon == 0, f"Input data must be divisible by number of lat/lon points ({self.num_latlon})"
        assert len(inp_data.shape) == 2 and inp_data.shape[1] == self.input_feature_len, "Expecting (batch_size, mli) size array"
        if(torch.is_tensor(inp_data)):
            inp_data = inp_data.detach().cpu().numpy()

        state_ps = inp_data[:,self.ps_index]
        if undo_norm:
            state_ps = state_ps*(self.input_max['state_ps'].values - self.input_min['state_ps'].values) + self.input_mean['state_ps'].values
        state_ps = np.reshape(state_ps, (-1, self.num_latlon)) # (time, ncol)
        pressure_grid_p1 = np.array(self.grid_info['P0']*self.grid_info['hyai'])[:,np.newaxis,np.newaxis]
        pressure_grid_p2 = self.grid_info['hybi'].values[:, np.newaxis, np.newaxis] * state_ps[np.newaxis, :, :]
        pressure_grid = pressure_grid_p1 + pressure_grid_p2
        dp = pressure_grid[1:61,:,:] - pressure_grid[0:60,:,:]
        dp = dp.transpose((1,2,0))
        return(dp) # time_bs, 384, 60
        
    def set_pressure_grid(self, data_split):
        '''
        This function sets the pressure weighting for metrics.
        '''
        assert data_split in ['train','val','scoring','test']
        arr = getattr(self, f"input_{data_split}")
        assert arr is not None, f"input_{data_split} is not set"
        dp = self.compute_dp(arr)
        setattr(self, f"dp_{data_split}", dp)

    def get_pressure_grid_plotting(self, data_split):
        '''
        This function creates the temporally and zonally averaged pressure grid corresponding to a given data split.
        '''
        filelist = self.get_filelist(data_split)
        ps = np.concatenate(
            [self.get_xrdata(file, file_vars=['state_ps'])['state_ps'].values[np.newaxis, :] for file in tqdm(filelist)],
            axis = 0)[:, :, np.newaxis]
        hyam_component = self.hyam[np.newaxis, np.newaxis, :]*self.p0
        hybm_component = self.hybm[np.newaxis, np.newaxis, :]*ps
        pressures = np.mean(hyam_component + hybm_component, axis = 0)
        pg_lats = []

        for lat in self.lats:
            indices = self.find_keys(self.indextolatlon, lat)
            pg_lats.append(np.mean(pressures[indices, :], axis = 0)[:, np.newaxis])
        pressure_grid_plotting = np.concatenate(pg_lats, axis = 1)
        return pressure_grid_plotting

    def denormalize(self, x, y, norm_method='nc'):
        if(torch.is_tensor(x)):
            x = x.detach().cpu().numpy()
        if(torch.is_tensor(y)):
            y = y.detach().cpu().numpy()
        
        if(norm_method == 'scale' or norm_method=='tendencies' or self.use_tendencies):
            mu = self.input_mean[self.input_vars].to_stacked_array(new_dim="mli", sample_dims=()).data
            imax = self.input_max[self.input_vars].to_stacked_array(new_dim='mli', sample_dims=()).data
            imin = self.input_min[self.input_vars].to_stacked_array(new_dim='mli', sample_dims=()).data
            scale_stacked = (
                self.output_scale[self.target_vars]
                .to_stacked_array(new_dim="mlo", sample_dims=())   # convert all vars→DataArray with dim 'mlo'
                .transpose("mlo",)                                 # shape: (368,)
            )
            return(x * (imax-imin) + mu, y / scale_stacked.values)
        else:
            xm = self.X_mean[self.input_vars].to_stacked_array('mli', sample_dims=()).data
            xs = self.X_std[self.input_vars].to_stacked_array('mli', sample_dims=()).data
            ym = self.Y_mean[self.target_vars].to_stacked_array('mlo', sample_dims=()).data
            ys = self.Y_std[self.target_vars].to_stacked_array('mlo', sample_dims=()).data
            x = x * xs + xm
            y = y * ys + ym
            return(x, y)

    def get_var_weights(self, output, dp, ret_type='dict'):
        if(torch.is_tensor(output)):
            output = output.detach().cpu().numpy()
        num_samples = output.shape[0]
        vert_levels_weight = dp / self.grav
        weight_mat = np.ones(output.shape)
        weightings = {}
        if(self.full_vars):
            ptend_u = output[:,240:300].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            ptend_v = output[:,300:360].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
            state_wind = ((ptend_u**2) + (ptend_v**2))**.5
            self.target_energy_conv['ptend_wind'] = state_wind
        
        for var_name, (start,stop) in self.target_var_idx.items():
            if(var_name in self.level_variables):
                w = np.ones((int(num_samples/self.num_latlon), self.num_latlon, 60))
                w *= vert_levels_weight * self.area_wgt[np.newaxis, :, np.newaxis]
            else:
                w = np.ones((int(num_samples/self.num_latlon), self.num_latlon))
                w *= self.area_wgt[np.newaxis, :]

            if(var_name not in self.target_energy_conv):
                if( "state" in var_name and var_name.replace("state", "ptend") in self.target_energy_conv):
                    energy_conv = self.target_energy_conv[var_name.replace("state", "ptend")]
                    w = w * energy_conv / 1200
                elif(var_name in ["state_u", "state_v", "ptend_u", "ptend_v"]):
                    w = w * w['ptend_wind']
            else:
                w = w * self.target_energy_conv[var_name]

            weightings[var_name] = w
            weight_mat[:, start:stop] = w.reshape(num_samples, -1)
        
        if(ret_type in ["dict", "map"]):
            return weightings
        else:
            return(weight_mat)


    def output_weighting(self, inp_data, output, undo_norm=True):
        '''
        This function does four transformations, and assumes we are using V1 variables:
        [0] Undos the output scaling
        [1] Weight vertical levels by dp/g
        [2] Weight horizontal area of each grid cell by a[x]/mean(a[x])
        [3] Unit conversion to a common energy unit
        '''
        if(undo_norm):
            inp_data, output = self.denormalize(inp_data, output)
        dp = self.compute_dp(inp_data, undo_norm=False)
        var_weights = self.get_var_weights(output, dp, "dict")
        var_dict = {}
        for var_name, (start,stop) in self.target_var_idx.items():
            assert (output[:, start:stop].shape == var_weights[var_name].reshape(-1, int(stop-start)).shape), f"Shapes for {var_name} are not matching"
            if(var_name in self.level_variables):
                output_val = output[:, start:stop].reshape(-1, self.num_latlon, 60)
            else:
                output_val = output[:, start:stop].reshape(-1, self.num_latlon)
            var_dict[var_name] = output_val * var_weights[var_name]
        return(var_dict)

        
    def calc_MAE(self, pred, target, avg_grid = True):
        '''
        calculate 'globally averaged' mean absolute error 
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        mae = np.abs(pred - target).mean(axis = 0)
        if avg_grid:
            return mae.mean(axis = 0) # we decided to average globally at end
        else:
            return mae
    
    def calc_RMSE(self, pred, target, avg_grid = True):
        '''
        calculate 'globally averaged' root mean squared error 
        for vertically-resolved variables, shape should be time x grid x level
        for scalars, shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        sq_diff = (pred - target)**2
        rmse = np.sqrt(sq_diff.mean(axis = 0)) # mean over time
        if avg_grid:
            return rmse.mean(axis = 0) # we decided to separately average globally at end
        else:
            return rmse

    def calc_R2(self, pred, target, avg_grid = True):
        '''
        calculate 'globally averaged' R-squared
        for vertically-resolved variables, input shape should be time x grid x level
        for scalars, input shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        sq_diff = (pred - target)**2
        tss_time = (target - target.mean(axis = 0)[np.newaxis, ...])**2 # mean over time
        r_squared = 1 - sq_diff.sum(axis = 0)/tss_time.sum(axis = 0) # sum over time
        if avg_grid:
            return r_squared.mean(axis = 0) # we decided to separately average globally at end
        else:
            return r_squared
    
    def calc_bias(self, pred, target, avg_grid = True):
        '''
        calculate bias
        for vertically-resolved variables, input shape should be time x grid x level
        for scalars, input shape should be time x grid

        returns vector of length level or 1
        '''
        assert pred.shape[1] == self.num_latlon
        assert pred.shape == target.shape
        bias = pred.mean(axis = 0) - target.mean(axis = 0)
        if avg_grid:
            return bias.mean(axis = 0) # we decided to separately average globally at end
        else:
            return bias
        
    def calc_CRPS(self, samplepreds, target, avg_grid = True):
        '''
        calculate 'globally averaged' continuous ranked probability score
        for vertically-resolved variables, input shape should be time x grid x level x num_crps_samples
        for scalars, input shape should be time x grid x num_crps_samples

        returns vector of length level or 1
        '''
        assert samplepreds.shape[1] == self.num_latlon
        assert len(samplepreds.shape) == len(target.shape) + 1
        assert len(samplepreds.shape) == 3 or len(samplepreds.shape) == 4
        num_crps = samplepreds.shape[-1]
        mae = np.mean(np.abs(samplepreds - target[..., np.newaxis]), axis = (0, -1)) # mean over time and crps samples
        samplepreds = np.sort(samplepreds, axis = -1)
        diff = samplepreds[..., 1:] - samplepreds[..., :-1]
        count = np.arange(1, num_crps) * np.arange(num_crps - 1, 0, -1)
        if len(samplepreds.shape) == 4:
            spread = (diff * count[np.newaxis, np.newaxis, np.newaxis, :]).sum(axis = -1).mean(axis = 0) # sum over crps samples and mean over time
        elif len(samplepreds.shape) == 3:
            spread = (diff * count[np.newaxis, np.newaxis, :]).sum(axis = -1).mean(axis = 0) # sum over crps samples and mean over time
        crps = mae - spread/(num_crps*(num_crps-1))
        # count was not multiplied by two so no need to divide by two
        if avg_grid:
            return crps.mean(axis = 0) # we decided to separately average globally at end
        else:
            return crps

    def create_metrics_df(self, x, y, predictions_dict, apply_weighting=True):
        '''
        creates a dataframe of metrics for each model
        predictions_matrix is a dict <model_name, weighted predictions y_hat>
        Both of these simply aply the output_weighting function. 
        '''
        assert len(self.metrics_names) != 0, "must specify metrics first"
        assert len(self.target_vars) != 0
        assert self.target_feature_len is not None
        metrics_var_train = {}
        metrics_idx_train = {}
        if apply_weighting:
            msg = "Applying variable reweighting to y" if self.use_tendencies else "Applying variable reweighting to y even though use_tendencies is False! Proceed with caution."
            print(msg)
            y = self.output_weighting(x, y, undo_norm=True)
        if(torch.is_tensor(y)):
            y = y.cpu().detach().numpy()
        
        for model_name, preds in predictions_dict.items():
            if apply_weighting:
                msg = "Applying variable reweighting to prediction" if self.use_tendencies else "Applying variable reweighting to prediction even though use_tendencies is False! Proceed with caution."
                print(msg)
                preds = self.output_weighting(x, preds, undo_norm=True)
            if(torch.is_tensor(preds)):
                preds = preds.cpu().detach().numpy()
                
            df_var = pd.DataFrame(columns = self.metrics_names, index = self.target_vars)
            df_var.index.name = 'variable'
            df_idx = pd.DataFrame(columns = self.metrics_names, index = range(self.target_feature_len))
            df_idx.index.name = 'output_idx'
            for metric_name in self.metrics_names:
                current_idx = 0
                metric_fn = self.metrics_dict[metric_name]
                for target_var in self.target_vars:
                    if(apply_weighting):
                        pred_var, y_var = preds[target_var], y[target_var]
                    else:
                        start, stop = self.target_var_idx[target_var]
                        if(target_var in self.level_variables):
                            pred_var = preds[:, start:stop].reshape(-1, self.num_latlon, 60)
                            y_var = y[:, start:stop].reshape(-1, self.num_latlon, 60)
                        else:
                            pred_var = preds[:, start:stop].reshape(-1, self.num_latlon)
                            y_var = y[:, start:stop].reshape(-1, self.num_latlon)
                    metric = metric_fn(pred_var, y_var)
                    df_var.loc[target_var, metric_name] = np.mean(metric)
                    df_idx.loc[current_idx:current_idx + self.var_lens[target_var] - 1, metric_name] = np.atleast_1d(metric)
                    current_idx += self.var_lens[target_var]
            metrics_var_train[model_name] = df_var
            metrics_idx_train[model_name] = df_idx
        return(metrics_var_train, metrics_idx_train)

    def reshape_daily(self, output):
        '''
        This function returns two numpy arrays, one for each vertically resolved variable (ptend_t and ptend_q0001).
        Dimensions of expected input are num_samples by 128 (number of target features).
        Output argument is espected to be have dimensions of num_samples by features.
        ptend_t is expected to be the first feature, and ptend_q0001 is expected to be the second feature.
        Data is expected to use a stride_sample of 6. (12 samples per day, 20 min timestep).
        '''
        num_samples = output.shape[0]
        ptend_t = output[:,:60].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
        ptend_q0001 = output[:,60:120].reshape((int(num_samples/self.num_latlon), self.num_latlon, 60))
        ptend_t_daily = np.mean(ptend_t.reshape((ptend_t.shape[0]//12, 12, self.num_latlon, 60)), axis = 1) # Nday x lotlonnum x 60
        ptend_q0001_daily = np.mean(ptend_q0001.reshape((ptend_q0001.shape[0]//12, 12, self.num_latlon, 60)), axis = 1) # Nday x lotlonnum x 60
        ptend_t_daily_long = []
        ptend_q0001_daily_long = []
        for i in range(len(self.lats)):
            ptend_t_daily_long.append(np.mean(ptend_t_daily[:,self.lat_indices_list[i],:],axis=1))
            ptend_q0001_daily_long.append(np.mean(ptend_q0001_daily[:,self.lat_indices_list[i],:],axis=1))
        ptend_t_daily_long = np.array(ptend_t_daily_long) # lat x Nday x 60
        ptend_q0001_daily_long = np.array(ptend_q0001_daily_long) # lat x Nday x 60
        return ptend_t_daily_long, ptend_q0001_daily_long

    def plot_r2_analysis(self, pressure_grid_plotting, save_path = ''):
        '''
        This function plots the R2 pressure latitude figure shown in the SI.
        '''
        self.set_plot_params()
        n_model = len(self.model_names)
        fig, ax = plt.subplots(2,n_model, figsize=(n_model*12,18))
        y = np.array(range(60))
        X, Y = np.meshgrid(np.sin(self.lats*np.pi/180), y)
        Y = pressure_grid_plotting/100
        test_heat_daily_long, test_moist_daily_long = self.reshape_daily(self.target_scoring)
        for i, model_name in enumerate(self.model_names):
            pred_heat_daily_long, pred_moist_daily_long = self.reshape_daily(self.preds_scoring[model_name])
            coeff = 1 - np.sum( (pred_heat_daily_long-test_heat_daily_long)**2, axis=1)/np.sum( (test_heat_daily_long-np.mean(test_heat_daily_long, axis=1)[:,None,:])**2, axis=1)
            coeff = coeff[self.sort_lat_key,:]
            coeff = coeff.T
            
            contour_plot = ax[0,i].pcolor(X, Y, coeff,cmap='Blues', vmin = 0, vmax = 1) # pcolormesh
            ax[0,i].contour(X, Y, coeff, [0.7], colors='orange', linewidths=[4])
            ax[0,i].contour(X, Y, coeff, [0.9], colors='yellow', linewidths=[4])
            ax[0,i].set_ylim(ax[0,i].get_ylim()[::-1])
            ax[0,i].set_title(self.model_names[i] + " - ptend_t")
            ax[0,i].set_xticks([])
            
            coeff = 1 - np.sum( (pred_moist_daily_long-test_moist_daily_long)**2, axis=1)/np.sum( (test_moist_daily_long-np.mean(test_moist_daily_long, axis=1)[:,None,:])**2, axis=1)
            coeff = coeff[self.sort_lat_key,:]
            coeff = coeff.T
            
            contour_plot = ax[1,i].pcolor(X, Y, coeff,cmap='Blues', vmin = 0, vmax = 1) # pcolormesh
            ax[1,i].contour(X, Y, coeff, [0.7], colors='orange', linewidths=[4])
            ax[1,i].contour(X, Y, coeff, [0.9], colors='yellow', linewidths=[4])
            ax[1,i].set_ylim(ax[1,i].get_ylim()[::-1])
            ax[1,i].set_title(self.model_names[i] + " - ptend_q0001")
            ax[1,i].xaxis.set_ticks([np.sin(-50/180*np.pi), 0, np.sin(50/180*np.pi)])
            ax[1,i].xaxis.set_ticklabels([r'50$^\circ$S', r'0$^\circ$', r'50$^\circ$N'])
            ax[1,i].xaxis.set_tick_params(width = 2)
            
            if i != 0:
                ax[0,i].set_yticks([])
                ax[1,i].set_yticks([])
                
        # lines below for x and y label axes are valid if 3 models are considered
        # we want to put only one label for each axis
        # if nbr of models is different from 3 please adjust label location to center it

        #ax[1,1].xaxis.set_label_coords(-0.10,-0.10)

        ax[0,0].set_ylabel("Pressure [hPa]")
        ax[0,0].yaxis.set_label_coords(-0.2,-0.09) # (-1.38,-0.09)
        ax[0,0].yaxis.set_ticks([1000,800,600,400,200,0])
        ax[1,0].yaxis.set_ticks([1000,800,600,400,200,0])
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.12, 0.02, 0.76])
        cb = fig.colorbar(contour_plot, cax=cbar_ax)
        cb.set_label("Skill Score "+r'$\left(\mathrm{R^{2}}\right)$',labelpad=50.1)
        plt.suptitle("Baseline Models Skill for Vertically Resolved Tendencies", y = 0.97)
        plt.subplots_adjust(hspace=0.13)
        plt.show()
        plt.savefig(save_path + 'press_lat_diff_models.png', bbox_inches='tight', pad_inches=0.1 , dpi = 300)
    
    @staticmethod
    def reshape_input_for_cnn(npy_input, save_path = ''):
        '''
        This function reshapes a numpy input array to be compatible with CNN training.
        Each variable becomes its own channel.
        For the input there are 6 channels, each with 60 vertical levels.
        The last 4 channels correspond to scalars repeated across all 60 levels.
        This is for V1 data only! (V2 data has more variables)
        '''
        npy_input_cnn = np.stack([
            npy_input[:, 0:60],
            npy_input[:, 60:120],
            np.repeat(npy_input[:, 120][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_input[:, 121][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_input[:, 122][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_input[:, 123][:, np.newaxis], 60, axis = 1)], axis = 2)
        
        if save_path != '':
            with open(save_path + 'train_input_cnn.npy', 'wb') as f:
                np.save(f, np.float32(npy_input_cnn))
        return npy_input_cnn
    
    @staticmethod
    def reshape_target_for_cnn(npy_target, save_path = ''):
        '''
        This function reshapes a numpy target array to be compatible with CNN training.
        Each variable becomes its own channel.
        For the input there are 6 channels, each with 60 vertical levels.
        The last 4 channels correspond to scalars repeated across all 60 levels.
        This is for V1 data only! (V2 data has more variables)
        '''
        npy_target_cnn = np.stack([
            npy_target[:, 0:60],
            npy_target[:, 60:120],
            np.repeat(npy_target[:, 120][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 121][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 122][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 123][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 124][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 125][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 126][:, np.newaxis], 60, axis = 1),
            np.repeat(npy_target[:, 127][:, np.newaxis], 60, axis = 1)], axis = 2)
        
        if save_path != '':
            with open(save_path + 'train_target_cnn.npy', 'wb') as f:
                np.save(f, np.float32(npy_target_cnn))
        return npy_target_cnn
    
    @staticmethod
    def reshape_target_from_cnn(npy_predict_cnn, save_path = ''):
        '''
        This function reshapes CNN target to (num_samples, 128) for standardized metrics.
        This is for V1 data only! (V2 data has more variables)
        '''
        npy_predict_cnn_reshaped = np.concatenate([
            npy_predict_cnn[:,:,0],
            npy_predict_cnn[:,:,1],
            np.mean(npy_predict_cnn[:,:,2], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,3], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,4], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,5], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,6], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,7], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,8], axis = 1)[:, np.newaxis],
            np.mean(npy_predict_cnn[:,:,9], axis = 1)[:, np.newaxis]], axis = 1)
        
        if save_path != '':
            with open(save_path + 'cnn_predict_reshaped.npy', 'wb') as f:
                np.save(f, np.float32(npy_predict_cnn_reshaped))
        return npy_predict_cnn_reshaped


def load_numpy_arrays(dconfig, bucket='persist', fprefix='climsim'):
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
    


def ds_to_npy(ds, load=False):
    from dask.diagnostics import ProgressBar
    ds = ds.stack({'sample' : ['time', 'ncol']}) # stacks the time and ncol into one MultiIndex
    ds = ds.to_stacked_array('mlvar', sample_dims=['sample']) # turns the data array of multiple vars into one stacked var
    if(load):
        print(f"{ds.nbytes / 1e9} gigabytes") # GB
        # visualize with progress bar
        with ProgressBar():
            # use .load() or .compute() to do the math and get the daily mean data
            ds.load()
    return(ds)

    
def save_arrays(X, Y, bucket='scratch', fprefix='climsim'):
    if(bucket== 'scratch'):
        bucket = "leap-scratch"
    elif(bucket == 'persist'):
        bucket = "leap-persistent"
    with fsspec.open(f"gs://{bucket}/sammyagrawal/input_{fprefix}.npy", 'wb') as f:
        np.save(f, X)
    with fsspec.open(f"gs://{bucket}/sammyagrawal/output_{fprefix}.npy", 'wb') as f:
        np.save(f, Y)



def eliq(T):
    """
    Function taking temperature (in K) and outputting liquid saturation
    pressure (in hPa) using a polynomial fit
    """
    a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,
                              0.206739458e-7,0.302950461e-5,0.264847430e-3,
                              0.142986287e-1,0.443987641,6.11239921]);
    c_liq = -80
    T0 = 273.16
    return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))

def eice(T):
    """
    Function taking temperature (in K) and outputting ice saturation
    pressure (in hPa) using a polynomial fit
    """
    a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,
                      0.602588177e-7,0.615021634e-5,0.420895665e-3,
                      0.188439774e-1,0.503160820,6.11147274]);
    c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
    T0 = 273.16
    return (T>c_ice[0])*eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*\
                       (c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))
