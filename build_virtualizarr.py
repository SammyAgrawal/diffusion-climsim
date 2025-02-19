import sys
import os
sys.path.append(os.path.abspath(os.path.join('diffusionsim')))
#import diffusers
import diffusionsim as diff
import diffusionsim.training_utils as tru
from diffusionsim import mydatasets as data
from diffusionsim import climsim_utils as cut
from virtualizarr import open_virtual_dataset
import xarray as xr
import fsspec
import time 
import numpy as np
path = lambda fname : os.path.join(os.path.expanduser("~/diffusion-climsim/"), fname)



dconfig = tru.DataConfig()
dconfig.source = "local"
dconfig.climsim_type = "low-res-expanded"
dconfig.data_dir = "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/train"


kwargs = {
    'base_dir' : "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/train",
    'normalize' : False,
}
kwargs['grid_info'] = xr.open_dataset("/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/ClimSim_low-res_grid-info.nc")

dutils = cut.setup_data_utils(dconfig.climsim_type, dconfig.source, dconfig.data_vars, use_tendencies=dconfig.use_tendencies, **kwargs)

manifest_save_dir = "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/manifests2/"

virtual_datasets = []
times = []


interval = 20 * 3

i = 1

for directory in os.listdir(kwargs['base_dir']):
    start_time = time.time()
    year, month = map(int, directory.split("-"))
    start = cut.tocft(year, month, 1);
    if month == 12: # december
        stop = cut.tocft(year + 1, 1, 1)  # Next month is January of the next year
    else:
        stop = cut.tocft(year, month + 1, 1)  # Normal case for other months
    interval = 20 * 3 * 6 # every 6 hours
    dutils.set_filelist_using_intervals('train', start, stop, interval)
    print(f"Virtualize {len(dutils.get_filelist('train'))} files")
    monthly_vds = [] # for safety, also store monthly vds
    monthly_times = []

    for fname in dutils.get_filelist('train'):
        try:
            fpath = os.path.join(dutils.data_path, fname.replace(f'.{dutils.mlivar}.','.mlo.'))
            ds = open_virtual_dataset(fpath)
            curr_time = dutils.parse_time(fname)
            ds = ds.expand_dims(time=[i]); i += 1;
            monthly_vds.append(ds)
            monthly_times.append(curr_time)
        except Exception as e:
            print(f"Error opening {fname}: {e}")
            continue
    
    #virtual_datasets = virtual_datasets + monthly_vds
    #times = times + monthly_times

    try:
        virtual_ds = xr.combine_nested(monthly_vds, concat_dim=['time'])
        fpath = os.path.join(manifest_save_dir, f"{year}-{month}-out.parquet")
        virtual_ds.virtualize.to_kerchunk(fpath, format='parquet')
        print(f"Time taken for {year}-{month}: {time.time() - start_time} seconds")
        np.save(os.path.join(manifest_save_dir, f"{year}-{month}-times.npy"), monthly_times, allow_pickle=True)
    except Exception as e:
        print(f"Error virtualizing {year}-{month}: {e}")

"""

print("Tring to combine all virtual datasets")
start_time = time.time()
directories = sorted(os.listdir(kwargs['base_dir']), key=lambda x: tuple(map(int, x.split("-"))))

for directory in directories:
    start_time = time.time()
    year, month = map(int, directory.split("-"))
    vds = xr.open_dataset(os.path.join(manifest_save_dir, f"{year}-{month}-out.parquet"), engine='kerchunk', chunks={})


try:
    virtual_ds = xr.combine_nested(virtual_datasets, concat_dim=['time'])
    fpath = "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/manifest.parquet"
    virtual_ds.virtualize.to_kerchunk(fpath, format='parquet')
    print(f"Time taken for entire dataset is {time.time() - start_time} seconds")
    np.save("/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/times.npy", times, allow_pickle=True)
except Exception as e:
    print(f"Error virtualizing {year}-{month}: {e}")

"""