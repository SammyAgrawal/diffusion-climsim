import sys
import os
sys.path.append(os.path.abspath(os.path.join("Climsim", "diffusion-climsim", "diffusionsim")))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import diffusionsim as diff
import diffusionsim.training_utils as tru
from diffusionsim import climsim_utils as cut
import xarray as xr
import fsspec
import icechunk
import time 
import numpy as np
from virtualizarr import open_virtual_dataset
from icechunk.xarray import to_icechunk



path = lambda fname : os.path.join(os.path.expanduser("~/Climsim/diffusion-climsim/"), fname)


dconfig = tru.DataConfig()
dconfig.source = "local"
dconfig.climsim_type = "low-res-expanded"

base_dir = "/mnt/lustre/columbia/ssa2206/ClimSim_low-res-expanded/train"

n = cut.expand_ds_name("low-res")
grid_url = f"https://huggingface.co/datasets/LEAP/{n}/resolve/main/{n}_grid-info.nc"
grid_info = cut.read_url(grid_url, copy_to_local=True)

dutils = cut.setup_data_utils(dconfig.climsim_type, dconfig.source, 
                              dconfig.data_vars, use_tendencies=dconfig.use_tendencies, 
                              grid_info=grid_info, base_dir=base_dir
                            )
manifest_dir = os.path.join("/mnt/home/ssa2206/Climsim/diffusion-climsim/data", "vlocal_manifests")
print(f"saving manifests to {manifest_dir}")
input_storage = icechunk.local_filesystem_storage(os.path.join(manifest_dir, dutils.mlivar))
input_repo = icechunk.Repository.open_or_create(input_storage)

target_storage = icechunk.local_filesystem_storage(os.path.join(manifest_dir, "mlo"))
target_repo = icechunk.Repository.open_or_create(target_storage)


desired_chunksizes = {'time': 1024, 'ncol': 384, 'lev': 60}


def fetch_file(fname, dutils=dutils):
    path = os.path.join(dutils.data_path, fname)
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(path, engine='netcdf4')
    time = dutils.parse_time(fname)
    ds = dutils.add_time(ds, time)
    return ds


def fetch_virtual_datasets(year, month):
    monthly_input_vds = []
    monthly_target_vds = []
    dutils.set_filelist_using_hfhub("train", year, month, stride_sample=1)
    print(f"Virtualizing {len(dutils.get_filelist('train'))} files for {year}-{month}", flush=True)
    for fname in dutils.get_filelist('train'):
        try:
            monthly_input_vds.append(fetch_file(fname))
            monthly_target_vds.append(fetch_file(fname.replace(f'.{dutils.mlivar}.','.mlo.')))
        except Exception as e:
            print(f"Error virtualizing {fname}: {e}")
    vds_inputs = xr.combine_nested(monthly_input_vds, concat_dim=['time'])
    vds_targets = xr.combine_nested(monthly_target_vds, concat_dim=['time'])
    return vds_inputs, vds_targets


def add_period(vds, commit_message, repo, appending=True):
    session = repo.writable_session("main")
    print(f"Saving ds of size {vds.sizes}", flush=True)
    if(appending):
        to_icechunk(vds, session, append_dim='time')
    else:
        to_icechunk(vds, session)
    msg = session.commit(commit_message)
    print(f"Committed {commit_message}, period added {msg}", flush=True)


for year in range(4,10):
    for month in range(1,13):
        if(year == 4 and month < 6):
            continue
        if(year == 9 and month > 1):
            break
        #appending = True#False if year == 1 and month == 2 else True
        start_time = time.time()
        vds_inputs, vds_targets = fetch_virtual_datasets(year, month)
        print(f"time taken to fetch {year}-{month}: {(time.time() - start_time)/60} minutes")
        add_period(vds_inputs, f"Appended {year}-{month} for inputs", input_repo)
        print(f"time taken to store inputs in icechunk: {(time.time() - start_time)/60} minutes")
        add_period(vds_targets, f"Appended {year}-{month} for targets", target_repo)
        print(f"Time taken to store outputs in icehunk: {(time.time() - start_time)/60} minutes")
