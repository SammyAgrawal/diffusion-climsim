import sys
import os
sys.path.append('/diffusionsim')
import diffusionsim as diff
import diffusionsim.training_utils as tru
from diffusionsim import climsim_utils as cut
import xarray as xr
import fsspec
import icechunk
import time 
import numpy as np
from virtualizarr import open_virtual_dataset


import tempfile
from concurrent.futures import ProcessPoolExecutor
from icechunk.distributed import merge_sessions


path = lambda fname : os.path.join(os.path.expanduser("~/Climsim/diffusion-climsim/"), fname)

def iterate_months():
    for year in range(1,10):
        for month in range(1,13):
            if(year == 1 and month == 1):
                continue
            if(year == 9 and month > 1):
                break
            yield(year, month)


dconfig = tru.DataConfig()
dconfig.source = "huggingface"
dconfig.climsim_type = "low-res-expanded"

print(f"Building virtualizarr manifest for Climsim {dconfig.climsim_type} using {dconfig.source} data")

kwargs = {
    'base_dir' : "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/train",
    'normalize' : False,
}
#kwargs['grid_info'] = xr.open_dataset("/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/ClimSim_low-res_grid-info.nc")

dutils = cut.setup_data_utils(dconfig.climsim_type, dconfig.source, dconfig.data_vars, use_tendencies=dconfig.use_tendencies)

manifest_dir = "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/hf_manifests/"

input_storage = icechunk.local_filesystem_storage(manifest_dir + dutils.mlivar)
input_repo = icechunk.Repository.open(input_storage)

target_storage = icechunk.local_filesystem_storage(manifest_dir + "mlo")
target_repo = icechunk.Repository.open(target_storage)


desired_chunksizes = {'time': 1024, 'ncol': 384, 'lev': 60}


def write_timestamp(*, itime: int, session: Session) -> Session:
    # pass a list to isel to preserve the time dimension
    ds = xr.tutorial.open_dataset("rasm").isel(time=[itime])
    # region="auto" tells Xarray to infer which "region" of the output arrays to write to.
    ds.to_zarr(session.store, region="auto", consolidated=False)
    return session

from concurrent.futures import ThreadPoolExecutor, wait

session = repo.writable_session("main")
with ThreadPoolExecutor() as executor:
    # submit the writes
    futures = [executor.submit(write_timestamp, itime=i, session=session) for i in range(ds.sizes["time"])]
    wait(futures)

print(session.commit("finished writes"))



def fetch_file(fname, dutils=dutils):
    path = os.path.join(dutils.data_path, fname)
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = open_virtual_dataset(path)
    time = dutils.parse_time(fname)
    ds = dutils.add_time(ds, time)
    return ds


def fetch_virtual_datasets(year, month):
    monthly_input_vds = []
    monthly_target_vds = []
    dutils.set_filelist_using_hfhub("train", year, month, stride_sample=1)
    print(f"Virtualizing {len(dutils.get_filelist('train'))} files for {year}-{month}")
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
    print(f"Saving ds of size {vds.sizes}")
    if(appending):
        vds.virtualize.to_icechunk(session.store, append_dim='time')
    else:
        vds.virtualize.to_icechunk(session.store)
    msg = session.commit(commit_message)
    print(f"Committed {commit_message}, period added {msg}")


for year in range(6,10):
    for month in range(1,13):
        if(year == 6 and month < 5): # months 2 and 3 already added
            continue
        if(year == 9 and month > 1):
            break
        start_time = time.time()
        vds_inputs, vds_targets = fetch_virtual_datasets(year, month)
        add_period(vds_inputs, f"Appended {year}-{month} for inputs", input_repo)
        add_period(vds_targets, f"Appended {year}-{month} for targets", target_repo)

        print(f"Time taken for {year}-{month}: {(time.time() - start_time)/60} minutes")




import dask

list_of_fnames_per_monthyear = [...]

client = dask.distributed.Client(n_workers=20)

@dask.delayed
def fetch_file(fname):
    return open_virtual_dataset(fname)

delayed_list = [fetch_file(fname) for fname in list_of_fnames_per_monthyear] # every file in a month, like ~2000

loaded_filelist = client.compute(delayed_list)
vds_of_one_month = xr.combine_nested(loaded_filelist, concat_dim=['time']) # 2000 files, all virtual. 
