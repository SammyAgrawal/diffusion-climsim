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
import icechunk
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

manifest_save_dir = "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/manifests"
print("Tring to combine all virtual datasets")
start_time = time.time()
directories = sorted(os.listdir(kwargs['base_dir']), key=lambda x: tuple(map(int, x.split("-"))))

virtual_datasets = []
times = []

for directory in directories:
    start_time = time.time()
    year, month = map(int, directory.split("-"))
    vds = xr.open_dataset(os.path.join(manifest_save_dir, f"{year}-{month}-out.parquet"), engine='kerchunk', chunks={})
    vds['time'] = np.load(os.path.join(manifest_save_dir, f"{year}-{month}-times.npy"), allow_pickle=True)
    print(f"loaded {year}-{month} with {len(vds.time)} samples")
    virtual_datasets.append(vds)
    
virtual_ds = xr.combine_nested(virtual_datasets, concat_dim=['time'])
print(f"writing to disk, {len(virtual_ds.time)} samples from list of {len(virtual_datasets)} virtual datasets")
fpath = "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/aggregate_manifest"

storage = icechunk.local_filesystem_storage(fpath)
repo = icechunk.Repository.create(storage)

session = repo.writable_session("main")
virtual_ds.virtualize.to_icechunk(session.store)
session.commit("Committed aggregate manifest!")
print(f"Time taken for entire dataset is {time.time() - start_time} seconds")