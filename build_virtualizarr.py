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

kwargs = {
    'base_dir' : "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/train",
    'normalize' : False,
}
#kwargs['grid_info'] = xr.open_dataset("/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/ClimSim_low-res_grid-info.nc")

dutils = cut.setup_data_utils(dconfig.climsim_type, dconfig.source, dconfig.data_vars, use_tendencies=dconfig.use_tendencies)

manifest_dir = "/mnt/lustre/columbia/ssa2206/data/ClimSim_low-res-expanded/hf_manifests/"

input_storage = icechunk.local_filesystem_storage(manifest_dir + dutils.mlivar)
input_repo = icechunk.Repository.create(input_storage)

target_storage = icechunk.local_filesystem_storage(manifest_dir + "mlo")
target_repo = icechunk.Repository.create(target_storage)



for year in range(1,10):
    for month in range(1,13):
        if(year == 1 and month == 1):
            continue
        if(year == 9 and month > 1):
            break
        start_time = time.time()

        monthly_input_vds = [] # for safety, also store monthly vds
        monthly_target_vds = []

        dutils.set_filelist_using_hfhub("train", year, month, stride_sample=1)
        print(f"Virtualizing {len(dutils.get_filelist('train'))} files for {year}-{month}")

        for fname in dutils.get_filelist('train'):
            try:
                monthly_input_vds.append(dutils.get_xrdata(fname, virtual=True))
                monthly_target_vds.append(dutils.get_xrdata(fname.replace(f'.{dutils.mlivar}.','.mlo.'), virtual=True))
            except Exception as e:
                print(f"Error opening {fname}: {e}")
                continue
        
        try:
            session = input_repo.writable_session("main")
            virtual_inputs = xr.combine_nested(monthly_input_vds, concat_dim=['time'])
            if(year == 1 and month == 1):
                 virtual_inputs.virtualize.to_icechunk(session.store)
            else:
                 virtual_inputs.virtualize.to_icechunk(session.store, append_dim='time')
            session.commit(f"Appended {year}-{month} for inputs")
            
            session = target_repo.writable_session("main")
            virtual_targets = xr.combine_nested(monthly_target_vds, concat_dim=['time'])
            if(year == 1 and month == 1):
                virtual_targets.virtualize.to_icechunk(session.store)
            else:
                virtual_targets.virtualize.to_icechunk(session.store, append_dim='time')
            session.commit(f"Appended {year}-{month} for targets")
        except Exception as e:
            print(f"Error virtualizing {year}-{month}: {e}")
            continue
        print(f"Time taken for {year}-{month}: {(time.time() - start_time)/60} minutes")
    


