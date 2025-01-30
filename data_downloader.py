import sys
import os
#sys.path.append(os.path.abspath(os.path.join('diffusionsim')))
#print(os.path.abspath(os.path.join('diffusionsim')))
import diffusionsim as diff
from diffusionsim import training_utils as tru
from diffusionsim import mydatasets as data
from diffusionsim import climsim_utils as cut
from pathlib import Path

BASE_PATH = "/mnt/lustre/columbia/ssa2206/data/"
print(f"Downloading data to {BASE_PATH}")
dconfig = tru.DataConfig()
dconfig.source = "huggingface"
dconfig.climsim_type = "expanded-low-res"
dutils = cut.setup_data_utils(dconfig.climsim_type, dconfig.source, None, True)

def generate_file_structure():
    file_structure = []
    years = range(1, 10)  # 第 1 年到第 9 年
    months = range(1, 13)  # 每年的 1 到 12 月
    for year in years:
        if year < 4:
            continue # skip the first 3 years
        for month in months:
            folder_name = f"{year:04d}-{month:02d}"
            Path(os.path.join(BASE_PATH, folder_name)).mkdir(parents=True, exist_ok=True)
            file_structure.append((month, year))
            #for day in range(1, 32): 
            #    for second in range(0, 86400, 1200): 
            #        file_name = f"E3SM-MMF.mli.{year:04d}-{month:02d}-{day:02d}-{second:05d}.nc"
           #         file_structure.append(f"{folder_name}/{file_name}")
    return file_structure
file_structure = generate_file_structure()
for month, year in file_structure:
    dutils.set_filelist_using_hfhub("train", year, month, stride_sample=1)
    #filelist.extend(dutils.train_filelist)
    sample_fname = dutils.train_filelist[0] # just first for testing
    print(f"Downloading {sample_fname}")
    ds = dutils.get_input(sample_fname)
    ds.to_netcdf(os.path.join(BASE_PATH, sample_fname))

