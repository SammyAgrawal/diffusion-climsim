import sys
import os
sys.path.append(os.path.abspath(os.path.join('diffusionsim')))
print(os.path.abspath(os.path.join('diffusionsim')))
import diffusionsim as diff
from diffusionsim import training_utils as tru
from diffusionsim import mydatasets as data
from diffusionsim import climsim_utils as cut

dconfig = tru.DataConfig()
dconfig.source = "huggingface"
dconfig.climsim_type = "expanded-low-res"
dutils = cut.setup_data_utils(dconfig.climsim_type, dconfig.source, None, True)
dutils.set_filelist_using_hfhub("train", year=1, month=3, stride_sample=1)
sample_fname = dutils.train_filelist[0]
ds = dutils.get_input(sample_fname)
print(sample_fname, ds)
#ds.to_netcdf(f"{sample_fname}")

