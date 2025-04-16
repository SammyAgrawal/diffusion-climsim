import random
import torch
import numpy as np
import xarray as xr
import pandas as pd



def read_xarray(dir_name, data_type, data_num):
    '''
     read_xarray(dir)name) opens data and returns data in xarray format for each feature
    '''
    if data_type == "CESM" or data_type == "MPI" or data_type == "GFDL":
        date = "198201-201701"
    elif data_type == "CanESM2":
        date = "198201-201712"
    dir_name2 = dir_name + "/" + data_type + "/member_" + data_num


    chl = xr.open_dataset(f'{dir_name2}/Chl_2D_mon_{data_type}{data_num}_1x1_{date}.nc')

    mld = xr.open_dataset(f'{dir_name2}/MLD_2D_mon_{data_type}{data_num}_1x1_{date}.nc')

    sss = xr.open_dataset(f'{dir_name2}/SSS_2D_mon_{data_type}{data_num}_1x1_{date}.nc')

    sst = xr.open_dataset(f'{dir_name2}/SST_2D_mon_{data_type}{data_num}_1x1_{date}.nc')

    u10 = xr.open_dataset(f'{dir_name2}/U10_2D_mon_{data_type}{data_num}_1x1_{date}.nc')

    xco2 = xr.open_dataset(f'{dir_name}/CESM/member_001/XCO2_1D_mon_CESM001_native_198201-201701.nc')

    icefrac = xr.open_dataset(f'{dir_name2}/iceFrac_2D_mon_{data_type}{data_num}_1x1_{date}.nc')

    patm = xr.open_dataset(f'{dir_name2}/pATM_2D_mon_{data_type}{data_num}_1x1_{date}.nc')

    pco2 = xr.open_dataset(f'{dir_name2}/pCO2_2D_mon_{data_type}{data_num}_1x1_{date}.nc')

    return chl, mld, sss, sst, u10, xco2, icefrac, patm, pco2