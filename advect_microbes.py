from datetime import datetime

import numpy as np
import xarray as xr
import parcels

velocity_dataset_filepath = '/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel2017_180.nc'

# Load velocity dataset
velocity_dataset = xr.open_dataset(velocity_dataset_filepath)
print(velocity_dataset)

# Choose subset of velocity field we want to use
velocity_subdataset = velocity_dataset.sel(time=np.datetime64('2017-01-01'), year=2017.0, depth=15.0, latitude=slice(60, 0), longitude=slice(-180, -120))
print(velocity_subdataset)

print(velocity_subdataset['u'])