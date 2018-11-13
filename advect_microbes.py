import os
import time
import pickle
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import parcels
import joblib

from constants import lon_min, lon_max, lat_min, lat_max
from constants import N, Tx, Ty, NTx, NTy
from constants import t, dt, tpd, n_periods
from constants import output_dir
from utils import closest_hour

def advect_microbes(jid, mlons, mlats):
    year = 2017
    velocity_dataset_filepath = "/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel" + str(year) + "_180.nc"

    # Load velocity dataset
    velocity_dataset = xr.open_dataset(velocity_dataset_filepath)
    n_fields = len(velocity_dataset["time"])

    # Choose subset of velocity field for initial conditions.
    velocity_subdataset = velocity_dataset.sel(time=np.datetime64('2017-01-01'), year=2017.0, depth=15.0,
        latitude=slice(60, 0), longitude=slice(-180, -120))

    glats0 = velocity_subdataset['latitude'].values
    glons0 = velocity_subdataset['longitude'].values
    depth = np.array([15.0])

    u0_data = velocity_subdataset['u'].values
    v0_data = velocity_subdataset['v'].values

    u0_field = parcels.field.Field(name='U', data=u0_data,
        lon=glons0, lat=glats0, depth=depth, mesh='spherical')
    v0_field = parcels.field.Field(name='V', data=v0_data,
        lon=glons0, lat=glats0, depth=depth, mesh='spherical')

    fieldset0 = parcels.fieldset.FieldSet(u0_field, v0_field)

    pset = parcels.ParticleSet.from_list(fieldset=fieldset0, pclass=parcels.JITParticle, lon=mlons, lat=mlats)

    for period in range(n_periods):
        t_start = velocity_dataset["time"][period].values

        # If we're on the last field, then t_end will be midnight of next year.
        if period == n_fields:
            t_end = datetime(t_start.year + 1, 1, 1)
        else:
            t_end = velocity_dataset["time"][period+1].values

        t_start_ch = closest_hour(t_start)
        t_end_ch = closest_hour(t_end)

        advection_hours = round((t_end_ch - t_start_ch) / timedelta(hours=1))
        # print("{:} -> {:} ({:d} hours)".format(t_start, t_end, advection_hours))

        # Choose subset of velocity field we want to use
        year_float = velocity_dataset["year"].values[period]
        depth_float = velocity_dataset["depth"].values[0]
        velocity_subdataset = velocity_dataset.sel(time=t_start, year=year_float, depth=depth_float, latitude=slice(60, 0), longitude=slice(-180, -120))

        glats = velocity_subdataset['latitude'].values
        glons = velocity_subdataset['longitude'].values
        depth = np.array([depth_float])

        u_data = velocity_subdataset['u'].values
        v_data = velocity_subdataset['v'].values

        u_field = parcels.field.Field(name='U', data=u_data,
            lon=glons, lat=glats, depth=depth, mesh='spherical')
        v_field = parcels.field.Field(name='V', data=v_data,
            lon=glons, lat=glats, depth=depth, mesh='spherical')

        pset.fieldset = parcels.fieldset.FieldSet(u_field, v_field)
        pset.fieldset.check_complete()

        dump_filename = "rps_microbe_locations_p" + str(period).zfill(4) + "_block" + str(jid).zfill(2) + ".joblib.pickle"
        dump_filepath = os.path.join(output_dir, dump_filename)
        print("[{:02d}] Advecting: {:} -> {:} ({:s})... ".format(jid, t_start_ch, t_end_ch, dump_filepath), end="")

        latlon_store = {
            "hours": advection_hours,
            "lat": np.zeros((advection_hours, NTx*NTy)),
            "lon": np.zeros((advection_hours, NTx*NTy))
        }

        tic = time.time()
        for h in range(advection_hours):
            pset.execute(parcels.AdvectionRK4,
                runtime=dt, dt=dt, verbose_progress=False, output_file=None)

            for i, p in enumerate(pset):
                latlon_store["lon"][h, i] = p.lon
                latlon_store["lat"][h, i] = p.lat

        # joblib.dump(latlon_store, dump_filename, compress=False, protocol=pickle.HIGHEST_PROTOCOL)
        with open(dump_filepath, "wb") as f:
            joblib.dump(latlon_store, f, compress=False, protocol=pickle.HIGHEST_PROTOCOL)

        toc = time.time()
        print("({:g} s)".format(toc - tic))

        # for i, p in enumerate(pset):
        #     if p.lat >= 59 or p.lat <= 1 or p.lon <= -179 or p.lon >= -121:
        #         print("Removing particle #{:d} @({:.2f},{:.2f}). Too close to boundary"
        #             .format(i, p.lat, p.lon))
        #         pset.remove(i)

if __name__ == "__main__":
    # psset = initialize_microbes()  # Particle superset (a list of ParticleSets)

    print("Found {:d} CPUs.".format(joblib.cpu_count()))

    mlon_blocks = Tx*Ty * [None]
    mlat_blocks = Tx*Ty * [None]

    delta_lon = (lon_max - lon_min) / Tx
    delta_lat = (lat_max - lat_min) / Ty
    for i in range(Tx):
        for j in range(Ty):
            mlon_min = lon_min + i*delta_lon
            mlon_max = lon_min + (i+1)*delta_lon
            mlat_min = lat_min + j*delta_lat
            mlat_max = lat_min + (j+1)*delta_lat

            # print("(Tx={:d}, Ty={:d}) {:.2f}-{:.2f} E, {:.2f}-{:.2f} N".format(i, j, mlon_min, mlon_max, mlat_min, mlat_max))
            # print("(i,j,i*Ty+j) = ({:d},{:d},{:d})".format(i, j, i*Ty+j))

            # Microbe longitudes and latitudes
            mlon_blocks[i*Ty + j] = np.repeat(np.linspace(mlon_min, mlon_max, NTx), NTy)
            mlat_blocks[i*Ty + j] = np.tile(np.linspace(mlat_min, mlat_max, NTy), NTx)

    joblib.Parallel(n_jobs=-1)(joblib.delayed(advect_microbes)(jid, mlon_blocks[jid], mlat_blocks[jid]) for jid in range(Tx*Ty))
