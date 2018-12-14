import os
import time
import pickle
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import parcels
import joblib

from constants import ADVECTION_OUTPUT_DIR
from constants import DOMAIN_LONS, DOMAIN_LATS
from constants import lon_min, lon_max, lat_min, lat_max
from constants import N, Tx, Ty, NTx, NTy
from constants import delta_mlon, delta_mlat
from constants import t, dt, tpd, n_periods
from utils import closest_hour

def advect_microbes(jid, mlons, mlats):
    year = 2017
    velocity_dataset_filepath = "/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel" + str(year) + "_180.nc"

    # Load velocity dataset
    velocity_dataset = xr.open_dataset(velocity_dataset_filepath)
    n_fields = len(velocity_dataset["time"])

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
        velocity_subdataset = velocity_dataset.sel(time=t_start, year=year_float, depth=depth_float, latitude=DOMAIN_LATS, longitude=DOMAIN_LONS)

        glats = velocity_subdataset['latitude'].values
        glons = velocity_subdataset['longitude'].values
        depth = np.array([depth_float])

        u_data = velocity_subdataset['u'].values
        v_data = velocity_subdataset['v'].values

        u_field = parcels.field.Field(name='U', data=u_data,
            lon=glons, lat=glats, depth=depth, mesh='spherical')
        v_field = parcels.field.Field(name='V', data=v_data,
            lon=glons, lat=glats, depth=depth, mesh='spherical')

        fieldset = parcels.fieldset.FieldSet(u_field, v_field)

        pset = parcels.ParticleSet.from_list(fieldset=fieldset, pclass=parcels.JITParticle, lon=mlons, lat=mlats)

        dump_filename = "rps_microbe_locations_p" + str(period).zfill(4) + "_block" + str(jid).zfill(2) + ".joblib.pickle"
        dump_filepath = os.path.join(ADVECTION_OUTPUT_DIR, dump_filename)
        
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

        with open(dump_filepath, "wb") as f:
            joblib.dump(latlon_store, f, compress=False, protocol=pickle.HIGHEST_PROTOCOL)

        # Create new mlon and mlat lists to create new particle set.
        n_particles = len(pset)
        mlons = np.zeros(n_particles)
        mlats = np.zeros(n_particles)
        for i, p in enumerate(pset):
            mlons[i] = p.lon
            mlats[i] = p.lat

        toc = time.time()
        print("({:g} s)".format(toc - tic))

        # for i, p in enumerate(pset):
        #     if p.lat >= 59 or p.lat <= 1 or p.lon <= -179 or p.lon >= -121:
        #         print("Removing particle #{:d} @({:.2f},{:.2f}). Too close to boundary"
        #             .format(i, p.lat, p.lon))
        #         pset.remove(i)

if __name__ == "__main__":
    print("Number of microbes: {:d}".format(N))
    print("Δlon={:3g}, Δlat={:3g}".format(delta_mlon, delta_mlat))
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
            mlon_blocks[i*Ty + j] = np.repeat(np.linspace(mlon_min, mlon_max - delta_mlon, NTx), NTy)
            mlat_blocks[i*Ty + j] = np.tile(np.linspace(mlat_min, mlat_max - delta_mlat, NTy), NTx)

    joblib.Parallel(n_jobs=-1)(joblib.delayed(advect_microbes)(jid, mlon_blocks[jid], mlat_blocks[jid]) for jid in range(Tx*Ty))
