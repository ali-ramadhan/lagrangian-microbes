import os

import xarray as xr

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)


def oscar_dataset_opendap_url(year):
    return r"https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/oscar/preview/L4/oscar_third_deg/oscar_vel" \
           + str(year) + ".nc.gz"


def oscar_dataset_filename(year):
    return "oscar_vel" + str(year) + ".nc"


def oscar_dataset(year):
    dataset_filepath = oscar_dataset_filename(year)
    if not os.path.isfile(dataset_filepath):
        oscar_url = oscar_dataset_opendap_url(year)

        logger.info("Accessing OSCAR dataset over OPeNDAP: {:s}".format(oscar_url))
        dataset = xr.open_dataset(oscar_url)

        logger.info("Saving OSCAR dataset to disk: {:}".format(os.path.abspath(dataset_filepath)))
        dataset.to_netcdf(dataset_filepath)

    return xr.open_dataset(dataset_filepath)
