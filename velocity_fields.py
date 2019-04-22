def oscar_dataset_opendap_url(year):
    return r"https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/oscar/preview/L4/oscar_third_deg/oscar_vel" \
           + str(year) + ".nc.gz"
