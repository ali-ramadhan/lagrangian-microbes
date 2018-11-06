from datetime import datetime, timedelta

import numpy as np

def closest_hour(ndt):
    sdt = str(ndt.astype('datetime64[s]'))  # string datetime
    pdt = datetime.strptime(sdt, "%Y-%m-%dT%H:%M:%S")  # python datetime
    
    if pdt.minute >= 30:
        pdt = pdt.replace(minute=0, second=0) + timedelta(hours=1)
    else:
        pdt = pdt.replace(minute=0, second=0)

    return pdt