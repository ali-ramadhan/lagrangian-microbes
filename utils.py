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

def runtime2str(t):
    s = ""
    if t < 1e-6:
        s = "{:.3g} ns".format(t * 1e9)
    elif 1e-6 <= t < 1e-3:
        s = "{:.3g} μs".format(t * 1e6)
    elif 1e-3 < t < 1:
        s = "{:.3g} ms".format(t * 1e3)
    else:
        s = "{:.3g} s".format(t)
    return s
