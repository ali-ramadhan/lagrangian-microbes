from datetime import datetime, timedelta

import numpy as np
from numpy import sqrt, abs, argmin


def factor(n):
    """
    Calculates the integer factors of n.

    Args:
        n: integer to be factored.

    Returns: A sorted list of the factors of n.

    """
    factors = set()
    for x in range(1, int(sqrt(n)) + 1):
        if n % x == 0:
            factors.add(x)
            factors.add(n // x)
    return sorted(factors)


def most_symmetric_integer_factorization(N):
    """
    Calculates the "most symmetric" integer factorization of N. This is the closest two integer factors of N.

    Args:
        N: Integer

    Returns: Tuple containing the two "most symmetric" integer factorization of N.

    """
    factors = np.array(factor(N))

    # Get in the index of the largest factor that is still smaller than sqrt(N).
    i = argmin(abs(factors - sqrt(N)))

    return factors[i], N // factors[i]


def closest_hour(ndt):
    sdt = str(ndt.astype('datetime64[s]'))  # string datetime
    pdt = datetime.strptime(sdt, "%Y-%m-%dT%H:%M:%S")  # python datetime
    
    if pdt.minute >= 30:
        pdt = pdt.replace(minute=0, second=0) + timedelta(hours=1)
    else:
        pdt = pdt.replace(minute=0, second=0)

    return pdt


def runtime2str(t):
    if t < 1e-6:
        return "{:.3g} ns".format(t * 1e9)
    elif 1e-6 <= t < 1e-3:
        return "{:.3g} μs".format(t * 1e6)
    elif 1e-3 < t < 1:
        return "{:.3g} ms".format(t * 1e3)
    else:
        return "{:.3g} s".format(t)
