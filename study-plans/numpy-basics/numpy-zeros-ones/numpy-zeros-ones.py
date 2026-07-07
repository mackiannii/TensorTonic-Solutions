import numpy as np

def create_filled_array(shape, kind):
    if kind == "zeros":
        A = np.zeros(shape, dtype=np.float64)
        return A
    elif kind == "ones":
        B = np.ones(shape, dtype=np.float64)
        return B