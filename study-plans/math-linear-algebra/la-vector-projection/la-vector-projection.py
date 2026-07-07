import numpy as np

def vector_projection(u, v):
    """
    Returns: float64 array, the projection of u onto v.
    """
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    numerator = u @ v      # u dot v
    denominator = v @ v    # v dot v

    projection = (numerator / denominator) * v

    return projection