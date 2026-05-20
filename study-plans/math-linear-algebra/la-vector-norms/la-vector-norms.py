import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.array(v)
    man_distance = np.sum(np.abs(v))
    euclidean_len = np.sqrt(np.sum((v)**2))
    max_abs_val = np.max(np.abs((v)))

    return np.array([man_distance, euclidean_len, max_abs_val])
    