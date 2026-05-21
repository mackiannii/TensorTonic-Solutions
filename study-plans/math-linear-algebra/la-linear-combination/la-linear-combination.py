import numpy as np

def linear_combination(vectors, coefficients):
    """
    Returns: float64 array, the weighted sum of vectors.
    """
    lc = 0
    v = np.array(vectors)
    c = np.array(coefficients).reshape(-1, 1)
    weighted = v * c
    lc += np.sum(weighted, axis = 0)
    return lc