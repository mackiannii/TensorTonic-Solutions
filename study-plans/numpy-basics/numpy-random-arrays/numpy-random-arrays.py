import numpy as np

def generate_random_array(shape, kind, seed):
    np.random.seed(seed)
    if kind == "uniform":
        return np.random.random(shape)
    if kind == "normal":
        return np.random.standard_normal(shape)
    
