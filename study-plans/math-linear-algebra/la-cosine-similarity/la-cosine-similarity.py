import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape: 
        raise ValueError("A and B not the same length")
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: 
        return 0
    cos_sim = np.dot(a, b) / denom
    return cos_sim