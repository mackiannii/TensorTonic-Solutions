import numpy as np

def gram_schmidt(vectors):
    """
    Returns: float64 array of shape (k, n), orthonormal basis spanning the input space.
    """
    v = np.array(vectors, dtype=np.float64)
    q = []

    for i in range(len(v)):
        subtotal = np.zeros_like(v[i])
        
        for j in range(len(q)):
            dot = v[i] @ q[j]
            subtotal += dot * q[j]
        
        u = v[i] - subtotal
        norm = np.linalg.norm(u)

        if norm != 0: 
            q_i = u / norm
            q.append(q_i)

    return np.array(q, dtype=np.float64)