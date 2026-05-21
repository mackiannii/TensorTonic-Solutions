import numpy as np

def matrix_trace(A):
    """
    Returns: float, the trace (sum of diagonal elements) of A.
    """
    total = 0
    A = np.array(A)
    for i in range(len(A)):
        total += A[i][i]
    return total