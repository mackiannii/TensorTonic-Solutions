import numpy as np

def matrix_vector_multiply(A, x):
    """
    Returns: 1-D float64 array, the product A @ x.
    """
    A = np.array(A)
    x = np.array(x)
    rows, cols = A.shape 
    result = np.zeros(rows)

    for i in range(rows):
        total = 0
        for j in range(cols):
            total += A[i][j] * x[j]
        result[i] = total
    return result