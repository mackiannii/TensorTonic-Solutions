import numpy as np

def matrix_multiply(A, B):
    """
    Returns: 2-D float64 array, the matrix product A @ B.
    """
    A = np.array(A)
    B = np.array(B)
    rows, cols = A.shape
    rows2, cols2 = B.shape

    new_matrix = np.zeros((rows, cols2))
    
    for i in range(rows):
        for j in range(cols2):
            for k in range(cols):
                new_matrix[i][j] += A[i][k] * B[k][j]
    return new_matrix
            

            