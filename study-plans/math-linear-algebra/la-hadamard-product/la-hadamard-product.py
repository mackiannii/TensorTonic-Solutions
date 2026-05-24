import numpy as np

def hadamard_product(A, B):
    """
    Returns: ndarray, the element-wise product A * B.
    """
    A = np.array(A)
    B = np.array(B)

    if A.shape != B.shape: 
        raise ValueError("A shape is not equal to B shape size")

    rows, cols = A.shape
    new_matrix = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            new_matrix[i][j] = A[i][j] * B[i][j]

    return new_matrix