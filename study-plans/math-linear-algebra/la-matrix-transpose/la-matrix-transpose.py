import numpy as np

def matrix_transpose(A):
    """
    Returns: ndarray, the transpose of A.
    """
    A = np.array(A)
    rows = len(A)
    cols = len(A[0])

    new_matrix = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            new_matrix[j][i] = A[i][j]

    return new_matrix