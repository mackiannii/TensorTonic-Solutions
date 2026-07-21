import numpy as np

def lu_decomposition(A):
    """
    Returns: tuple (L, U) where A = L @ U.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    L = np.eye(n)
    U = np.copy(A)

    for pivot in range(n):
        for row in range(pivot + 1, n): 
            factor = U[row, pivot] / U[pivot, pivot]

            L[row, pivot] = factor 
            U[row, pivot:] -= factor * U[pivot, pivot:]

    return L, U