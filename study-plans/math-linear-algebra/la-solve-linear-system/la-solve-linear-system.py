def solve_linear_system(A, b):
    """
    Returns a float64 array x satisfying A @ x = b.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape

    if m == n:
        if np.linalg.matrix_rank(A) == n:
            return np.linalg.solve(A, b)

        raise ValueError("Square system does not have a unique solution.")
        