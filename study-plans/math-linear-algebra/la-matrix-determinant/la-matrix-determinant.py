import numpy as np

def matrix_determinant(A):
    """
    Returns: float, the determinant of square matrix A.
    """
    A = np.array(A, dtype = float)
    rows, cols = A.shape
    total = 0
    if rows != cols:
         raise ValueError("This is a not sqaure matrix")
    # base case
    if A.shape == (1,1): 
        return A[0, 0]
    # base case
    if A.shape == (2, 2): 
        return A[0, 0] * A[1 ,1] - A[0 ,1] * A[1, 0]
    for j in range(cols):
        sign = (-1)**j
        coeff = A[0, j]
        without_row = np.delete(A, 0, axis=0)
        minor = np.delete(without_row, j, axis=1)
        total += sign * coeff * matrix_determinant(minor)
    return total

        
        
        
    
    

# 1 2
# 3 4 
