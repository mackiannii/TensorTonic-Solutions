import numpy as np

def matrix_rank(A):
    # Convert every entry to float so row operations use decimal division
    A = [list(map(float, row)) for row in A]

    rows = len(A)
    cols = len(A[0])

    # rank also tells us where the next pivot row should be placed
    rank = 0

    # Move left to right through each column looking for pivots
    for c in range(cols):

        # We have not found a pivot row for this column yet
        pivot_row = None

        # Search downward from the current rank row.
        # Rows above rank already have pivots, so we skip them.
        for r in range(rank, rows):
            if A[r][c] != 0:
                pivot_row = r
                break

        # If this entire column has no usable pivot, move to the next column
        if pivot_row is None:
            continue

        # Move the pivot row into the next available pivot position
        A[rank], A[pivot_row] = A[pivot_row], A[rank]

        # The pivot value is the entry at the pivot row and current column
        pivot = A[rank][c]

        # Use the pivot row to eliminate entries below the pivot
        for r in range(rank + 1, rows):
            factor = A[r][c] / pivot

            # Start at column c because entries before c do not matter for this pivot step
            for j in range(c, cols):
                A[r][j] -= factor * A[rank][j]

        # We successfully found one more pivot, so rank increases by 1
        rank += 1

    return rank