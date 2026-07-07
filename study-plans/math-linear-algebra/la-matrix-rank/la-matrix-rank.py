import numpy as np

def matrix_rank(A):
    # Make a copy of A as floats so we can safely divide during elimination
    working_matrix = [list(map(float, row)) for row in A]

    num_rows = len(working_matrix)
    num_cols = len(working_matrix[0])

    # This counts how many pivots we have found.
    # It also tells us which row the next pivot should be moved into.
    next_pivot_row = 0

    # Try to find one pivot in each column, moving left to right
    for pivot_col in range(num_cols):

        # We have not found a usable pivot row in this column yet
        found_pivot_row = None

        # Search downward from next_pivot_row.
        # Rows above next_pivot_row already contain pivots.
        for candidate_row in range(next_pivot_row, num_rows):
            if working_matrix[candidate_row][pivot_col] != 0:
                found_pivot_row = candidate_row
                break

        # If every entry in this column below next_pivot_row is zero,
        # then this column cannot give us a new pivot.
        if found_pivot_row is None:
            continue

        # Move the found pivot row into the official pivot position
        working_matrix[next_pivot_row], working_matrix[found_pivot_row] = (
            working_matrix[found_pivot_row],
            working_matrix[next_pivot_row],
        )

        # Get the pivot value after the swap
        pivot_value = working_matrix[next_pivot_row][pivot_col]

        # Use the pivot row to eliminate all entries below the pivot
        for row_to_eliminate in range(next_pivot_row + 1, num_rows):

            # How much of the pivot row do we subtract from this row?
            elimination_factor = (
                working_matrix[row_to_eliminate][pivot_col] / pivot_value
            )

            # Update this row from pivot_col onward
            for col_to_update in range(pivot_col, num_cols):
                working_matrix[row_to_eliminate][col_to_update] -= (
                    elimination_factor
                    * working_matrix[next_pivot_row][col_to_update]
                )

        # We successfully found one more pivot
        next_pivot_row += 1

    # Number of pivots found equals the rank
    return next_pivot_row