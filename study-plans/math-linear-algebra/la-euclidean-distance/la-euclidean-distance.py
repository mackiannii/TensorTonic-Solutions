import math

def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = np.array(x)
    y = np.array(y)
    total = 0

    if len(x) != len(y):
        raise ValueError("X and Y are not of equal length")
    total += np.sqrt(np.sum((x - y)**2))
    return total