import numpy as np

__all__ = ["cellinfo"]

points = np.array([[0, 0], [0, 1/np.sqrt(3)]])
tvs = np.array([[1, 0], [1.0/2, np.sqrt(3)/2]])
cellinfo = {"points": points, "tvs": tvs}
