import numpy as np

__all__ = ["cellinfo", "clusterinfo22", "clusterinfo23", "clusterinfo32", 
           "clusterinfo24", "clusterinfo42", "clusterinfo7"]

cell_points = np.array([[0, 0], [0.25, np.sqrt(3)/4], [0.5, 0]], 
                       dtype=np.float64)
cell_tvs = np.array([[1, 0], [0.5, np.sqrt(3)/2]], dtype=np.float64)
cellinfo = {"points": cell_points, "tvs": cell_tvs}

translations = [(0, 0), (0, 1), (1, 0), (1, 1)]
points = [cell_points + np.dot([x, y], cell_tvs) for x, y in translations]
tvs = cell_tvs * 2
clusterinfo22 = {"points": np.concatenate(points), "tvs": tvs}

translations = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
points = [cell_points + np.dot([x, y], cell_tvs) for x, y in translations]
tvs = cell_tvs * np.array([[3, 3], [2, 2]])
clusterinfo23 = {"points": np.concatenate(points), "tvs": tvs}

translations = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
points = [cell_points + np.dot([x, y], cell_tvs) for x, y in translations]
tvs = cell_tvs * np.array([[2, 2], [3, 3]])
clusterinfo32 = {"points": np.concatenate(points), "tvs": tvs}

translations = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
points = [cell_points + np.dot([x, y], cell_tvs) for x, y in translations]
tvs = cell_tvs * np.array([[4, 4], [2, 2]])
clusterinfo24 = {"points": np.concatenate(points), "tvs": tvs}

translations = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
points = [cell_points + np.dot([x, y], cell_tvs) for x, y in translations]
tvs = cell_tvs * np.array([[2, 2], [4, 4]])
clusterinfo42 = {"points": np.concatenate(points), "tvs": tvs}

translations = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
points = [cell_points + np.dot([x, y], cell_tvs) for x, y in translations]
tvs = np.array([[2, np.sqrt(3)], [2.5, -np.sqrt(3)/2]], dtype=np.float64)
clusterinfo7 = {"points": np.concatenate(points), "tvs": tvs}
