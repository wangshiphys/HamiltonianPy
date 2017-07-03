import numpy as np

__all__ = ["cellinfo"]

points = np.array([[0, 0], [0, 1/np.sqrt(3)]])
tvs = np.array([[1, 0], [1.0/2, np.sqrt(3)/2]])
cellinfo = {"points": points, "tvs": tvs}

rotation = np.array([[1/2, np.sqrt(3)/2], [-np.sqrt(3)/2, 1/2]])
temp = np.array([[0, 1/np.sqrt(3)], [0, 2/np.sqrt(3)], 
                 [-1/2, 5/2/np.sqrt(3)], [1/2, 5/2/np.sqrt(3)]])

sites = [temp]
for i in range(1, 6):
    temp = np.dot(temp, rotation)
    sites.append(temp)
sites = np.concatenate(sites, axis=0)
cluster24info = {"points": sites, 
                 "tvs": np.array([[3, np.sqrt(3)], [0, 2 * np.sqrt(3)]])}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.figure()
    for x, y in sites:
        plt.plot(x, y, marker='o', markersize=9)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
