"""
Plot the octahedron in 3D
"""


import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


color_map = plt.get_cmap("tab10")
colors = color_map(range(color_map.N))
marker_size_center = 25
marker_size_corner = 22
line_width = 8

# Coordinates of the points
point0 = [0, 0, 0]
point1 = [1, -1, 0]
point2 = [1, 1, 0]
point3 = [-1, 1, 0]
point4 = [-1, -1, 0]
point5 = [0, 0, np.sqrt(2)]
point6 = [0, 0, -np.sqrt(2)]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
lines = [
    [np.array([point1, point2]), "solid", line_width, None],
    [np.array([point2, point3]), "solid", line_width, None],
    [np.array([point3, point4]), "dashed", line_width, None],
    [np.array([point4, point1]), "solid", line_width, None],

    [np.array([point0, point1]), "dashed", line_width/2, "gray"],
    [np.array([point0, point2]), "dashed", line_width/2, "gray"],
    [np.array([point0, point3]), "dashed", line_width/2, "gray"],
    [np.array([point0, point4]), "dashed", line_width/2, "gray"],
    [np.array([point0, point5]), "dashed", line_width/2, "gray"],
    [np.array([point0, point6]), "dashed", line_width/2, "gray"],

    [np.array([point5, point1]), "solid", line_width, None],
    [np.array([point5, point2]), "solid", line_width, None],
    [np.array([point5, point3]), "solid", line_width, None],
    [np.array([point5, point4]), "solid", line_width, None],

    [np.array([point6, point1]), "solid", line_width, None],
    [np.array([point6, point2]), "solid", line_width, None],
    [np.array([point6, point3]), "dashed", line_width, None],
    [np.array([point6, point4]), "dashed", line_width, None],
]

for endpoints, ls, lw, color in lines:
    ax.plot(
        endpoints[:, 0], endpoints[:, 1], endpoints[:, 2],
        lw=lw, ls=ls, color=color,
    )


# Plot the 7 points
ax.plot(
    point0[0:1], point0[1:2], point0[2:], ls="",
    marker="o", color=colors[0], markersize=marker_size_center
)
points = np.array([point1, point2, point3, point4, point5, point6])
ax.plot(
    points[:, 0], points[:, 1], points[:, 2], ls="",
    marker="o", color=colors[1], markersize=marker_size_corner
)
azimuth = 0
# elevation = -36
elevation = 20
ax.view_init(elevation, azimuth)

ax.set_axis_off()
plt.show()
fig.savefig(
    "Octahedron_azimuth={0}_elevation={1}.png".format(azimuth, elevation),
)
plt.close("all")
