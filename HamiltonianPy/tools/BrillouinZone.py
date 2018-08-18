"""
Draw the first Brillouin zone for triangle, honeycomb and kagome lattice
"""


from itertools import product

import matplotlib.pyplot as plt
import numpy as np


dx = dy = 0.3

_dtype = np.float64

# Translation vectors of real space and reciprocal space
As = np.array([[1.0, 0.0], [0.5, np.sqrt(3)/2]], dtype=_dtype)
Bs = 2 * np.pi * np.array(
    [[1.0, -1/np.sqrt(3)], [0.0, 2/np.sqrt(3)]], dtype=_dtype
)

# Some high symmetry points in the first Brillouin zone
Gamma = np.array([0.0, 0.0], dtype=_dtype)

Ks = 2 * np.pi * np.array(
    [[-2/3.0, 0.0], [-1/3.0, 1/np.sqrt(3)], [1/3.0, 1/np.sqrt(3)],
     [2/3.0, 0.0], [1/3.0, -1/np.sqrt(3)], [-1/3.0, -1/np.sqrt(3)]],
    dtype=_dtype
)

Ms = np.pi * np.array(
    [[-1.0, 1/np.sqrt(3)], [0.0, 2/np.sqrt(3)], [1.0, 1/np.sqrt(3)],
     [1.0, -1/np.sqrt(3)], [0.0, -2/np.sqrt(3)], [-1.0, -1/np.sqrt(3)]],
    dtype=_dtype
)

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_axis_off()

# Draw the boundary of the first Brillouin zone
color0 = "#6C71C4"
color1 = "#F57900"
for config in product(range(-1, 2), repeat=2):
    K = np.matmul(config, Bs)
    ax.plot(K[0], K[1], marker="o", mec=color0, mfc=color0, markersize=8)
    ax.text(K[0]+dx, K[1]-dy, str(config), size="large", color=color0)
    if config != (0, 0):
        center = K / 2
        orthogonal_vector = np.array([-K[1], K[0]])
        p0 = center + orthogonal_vector
        p1 = center - orthogonal_vector
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color=color1, lw=2)

size = "xx-large"
# Draw the Gamma point
color = "#EF2929"
ax.plot(Gamma[0], Gamma[1], marker='o', mec=color, mfc=color, markersize=12)
ax.text(
    Gamma[0]+dx, Gamma[1]+dy, s=r"$\mathit{\Gamma}$",
    size=size, color=color
)

# Draw the M points
color = "#3465A4"
for x, y in Ms:
    ax.plot(x, y, marker='o', mec=color, mfc=color, markersize=12)
    ax.text(x+dx, y+dy, s=r"$\mathit{M}$", size=size, color=color)

# Draw the K points
color= "#5C3566"
for x, y in Ks:
    ax.plot(x, y, marker='o', mec=color, mfc=color, markersize=12)
    ax.text(x+dx, y+dy, s=r"$\mathit{K}$", size=size, color=color)

plt.show()