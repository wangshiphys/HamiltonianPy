import matplotlib.pyplot as plt

import numpy as np

DX = 0.1
DY = 0.1

a1 = np.array([1.0, 0.0], dtype=np.float64)
a2 = np.array([0.5, np.sqrt(3)/2], dtype=np.float64)

b1 = 2 * np.pi * np.array([1.0, -1/np.sqrt(3)], dtype=np.float64)
b2 = 2 * np.pi * np.array([0.0, 2/np.sqrt(3)], dtype=np.float64)

Gamma = np.array([0.0, 0.0], dtype=np.float64)

K0 = np.array([1.0/3, 1.0/np.sqrt(3)], dtype=np.float64) * 2 * np.pi
K1 = np.array([-1.0/3, 1.0/np.sqrt(3)], dtype=np.float64) * 2 * np.pi
K2 = np.array([-4*np.pi/3, 0.0], dtype=np.float64)
K3 = np.array([-1.0/3, -1.0/np.sqrt(3)], dtype=np.float64) * 2 * np.pi
K4 = np.array([1.0/3, -1.0/np.sqrt(3)], dtype=np.float64) * 2 * np.pi
K5 = np.array([4*np.pi/3, 0.0], dtype=np.float64)

M0 = np.array([0.0, 1.0/np.sqrt(3)], dtype=np.float64) * 2 * np.pi
M1 = np.array([-1.0, 1/np.sqrt(3)], dtype=np.float64) * np.pi
M2 = np.array([-1.0, -1/np.sqrt(3)], dtype=np.float64) * np.pi
M3 = np.array([0.0, -1.0/np.sqrt(3)], dtype=np.float64) * 2 * np.pi
M4 = np.array([1.0, -1/np.sqrt(3)], dtype=np.float64) * np.pi
M5 = np.array([1.0, 1/np.sqrt(3)], dtype=np.float64) * np.pi

Ks = [K0, K1, K2, K3, K4, K5]
Ms = [M0, M1, M2, M3, M4, M5]

p0 = np.array([0.0, 0.0], dtype=np.float64)
p1 = b2
p2 = -b1
p3 = -b1 - b2
p4 = -b2
p5 = b1
p6 = b1 + b2

KSpace_Points = [p0, p1, p2, p3, p4, p5, p6]
BZ_Boundaries = [(p1, p3), (p2, p4), (p3, p5), (p4, p6), (p5, p1), (p6, p2)]

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

# Show the boundaries of the first brillouin zone
for (x0, y0), (x1, y1) in BZ_Boundaries:
    ax0.plot([x0, x1], [y0, y1], linewidth=3)

for x, y in KSpace_Points:
    ax0.plot(x, y, marker='o', markersize=8)

# Show the Gamma point
ax0.plot(Gamma[0], Gamma[1], marker='o', markersize=12)
ax0.text(Gamma[0]+DX, Gamma[1]+DY, s=r"$\mathit{\Gamma}$", fontsize=20)

# Show the M points
for x, y in Ms:
    ax0.plot(x, y, marker='o', markersize=12)
    ax0.text(x+DX, y+DY, s=r"$\mathit{M}$", fontsize=20)

# Show the K points
for x, y in Ks:
    ax0.plot(x, y, marker='o', markersize=12)
    ax0.text(x+DX, y+DY, s=r"$\mathit{K}$", fontsize=20)

ax0.axis("equal")
ax0.axis("off")


lattice_info = r"""
    $\mathbf{a}_0 = (1.0, 0.0)$
    $\mathbf{a}_1 = (\frac{1}{2}, \frac{\sqrt{3}}{2})$
    $\mathbf{b}_1 = 2\pi(1, -\frac{1}{\sqrt{3}})$
    $\mathbf{b}_2 = 2\pi(0, \frac{2}{\sqrt{3}})$
    $\mathbf{M} = (0, \frac{2\pi}{\sqrt{3}})$
    $\mathbf{K} = (\frac{2\pi}{3}, \frac{2\pi}{\sqrt{3}})$
"""

ax1.text(0.5, 0.5, s=lattice_info, fontsize=20, ha="center", va="center")
ax1.axis("off")

fig.tight_layout()

plt.show()