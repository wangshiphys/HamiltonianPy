from time import time

import matplotlib.pyplot as plt
import numpy as np
from _GFSolver import GFSolverExactMultiple

from HamiltonianPy import lattice_generator, SpinOperator, SpinInteraction


J = 1.0
site_num = 12
which = "pmz"
print("J={0:.1f}, num={1}, which={2}".format(J, site_num, which))

cell = lattice_generator("chain", num0=1)
cluster = lattice_generator("chain", num0=site_num)
intra_bonds, inter_bonds = cluster.bonds(nth=1)

HM = 0.0
t0 = time()
for bond in intra_bonds + inter_bonds:
    p0, p1 = bond.endpoints
    index0 = cluster.getIndex(p0, fold=True)
    index1 = cluster.getIndex(p1, fold=True)
    if which == "pmz":
        HM += SpinInteraction.matrix_function(
            [(index0, "z"), (index1, "z")], total_spin=site_num, coeff=J / 2
        )
        HM += SpinInteraction.matrix_function(
            [(index0, "p"), (index1, "m")], total_spin=site_num, coeff=J / 2
        )
    else:
        HM += SpinInteraction.matrix_function(
            [(index0, "x"), (index1, "x")], total_spin=site_num, coeff=J
        )
        HM += SpinInteraction.matrix_function(
            [(index0, "y"), (index1, "y")], total_spin=site_num, coeff=J
        )
        HM += SpinInteraction.matrix_function(
            [(index0, "z"), (index1, "z")], total_spin=site_num, coeff=J
        )
if which == "pmz":
    HM += HM.getH()
HM = HM.toarray()
t1 = time()
print("Time spend on HM: {0:.3f}s".format(t1 - t0))

values, vectors = np.linalg.eigh(HM)
GE = values[0]
GS = vectors[:, [0]]
print("GE = {0}".format((GE)))

As = []
Bs = []
excited_states = {}
t0 = time()
for site in cluster.points:
    index = cluster.getIndex(site, fold=False)
    Sp = SpinOperator("p", site=site)
    Sm = SpinOperator("m", site=site)
    As.append(Sp)
    Bs.append(Sm)
    excited_states[Sm] = SpinOperator.matrix_function(
        (index, "m"), total_spin=site_num
    ).dot(GS)
t1 = time()
print("Time spend on excited states: {0:.3f}s".format(t1 - t0))

omegas = np.linspace(0, 2.5, 251)
kpoints = np.dot([[i / site_num] for i in range(site_num + 1)], cell.bs)

t0 = time()
gfs_vs_omegas = GFSolverExactMultiple(
    omegas, As, Bs, GE, HM, excited_states, eta=0.05, sign="-"
)
t1 = time()
print("Time spend on cluster GFs: {0:.3f}s".format(t1 - t0))

gfs = 0.0
for A, B in gfs_vs_omegas:
    site0 = A.site
    site1 = B.site
    phase_factors = np.exp(1j * np.dot(kpoints, site1 - site0))
    gfs += np.outer(gfs_vs_omegas[(A, B)], phase_factors)
gfs /= site_num
spectrum = -2 * gfs.imag

fig, ax = plt.subplots()
cs = ax.contourf(
    range(len(kpoints)), omegas, spectrum,
    cmap="hot", levels=500,
)
ax.set_xticks(range(site_num + 1))
ax.grid(ls="dashed")
fig.colorbar(cs, ax=ax)
plt.show()
fig_name = "J={0:.1f}_num={1}_which={2}.png".format(J, site_num, which)
fig.savefig(fig_name, dpi=300)
plt.close("all")
