"""
Calculate the spin excitation spectrum of 1D Heisenberg model.

$G(\\omega) = <GS| S_{i}^{+} M^{-1} S_{j}^{-} |GS>$
where $M = (\\omega + i \\eta) - (H - GE)$.
$H$ is the model Hamiltonian, $|GS>$ the ground state and $GE$ the ground
state energy.

The above Green-Function are calculated using Lanczos algorithm.
"""


import logging
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

from HamiltonianPy import lattice_generator, SpinOperator, SpinInteraction, \
    MultiKrylov
from HamiltonianPy.GreenFunction import GFSolverLanczosMultiple

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(message)s",
)
logging.info("Program start running")


J = 1.0
site_num = 18
cell = lattice_generator("chain", num0=1)
cluster = lattice_generator("chain", num0=site_num)
intra_bonds, inter_bonds = cluster.bonds(nth=1)

HM = 0.0
t0 = time()
for bond in intra_bonds + inter_bonds:
    p0, p1 = bond.endpoints
    index0 = cluster.getIndex(p0, fold=True)
    index1 = cluster.getIndex(p1, fold=True)
    HM += SpinInteraction.matrix_function(
        [(index0, "z"), (index1, "z")], total_spin=site_num, coeff=J / 2
    )
    HM += SpinInteraction.matrix_function(
        [(index0, "p"), (index1, "m")], total_spin=site_num, coeff=J / 2
    )
HM += HM.getH()
t1 = time()
logging.info("Time spend on HM: %.3fs", t1 - t0)

t0 = time()
(GE, ), GS = eigsh(HM, k=1, which="SA")
t1 = time()
logging.info("GE = %f", GE)
logging.info("Time spend on GE: %.3fs", t1 - t0)

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
logging.info("Time spend on excited states: %.3fs", t1 - t0)

omegas = np.linspace(0, 2.5, 251)
kpoints = np.dot([[i / site_num] for i in range(site_num + 1)], cell.bs)

t0 = time()
projected_matrices, projected_vectors = MultiKrylov(HM, excited_states)
t1 = time()
logging.info("Time spend on Lanczos projection: %.3fs", t1 - t0)

t0 = time()
cluster_gfs = GFSolverLanczosMultiple(
    omegas, As, Bs, GE,
    projected_matrices, projected_vectors,
    eta=0.05, sign="-", structure="dict",
)
t1 = time()
logging.info("Time spend on cluster Green-Functions: %.3fs", t1 - t0)

# Fourier transformation of the `cluster_gfs` to get gfs in k-space.
kspace_gfs = 0.0
for A, B in cluster_gfs:
    site0 = A.site
    site1 = B.site
    phase_factors = np.exp(1j * np.dot(kpoints, site1 - site0))
    kspace_gfs += np.outer(cluster_gfs[(A, B)], phase_factors)
kspace_gfs /= site_num
spectrum = -kspace_gfs.imag / np.pi

fig, ax = plt.subplots()
cs = ax.contourf(range(len(kpoints)), omegas, spectrum, cmap="hot", levels=500)
fig.colorbar(cs, ax=ax)
ax.set_xticks(range(site_num + 1))
ax.grid(ls="dashed")
plt.show()
plt.close("all")
logging.info("Program stop running")
