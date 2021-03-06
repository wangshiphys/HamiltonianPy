"""
Calculate the excitation spectrum of 1D nearest-neighbor tight-binding model.

The system is spin-1/2 and the Green-Function are calculated using Lanczos
algorithm.
"""


import logging
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import eigsh

import HamiltonianPy as HP

SPINS = (HP.SPIN_DOWN, HP.SPIN_UP)

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(message)s",
)
logging.info("Program start running")


t = 1.0
site_num = 10
cell = HP.lattice_generator("chain", num0=1)
cluster = HP.lattice_generator("chain", num0=site_num)
intra_bonds, inter_bonds = cluster.bonds(nth=1)
terms = []
for bond in intra_bonds + inter_bonds:
    p0, p1 = bond.endpoints
    p0_eqv, trash = cluster.decompose(p0)
    p1_eqv, trash = cluster.decompose(p1)
    for spin in SPINS:
        terms.append(HP.HoppingFactory(p0_eqv, p1_eqv, spin0=spin, coeff=t))

basis = HP.base_vectors(2 * site_num)
state_indices_table = HP.IndexTable(
    HP.StateID(site=site, spin=spin)
    for spin in SPINS for site in cluster.points
)

HM = 0.0
t0 = time()
for term in terms:
    HM += term.matrix_repr(state_indices_table, basis)
HM += HM.getH()
t1 = time()
logging.info("Time spend on HM: %.3fs", t1 - t0)

t0 = time()
values, vectors = eigsh(HM, k=1, which="SA")
GE = values[0]
GS = vectors[:, 0]
t1 = time()
logging.info("GE = %f", GE)
logging.info("Time spend on GE: %.3fs", t1 - t0)

As = []
Cs = []
excited_states = {}
t0 = time()
for spin in SPINS:
    for site in cluster.points:
        C = HP.AoC(HP.CREATION, site=site, spin=spin)
        A = HP.AoC(HP.ANNIHILATION, site=site, spin=spin)
        Cs.append(C)
        As.append(A)
        excited_states[C] = C.matrix_repr(state_indices_table, basis).dot(GS)
        excited_states[A] = A.matrix_repr(state_indices_table, basis).dot(GS)
t1 = time()
logging.info("Time spend on excited states: %.3fs", t1 - t0)

omegas = np.linspace(-3, 3, 601)
kpoints = np.dot([[i / site_num] for i in range(site_num + 1)], cell.bs)

t0 = time()
projected_matrices, projected_vectors = HP.MultiKrylov(HM, excited_states)
t1 = time()
logging.info("Time spend on Lanczos projection: %.3fs", t1 - t0)

t0 = time()
cluster_gfs = HP.RGFSolverLanczosMultiple(
    omegas, As, Cs, GE, projected_matrices, projected_vectors,
    eta=0.05, structure="dict",
)
t1 = time()
logging.info("Time spend on cluster Green-Functions: %.3fs", t1 - t0)

kspace_gfs = 0.0
for A, B in cluster_gfs:
    site0 = A.site
    site1 = B.site
    phase_factors = np.exp(1j * np.dot(kpoints, site1 - site0))
    kspace_gfs += np.outer(cluster_gfs[(A, B)], phase_factors)
kspace_gfs /= site_num
spectrum = -kspace_gfs.imag / np.pi
print(np.sum(spectrum, axis=0) * (omegas[1] - omegas[0]))

fig, ax = plt.subplots()
cs = ax.contourf(range(len(kpoints)), omegas, spectrum, cmap="hot", levels=500)
fig.colorbar(cs, ax=ax)
ax.plot(range(len(kpoints)), 2 * t * np.cos(kpoints), alpha=0.75)
ax.set_xticks(range(site_num + 1))
ax.grid(ls="dashed")
plt.show()
plt.close("all")
logging.info("Program stop running")
