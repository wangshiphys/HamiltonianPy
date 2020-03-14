"""
Calculate the excitation spectrum of 1D nearest-neighbor tight-binding model.

The system is spin-1/2 and the Green-Function are calculated using Lanczos
algorithm. The particle number and spin are conserved.
"""


import logging
import sys
from time import time
import matplotlib.pyplot as plt
import numpy as np

from scipy.sparse.linalg import eigsh
from HamiltonianPy import SPIN_DOWN, SPIN_UP, ANNIHILATION, CREATION
from HamiltonianPy import lattice_generator, base_vectors, IndexTable
from HamiltonianPy import StateID, AoC, HoppingFactory, MultiKrylov
from HamiltonianPy.GreenFunction import RGFSolverLanczosMultiple

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(message)s",
)
logging.info("Program start running")


t = 1.0
site_num = 10
cell = lattice_generator("chain", num0=1)
cluster = lattice_generator("chain", num0=site_num)
intra_bonds, inter_bonds = cluster.bonds(nth=1)
terms = []
for bond in intra_bonds + inter_bonds:
    p0, p1 = bond.endpoints
    p0_eqv, trash = cluster.decompose(p0)
    p1_eqv, trash = cluster.decompose(p1)
    terms.append(HoppingFactory(p0_eqv, p1_eqv, spin0=SPIN_UP, coeff=t))
    terms.append(HoppingFactory(p0_eqv, p1_eqv, spin0=SPIN_DOWN, coeff=t))

state_indices_table = IndexTable(
    StateID(site=site, spin=spin)
    for spin in [SPIN_DOWN, SPIN_UP] for site in cluster.points
)
spin_up_state_indices = [
    state_indices_table(StateID(site=site, spin=SPIN_UP))
    for site in cluster.points
]
spin_down_state_indices = [
    state_indices_table(StateID(site=site, spin=SPIN_DOWN))
    for site in cluster.points
]

basis = base_vectors(
    [spin_up_state_indices, site_num // 2],
    [spin_down_state_indices, site_num // 2],
)

HM = 0.0
t0 = time()
for term in terms:
    HM += term.matrix_repr(state_indices_table, basis)
HM += HM.getH()
t1 = time()
logging.info("Time spend on HM: %.3fs", t1 - t0)

t0 = time()
(GE, ), GS = eigsh(HM, k=1, which="SA")
del HM
t1 = time()
logging.info("GE = %f", GE)
logging.info("Time spend on GE: %.3fs", t1 - t0)

cluster_gfs = {}
omegas = np.linspace(-3, 3, 601)
for spin in [SPIN_DOWN, SPIN_UP]:
    if spin == SPIN_UP:
        basis_h = base_vectors(
            [spin_up_state_indices, site_num // 2 - 1],
            [spin_down_state_indices, site_num // 2],
        )
        basis_p = base_vectors(
            [spin_up_state_indices, site_num // 2 + 1],
            [spin_down_state_indices, site_num // 2],
        )
    else:
        basis_h = base_vectors(
            [spin_up_state_indices, site_num // 2],
            [spin_down_state_indices, site_num // 2 - 1],
        )
        basis_p = base_vectors(
            [spin_up_state_indices, site_num // 2],
            [spin_down_state_indices, site_num // 2 + 1],
        )

    HM_H = 0.0
    HM_P = 0.0
    t0 = time()
    for term in terms:
        HM_H += term.matrix_repr(state_indices_table, basis_h)
        HM_P += term.matrix_repr(state_indices_table, basis_p)
    HM_H += HM_H.getH()
    HM_P += HM_P.getH()
    t1 = time()
    logging.info("Time spend on HM_H and HM_P: %.3fs", t1 - t0)

    As = []
    Cs = []
    excited_states_h = {}
    excited_states_p = {}
    t0 = time()
    for site in cluster.points:
        C = AoC(CREATION, site=site, spin=spin)
        A = AoC(ANNIHILATION, site=site, spin=spin)
        Cs.append(C)
        As.append(A)
        excited_states_h[A] = A.matrix_repr(
            state_indices_table, basis, left_bases=basis_h
        ).dot(GS)
        excited_states_p[C] = C.matrix_repr(
            state_indices_table, basis, left_bases=basis_p
        ).dot(GS)
    t1 = time()
    logging.info("Time spend on excited states: %.3fs", t1 - t0)
    del basis_h, basis_p

    t0 = time()
    projected_matrices, projected_vectors = MultiKrylov(HM_P, excited_states_p)
    del HM_P, excited_states_p
    projected_matrices_h, projected_vectors_h = MultiKrylov(
        HM_H, excited_states_h
    )
    del HM_H, excited_states_h
    projected_vectors.update(projected_vectors_h)
    projected_matrices.update(projected_matrices_h)
    t1= time()
    logging.info("Time spend on Lanczos projection: %.3fs", t1 - t0)

    t0 = time()
    cluster_gfs_temp = RGFSolverLanczosMultiple(
        omegas, As, Cs, GE, projected_matrices, projected_vectors,
        eta=0.05, structure="dict",
    )
    t1 = time()
    logging.info("Time spend on cluster Green-Functions: %.3fs", t1 - t0)
    cluster_gfs.update(cluster_gfs_temp)

kspace_gfs = 0.0
kpoints = np.dot([[i / site_num] for i in range(site_num + 1)], cell.bs)
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
