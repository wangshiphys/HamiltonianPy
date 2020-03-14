"""
Calculate the excitation spectrum of 1D nearest-neighbor Hubbard model.

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
from HamiltonianPy import AoC, HoppingFactory, HubbardFactory, MultiKrylov
from HamiltonianPy import lattice_generator, base_vectors, IndexTable, StateID
from HamiltonianPy.GreenFunction import RGFSolverLanczosMultiple

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s - %(message)s",
)
logging.info("Program start running")


t = 1.0
U = 4.0
site_num = 10
step = 0.01
cell = lattice_generator("chain", num0=1)
cluster = lattice_generator("chain", num0=site_num)
omegas = np.arange(-3, 8 + step, step)
kpoints = np.dot(np.arange(-0.5, 0.5 + step, step).reshape((-1, 1)), cell.bs)

intra_bonds, inter_bonds = cluster.bonds(nth=1)
HTerms = [HubbardFactory(site=site, coeff=U/2) for site in cluster.points]
for bond in intra_bonds:
    p0, p1 = bond.endpoints
    HTerms.append(HoppingFactory(p0, p1, spin0=SPIN_UP, coeff=t))
    HTerms.append(HoppingFactory(p0, p1, spin0=SPIN_DOWN, coeff=t))

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
for term in HTerms:
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

cluster_gfs_dict = {}
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
    for term in HTerms:
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
    t1 = time()
    logging.info("Time spend on Lanczos projection: %.3fs", t1 - t0)

    t0 = time()
    cluster_gfs_temp = RGFSolverLanczosMultiple(
        omegas, As, Cs, GE, projected_matrices, projected_vectors,
        eta=0.05, structure="dict",
    )
    t1 = time()
    logging.info("Time spend on cluster Green-Functions: %.3fs", t1 - t0)
    cluster_gfs_dict.update(cluster_gfs_temp)

Cell_Cs = [
    AoC(otype=CREATION, site=site, spin=spin)
    for spin in [SPIN_DOWN, SPIN_UP] for site in cell.points
]
Cell_As = [C.dagger() for C in Cell_Cs]
Cluster_Cs = [
    AoC(otype=CREATION, site=site, spin=spin)
    for spin in [SPIN_DOWN, SPIN_UP] for site in cluster.points
]
Cluster_As = [C.dagger() for C in Cluster_Cs]

cluster_gfs_array = np.zeros(
    (len(omegas), len(Cluster_Cs), len(Cluster_As)), dtype=np.complex128
)
for row, C in enumerate(Cluster_Cs):
    for col, A in enumerate(Cluster_As):
        key = (C.dagger(), A.dagger())
        try:
            cluster_gfs_array[:, row, col] = cluster_gfs_dict[key]
        except KeyError:
            pass
del cluster_gfs_dict

FTs = []
for i, C in enumerate(Cluster_Cs):
    p0 = C.site
    p0_eqv, dR0 = cell.decompose(p0)
    index0 = Cell_Cs.index(C.derive(site=p0_eqv))
    for j, A in enumerate(Cluster_As):
        p1 = A.site
        p1_eqv, dR1 = cell.decompose(p1)
        index1 = Cell_As.index(A.derive(site=p1_eqv))
        FTs.append((i, j, index0, index1, dR1 - dR0))

VTerms = []
for bond in inter_bonds:
    p0, p1 = bond.endpoints
    p0_eqv, dR0 = cluster.decompose(p0)
    p1_eqv, dR1 = cluster.decompose(p1)
    for spin in [SPIN_DOWN, SPIN_UP]:
        C = AoC(otype=CREATION, site=p0_eqv, spin=spin)
        A = AoC(otype=ANNIHILATION, site=p1_eqv, spin=spin)
        index0 = Cluster_Cs.index(C)
        index1 = Cluster_As.index(A)
        VTerms.append((index0, index1, t, dR1 - dR0))

final_gfs = []
t0 = time()
for kpoint in kpoints:
    VMatrix = np.zeros((len(Cluster_Cs), len(Cluster_As)), dtype=np.complex128)
    for i, j, coeff, dR in VTerms:
        VMatrix[i, j] += coeff * np.exp(1j * np.dot(kpoint, dR))
    VMatrix += VMatrix.T.conj()
    cpt_gfs = np.linalg.inv(np.linalg.inv(cluster_gfs_array) - VMatrix)

    temp = np.zeros(
        (len(omegas), len(Cell_Cs), len(Cell_As)), dtype=np.complex128
    )
    for i0, j0, i1, j1, dR in FTs:
        temp[:, i1, j1] += cpt_gfs[:, i0, j0] * np.exp(1j * np.dot(kpoint, dR))
    final_gfs.append(np.trace(temp, axis1=1, axis2=2))
final_gfs = np.array(final_gfs).T / site_num
t1 = time()
logging.info("Time spend on cpt and periodization: %.3fs", t1 - t0)

spectrum = -final_gfs.imag / np.pi
print(np.sum(spectrum, axis=0) * (omegas[1] - omegas[0]))

fig, ax = plt.subplots()
cs = ax.contourf(range(len(kpoints)), omegas, spectrum, cmap="hot", levels=500)
fig.colorbar(cs, ax=ax)
ax.set_xticks([0, len(kpoints)//2, len(kpoints) - 1])
ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
ax.grid(ls="dashed")
plt.show()
plt.close("all")
logging.info("Program stop running")
