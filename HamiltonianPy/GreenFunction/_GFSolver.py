"""
This module provides functions to calculate the following Green-Function term:
    $G_{AB}(z) = <GS| A M^{-1} B |GS>$
where $M = z + (H - GE)$ or $M = z - (H - GE)$. $H$ is the model
Hamiltonian, $|GS>$ the ground state and $GE$ the ground state energy.
"""


__all__ = [
    "GFSolverExactSingle",
    "GFSolverExactMultiple",
    "GFSolverLanczosSingle",
    "GFSolverLanczosMultiple",
]


import numpy as np
from numba import jit, prange


def GFSolverExactSingle(
        omega, A, B, GE, HM, excited_states, eta=0.01, sign="+"
):
    z = omega + 1j * eta
    I = np.identity(HM.shape[0])
    M = (z - GE) * I + HM if sign == "+" else (z + GE) * I - HM

    B_dot_GS = excited_states[B]
    A_dagger_dot_GS = excited_states[A.dagger()]
    return np.vdot(A_dagger_dot_GS, np.linalg.solve(M, B_dot_GS))


@jit(nopython=True, cache=True, parallel=True)
def _IterateOverZsExact(zs, GE, HM, bras, kets, sign):
    z_num = zs.shape[0]
    I = np.identity(HM.shape[0])
    res = np.empty((z_num, bras.shape[0], kets.shape[1]), dtype=np.complex128)
    for i in prange(z_num):
        M = (zs[i] - GE) * I + HM if sign == "+" else (zs[i] + GE) * I - HM
        # The `@` operator performs matrix multiplication
        res[i] = bras @ np.linalg.solve(M, kets)
    return res


def GFSolverExactMultiple(
        omegas, As, Bs, GE, HM, excited_states, eta=0.01, sign="+"
):
    kets = np.concatenate(
        [excited_states[B] for B in Bs], axis=1
    ).astype(np.complex128)
    bras = np.concatenate(
        [excited_states[A.dagger()] for A in As], axis=1
    ).T.conj().astype(np.complex128)
    gfs_temp = _IterateOverZsExact(omegas + 1j * eta, GE, HM, bras, kets, sign)

    gfs_vs_omegas = dict()
    for row_index, A in enumerate(As):
        for col_index, B in enumerate(Bs):
            gfs_vs_omegas[(A, B)] = gfs_temp[:, row_index, col_index]
    return gfs_vs_omegas


def GFSolverLanczosSingle(
        omega, A, B, GE, projected_matrices, projected_vectors,
        eta=0.01, sign="+"
):
    z = omega + 1j * eta
    HM = projected_matrices[B]
    I = np.identity(HM.shape[0])
    ket = projected_vectors[B][B]
    bra_dagger = projected_vectors[B][A.dagger()]
    M = (z - GE) * I + HM if sign == "+" else (z + GE) * I - HM
    return np.vdot(bra_dagger, np.linalg.solve(M, ket))


@jit(nopython=True, cache=True, parallel=True)
def _IterateOverZsLanczos(zs, GE, HM, bra_dagger, ket, sign):
    z_num = zs.shape[0]
    I = np.identity(HM.shape[0])
    res = np.empty(z_num, dtype=np.complex128)
    for i in prange(z_num):
        M = (zs[i] - GE) * I + HM if sign == "+" else (zs[i] + GE) * I - HM
        res[i] = np.vdot(bra_dagger, np.linalg.solve(M, ket))
    return res


def GFSolverLanczosMultiple(
        omegas, As, Bs, GE, projected_matrices, projected_vectors,
        eta=0.01, sign="+"
):
    gfs_vs_omegas = {}
    for B in Bs:
        # HM is real matrix
        # ket and bra_dagger is complex128
        HM = projected_matrices[B]
        ket = np.ravel(projected_vectors[B][B])
        for A in As:
            bra_dagger = np.ravel(projected_vectors[B][A.dagger()])
            gfs_vs_omegas[(A, B)] = _IterateOverZsLanczos(
                omegas + 1j * eta, GE, HM, bra_dagger, ket, sign
            )
    return gfs_vs_omegas
