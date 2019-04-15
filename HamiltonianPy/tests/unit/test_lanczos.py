"""
A test script for the lanczos module
"""


from scipy.sparse import random
from scipy.sparse.linalg import eigsh

import numpy as np
import pytest

from HamiltonianPy.constant import ANNIHILATION, CREATION
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.lanczos import set_float_point_precision, Lanczos
from HamiltonianPy.termofH import AoC, StateID, ParticleTerm


def test_init():
    dim = 100
    HM = random(dim, dim, format="csr") * 1j
    HM += random(dim, dim, format="csr")

    with pytest.raises(
            AssertionError, match="`HM` must be instance of csr_matrix"
    ):
        Lanczos(HM.tocsc())

    with pytest.raises(AssertionError, match="`HM` must be Hermitian"):
        Lanczos(HM)

    HM += HM.getH()
    lanczos_HM = Lanczos(HM)
    assert lanczos_HM._HMDim == dim


def test_ground_state():
    dim = 100
    HM = random(dim, dim, format="csr") * 1j
    HM += random(dim, dim, format="csr")
    HM += HM.getH()
    lanczos_HM = Lanczos(HM)
    GE = lanczos_HM.ground_state()
    GE_ref = np.linalg.eigvalsh(HM.toarray())[0]
    assert np.allclose(GE, GE_ref)


def test_call():
    site_num = 10
    sites = np.arange(site_num).reshape((-1, 1))
    state_indices_table = IndexTable(StateID(site=site) for site in sites)
    bases = np.arange(1<<site_num, dtype=np.uint64)

    HM = 0.0
    for i in range(site_num):
        C = AoC(CREATION, site=sites[i])
        A = AoC(ANNIHILATION, site=sites[(i+1) % site_num])
        HM += ParticleTerm([C, A]).matrix_repr(state_indices_table, bases)
    HM += HM.getH()

    (GE, ), GS = eigsh(HM, k=1, which="SA")

    excited_states = {}
    i, j = np.random.randint(0, site_num, 2)
    excited_states["Ci"] = AoC(CREATION, site=sites[i]).matrix_repr(
        state_indices_table, bases
    ).dot(GS)
    excited_states["Cj"] = AoC(CREATION, site=sites[j]).matrix_repr(
        state_indices_table, bases
    ).dot(GS)
    excited_states["Ai"] = AoC(ANNIHILATION, site=sites[i]).matrix_repr(
        state_indices_table, bases
    ).dot(GS)
    excited_states["Aj"] = AoC(ANNIHILATION, site=sites[j]).matrix_repr(
        state_indices_table, bases
    ).dot(GS)
    del GS

    lanczos_HM = Lanczos(HM)
    krylov_reprs_matrix, krylov_reprs_vectors = lanczos_HM(excited_states)

    omega = np.random.random() + 0.01j

    HM_Projected = krylov_reprs_matrix["Cj"]
    I = np.identity(HM_Projected.shape[0])
    res = np.vdot(
        krylov_reprs_vectors["Cj"]["Ci"],
        np.linalg.solve(
            (omega + GE) * I - HM_Projected, krylov_reprs_vectors["Cj"]["Cj"]
        )
    )
    res_ref = np.vdot(
        excited_states["Ci"],
        np.linalg.solve(
            (omega + GE) * np.identity(len(bases)) - HM.toarray(),
            excited_states["Cj"]
        )
    )
    assert np.allclose(res, res_ref)

    HM_Projected = krylov_reprs_matrix["Ai"]
    I = np.identity(HM_Projected.shape[0])
    res = np.vdot(
        krylov_reprs_vectors["Ai"]["Aj"],
        np.linalg.solve(
            (omega - GE) * I + HM_Projected, krylov_reprs_vectors["Ai"]["Ai"]
        )
    )
    res_ref = np.vdot(
        excited_states["Ai"],
        np.linalg.solve(
            (omega - GE) * np.identity(len(bases)) + HM.toarray(),
            excited_states["Aj"]
        )
    )
    assert np.allclose(res, res_ref)