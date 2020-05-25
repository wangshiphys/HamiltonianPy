"""
A test script for the lanczos module.
"""


import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from HamiltonianPy import lanczos
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.quantumoperator.constant import ANNIHILATION, CREATION
from HamiltonianPy.quantumoperator.particlesystem import AoC, ParticleTerm
from HamiltonianPy.quantumoperator.quantumstate import StateID


def test_set_threshold():
    assert lanczos._VIEW_AS_ZERO == 1E-4
    lanczos.set_threshold(1E-12)
    assert lanczos._VIEW_AS_ZERO == 1E-12
    lanczos.set_threshold()
    assert lanczos._VIEW_AS_ZERO == 1E-4


@pytest.mark.slow
def test_MultiKrylov():
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    site_num = 12
    sites = np.arange(site_num).reshape((-1, 1))
    state_indices_table = IndexTable(StateID(site=site) for site in sites)
    bases = np.arange(1 << site_num, dtype=np.uint64)

    HM = 0.0
    for i in range(site_num):
        C = AoC(CREATION, site=sites[i])
        A = AoC(ANNIHILATION, site=sites[(i + 1) % site_num])
        HM += ParticleTerm([C, A]).matrix_repr(state_indices_table, bases)
    HM += HM.getH()
    (GE, ), GS = eigsh(HM, k=1, which="SA")

    excited_states = {}
    i, j = np.random.randint(0, site_num, 2)
    keys = ["Ci", "Cj", "Ai", "Aj"]
    otypes = [CREATION, CREATION, ANNIHILATION, ANNIHILATION]
    indices = [i, j, i, j]
    for key, otype, index in zip(keys, otypes, indices):
        excited_states[key] = AoC(otype, site=sites[i]).matrix_repr(
            state_indices_table, bases
        ).dot(GS)


    krylovs_matrix, krylovs_vectors = lanczos.MultiKrylov(HM, excited_states)
    omega = np.random.random() + 0.01j

    pairs = [("Ci", "Cj"), ("Ai", "Aj")]
    for coeff, (key0, key1) in zip([-1, 1], pairs):
        HMProjected = krylovs_matrix[key1]
        krylov_space_dim = HMProjected.shape[0]
        I = np.identity(krylov_space_dim)
        bra = krylovs_vectors[key1][key0]
        ket = krylovs_vectors[key1][key1]
        gf = np.vdot(
            bra, np.linalg.solve(
                (omega + coeff * GE) * I + coeff * HMProjected, ket
            )
        )
        I = np.identity(len(bases))
        gf_ref = np.vdot(
            excited_states[key0],
            np.linalg.solve(
                (omega + coeff * GE) * I + coeff * HM.toarray(),
                excited_states[key1]
            )
        )
        assert np.allclose(gf, gf_ref)
