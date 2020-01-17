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
    assert lanczos._VIEW_AS_ZERO == 1E-10
    lanczos.set_threshold(1E-12)
    assert lanczos._VIEW_AS_ZERO == 1E-12
    lanczos.set_threshold()
    assert lanczos._VIEW_AS_ZERO == 1E-10

    v0 = np.array([0, 0, 0, 0], dtype=np.float64)
    lanczos.set_threshold(1E-12)
    v0[0] = 2E-12
    tmp = lanczos._starting_vector(N=4, v0=v0)
    assert np.all(tmp == np.array([[1], [0], [0], [0]]))

    v0[0] = 1E-12
    with pytest.raises(ValueError, match="`v0` is a zero vector"):
        lanczos._starting_vector(N=4, v0=v0)
    lanczos.set_threshold()


def test_Schimdt():
    v0 = np.array([1, 2, 3], dtype=np.float64)
    v1 = np.array([4, 5, 6], dtype=np.float64)
    vectors = [v0, v1]
    v2, v3 = lanczos.Schmidt(vectors)
    tmp = np.array([12.0/7, 3.0/7, -6.0/7], dtype=np.float64)
    assert np.allclose(v2, v0 / np.sqrt(14))
    assert np.allclose(v3, tmp / np.linalg.norm(tmp))


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
