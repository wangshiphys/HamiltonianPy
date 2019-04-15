"""
A test script for the ParticleTerm class
"""


import numpy as np
import pytest

from HamiltonianPy.constant import ANNIHILATION, CREATION
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import AoC, ParticleTerm


def test_coeff():
    sites = np.random.random((2, 3))
    C = AoC(CREATION, site=sites[0], spin=3, orbit=5)
    A = AoC(ANNIHILATION, site=sites[1], spin=9, orbit=13)
    term = ParticleTerm([C, A])

    assert term.coeff == 1.0

    with pytest.raises(AssertionError, match="Invalid coefficient"):
        term.coeff = "test"

    term.coeff = 0.5
    assert term.coeff == 0.5


def test_multiply():
    site = np.array([0, 0])
    C_UP = AoC(CREATION, site=site, spin=1)
    C_DOWN = AoC(CREATION, site=site, spin=0)
    A_UP = AoC(ANNIHILATION, site=site, spin=1)
    A_DOWN = AoC(ANNIHILATION, site=site, spin=0)

    term0 = ParticleTerm([C_UP, A_UP], coeff=1.5)
    term1 = ParticleTerm([C_DOWN, A_DOWN], coeff=0.5)
    res = term0 * term1
    assert res.coeff == 0.75
    assert res._aocs == (C_UP, A_UP, C_DOWN, A_DOWN)

    res = term1 * term0
    assert res.coeff == 0.75
    assert res._aocs == (C_DOWN, A_DOWN, C_UP, A_UP)

    res = term0 * C_DOWN * A_DOWN
    assert res.coeff == 1.5
    assert res._aocs == (C_UP, A_UP, C_DOWN, A_DOWN)

    res = term0 * 2j
    assert res.coeff == 3.0j
    assert res._aocs == (C_UP, A_UP)

    res = C_UP * term1 * A_UP
    assert res.coeff == 0.5
    assert res._aocs == (C_UP, C_DOWN, A_DOWN, A_UP)

    res = 0.2j * term1
    assert res.coeff == 0.1j
    assert res._aocs == (C_DOWN, A_DOWN)


def test_normalize():
    sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    C0 = AoC(CREATION, site=sites[0])
    A1 = AoC(ANNIHILATION, site=sites[1])
    C2 = AoC(CREATION, site=sites[2])
    A3 = AoC(ANNIHILATION, site=sites[3])

    aocs = [C0, A1, C2, A3]
    aocs_normalized, swap = ParticleTerm.normalize(aocs)
    assert aocs_normalized == [C0, C2, A3, A1]
    assert swap == 2


def test_dagger():
    sites = np.array([[0, 0], [0, 1]])
    C0 = AoC(CREATION, site=sites[0])
    A0 = AoC(ANNIHILATION, site=sites[0])
    C1 = AoC(CREATION, site=sites[1])
    A1 = AoC(ANNIHILATION, site=sites[1])

    term = ParticleTerm([C0, A1], coeff=1j)
    term_dagger = term.dagger()
    assert term_dagger.coeff == -1j
    assert term_dagger._aocs == (C1, A0)

    term = ParticleTerm([C0, A0])
    term_dagger = term.dagger()
    assert term_dagger.coeff == 1.0
    assert term_dagger._aocs == (C0, A0)

    term = ParticleTerm([C0, C1], coeff=1j)
    term_dagger = term.dagger()
    assert term_dagger.coeff == -1j
    assert term_dagger._aocs == (A1, A0)

    term = ParticleTerm([C0, C1, A1, A0])
    term_dagger = term.dagger()
    assert term_dagger.coeff == 1.0
    assert term_dagger._aocs == (C0, C1, A1, A0)


def test_matrix_repr():
    # Currently the reliability of the `matrix_repr` method is guaranteed by
    # the `matrix_function` from `matrixrepr` module.
    assert True