"""
A test script for the SpinInteraction class
"""


import numpy as np
import pytest

from HamiltonianPy.constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import AoC, SiteID, SpinOperator, SpinInteraction


def test_coeff():
    sites = np.random.random((2, 3))
    sx = SpinOperator("x", site=sites[0])
    sy = SpinOperator("y", site=sites[1])
    term = SpinInteraction([sx, sy])

    assert term.coeff == 1.0

    with pytest.raises(AssertionError, match="Invalid coefficient"):
        term.coeff = "test"

    term.coeff = 0.5
    assert term.coeff == 0.5


def test_multiply():
    sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    sx = SpinOperator("x", site=sites[0])
    sy = SpinOperator("y", site=sites[1])
    sz = SpinOperator("z", site=sites[2])

    term0 = SpinInteraction([sx, sy], coeff=2.0)
    term1 = SpinInteraction([sy, sz], coeff=-1.0)

    res = term0 * term1
    assert res.coeff == -2
    assert res.components == (sx, sy, sy, sz)

    res = term1 * term0
    assert res.coeff == -2
    assert res.components == (sx, sy, sy, sz)

    res = term0 * sz
    assert res.coeff == 2
    assert res.components == (sx, sy, sz)

    res = term0 * 0.5
    assert res.coeff == 1
    assert res.components == (sx, sy)

    res = sz * term0
    assert res.coeff == 2
    assert res.components == (sx, sy, sz)

    res = 0.5 * term0
    assert res.coeff == 1
    assert res.components == (sx, sy)


def test_dagger():
    sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for otype in ("x", "y", "z"):
        s0 = SpinOperator(otype, site=sites[0])
        s1 = SpinOperator(otype, site=sites[1])
        term = SpinInteraction([s0, s1], coeff=1j)
        term_dagger = term.dagger()
        assert term_dagger.coeff == -1j
        assert term_dagger.components == (s0, s1)
    sp0 = SpinOperator("p", site=sites[0])
    sp1 = SpinOperator("p", site=sites[1])
    sm0 = SpinOperator("m", site=sites[0])
    sm1 = SpinOperator("m", site=sites[1])
    term = SpinInteraction((sp0, sm1), coeff=1j)
    term_dagger = term.dagger()
    assert term_dagger.coeff == -1j
    assert term_dagger.components == (sm0, sp1)

    sx = SpinOperator("x", site=sites[0])
    sp = SpinOperator("p", site=sites[0])
    sm = SpinOperator("m", site=sites[0])
    term = SpinInteraction((sx, sp, sm), coeff=1j)
    term_dagger = term.dagger()
    assert term_dagger.coeff == -1j
    assert term_dagger.components == (sp, sm, sx)


def test_Schwinger():
    sites = np.array([[0, 0], [1, 1]])
    sx0 = SpinOperator("x", site=sites[0])
    sx1 = SpinOperator("x", site=sites[1])
    sy0 = SpinOperator("y", site=sites[0])
    sy1 = SpinOperator("y", site=sites[1])
    sz0 = SpinOperator("z", site=sites[0])
    sz1 = SpinOperator("z", site=sites[1])

    C0_UP = AoC(CREATION, site=sites[0], spin=SPIN_UP)
    C1_UP = AoC(CREATION, site=sites[1], spin=SPIN_UP)
    C0_DOWN = AoC(CREATION, site=sites[0], spin=SPIN_DOWN)
    C1_DOWN = AoC(CREATION, site=sites[1], spin=SPIN_DOWN)
    A0_UP = AoC(ANNIHILATION, site=sites[0], spin=SPIN_UP)
    A1_UP = AoC(ANNIHILATION, site=sites[1], spin=SPIN_UP)
    A0_DOWN = AoC(ANNIHILATION, site=sites[0], spin=SPIN_DOWN)
    A1_DOWN = AoC(ANNIHILATION, site=sites[1], spin=SPIN_DOWN)

    sx0_times_sx1 = SpinInteraction((sx0, sx1))
    sy0_times_sy1 = SpinInteraction((sy0, sy1))
    sz0_times_sz1 = SpinInteraction((sz0, sz1))

    sx0_times_sx1_schwinger = sx0_times_sx1.Schwinger()
    sy0_times_sy1_schwinger = sy0_times_sy1.Schwinger()
    sz0_times_sz1_schwinger = sz0_times_sz1.Schwinger()

    assert sx0_times_sx1_schwinger[0].coeff == 0.25
    assert sx0_times_sx1_schwinger[0].components == (
        C0_UP, A0_DOWN, C1_UP, A1_DOWN
    )
    assert sx0_times_sx1_schwinger[1].coeff == 0.25
    assert sx0_times_sx1_schwinger[1].components == (
        C0_UP, A0_DOWN, C1_DOWN, A1_UP
    )
    assert sx0_times_sx1_schwinger[2].coeff == 0.25
    assert sx0_times_sx1_schwinger[2].components == (
        C0_DOWN, A0_UP, C1_UP, A1_DOWN
    )
    assert sx0_times_sx1_schwinger[3].coeff == 0.25
    assert sx0_times_sx1_schwinger[3].components == (
        C0_DOWN, A0_UP, C1_DOWN, A1_UP
    )

    assert sy0_times_sy1_schwinger[0].coeff == -0.25
    assert sy0_times_sy1_schwinger[0].components == (
        C0_UP, A0_DOWN, C1_UP, A1_DOWN
    )
    assert sy0_times_sy1_schwinger[1].coeff == 0.25
    assert sy0_times_sy1_schwinger[1].components == (
        C0_UP, A0_DOWN, C1_DOWN, A1_UP
    )
    assert sy0_times_sy1_schwinger[2].coeff == 0.25
    assert sy0_times_sy1_schwinger[2].components == (
        C0_DOWN, A0_UP, C1_UP, A1_DOWN
    )
    assert sy0_times_sy1_schwinger[3].coeff == -0.25
    assert sy0_times_sy1_schwinger[3].components == (
        C0_DOWN, A0_UP, C1_DOWN, A1_UP
    )

    assert sz0_times_sz1_schwinger[0].coeff == 0.25
    assert sz0_times_sz1_schwinger[0].components == (
        C0_UP, A0_UP, C1_UP, A1_UP
    )
    assert sz0_times_sz1_schwinger[1].coeff == -0.25
    assert sz0_times_sz1_schwinger[1].components == (
        C0_UP, A0_UP, C1_DOWN, A1_DOWN
    )
    assert sz0_times_sz1_schwinger[2].coeff == -0.25
    assert sz0_times_sz1_schwinger[2].components == (
        C0_DOWN, A0_DOWN, C1_UP, A1_UP
    )
    assert sz0_times_sz1_schwinger[3].coeff == 0.25
    assert sz0_times_sz1_schwinger[3].components == (
        C0_DOWN, A0_DOWN, C1_DOWN, A1_DOWN
    )


def test_matrix_repr():
    sites = np.array([[0, 0], [1, 1]])
    site_indices_table = IndexTable(SiteID(site=site) for site in sites)
    SX = np.array([[0, 0.5], [0.5, 0]])
    SY = np.array([[0, -0.5j], [0.5j, 0]])
    SZ = np.array([[0.5, 0], [0, -0.5]])
    I = np.array([[1, 0], [0, 1]])

    for otype, SMatrix in zip(["x", "y", "z"], [SX, SY,SZ]):
        s0 = SpinOperator(otype, site=sites[0])
        s1 = SpinOperator(otype, site=sites[1])
        M = SpinInteraction((s0, s1)).matrix_repr(site_indices_table).toarray()
        assert np.all(M == np.kron(SMatrix, SMatrix))

    sx0 = SpinOperator("x", site=sites[0])
    sy0 = SpinOperator("y", site=sites[0])
    sz0 = SpinOperator("z", site=sites[0])

    M = SpinInteraction((sx0, sy0)).matrix_repr(site_indices_table).toarray()
    assert np.all(M == np.kron(I, np.dot(SX, SY)))

    M = SpinInteraction((sy0, sz0)).matrix_repr(site_indices_table).toarray()
    assert np.all(M == np.kron(I, np.dot(SY, SZ)))

    M = SpinInteraction((sx0, sz0)).matrix_repr(site_indices_table).toarray()
    assert np.all(M == np.kron(I, np.dot(SX, SZ)))