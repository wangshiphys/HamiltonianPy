"""
A test script for the SpinOperator class
"""


import numpy as np
import pytest

from HamiltonianPy.constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import AoC, SiteID, SpinOperator


def test_init():
    with pytest.raises(AssertionError):
        SpinOperator(otype="X", site=np.array([0, 0]))

    site = np.array([1.2, 2.3])
    spin_operator = SpinOperator(otype="x", site=site)
    assert spin_operator.otype == "x"
    assert np.all(spin_operator.site == site)
    tmp = 'SpinOperator(otype="x", site=array([1.2, 2.3]))'
    assert repr(spin_operator) == tmp
    assert spin_operator.getSiteID() == SiteID(site=site)


def test_multiply():
    operator0 = SpinOperator("x", site=np.array([0, 1]))
    operator1 = SpinOperator("y", site=np.array([0, 0]))

    res = operator0 * operator1
    assert res.coeff == 1.0
    assert res.components == (operator1, operator0)

    res = operator0 * 0.5
    assert res.coeff == 0.5
    assert res.components == (operator0, )

    res = operator1 * operator0
    assert res.coeff == 1.0
    assert res.components == (operator1, operator0)

    res = 0.5 * operator1
    assert res.coeff == 0.5
    assert res.components == (operator1, )


def test_matrix():
    operator = SpinOperator("x", site=np.random.random(3))
    assert np.all(operator.matrix() == np.array([[0, 0.5], [0.5, 0.0]]))

    operator = SpinOperator("y", site=np.random.random(3))
    assert np.all(operator.matrix() == np.array([[0, -0.5j], [0.5j, 0.0]]))

    operator = SpinOperator("z", site=np.random.random(3))
    assert np.all(operator.matrix() == np.array([[0.5, 0], [0, -0.5]]))

    operator = SpinOperator("p", site=np.random.random(3))
    assert np.all(operator.matrix() == np.array([[0, 1], [0, 0]]))

    operator = SpinOperator("m", site=np.random.random(3))
    assert np.all(operator.matrix() == np.array([[0, 0], [1, 0]]))


def test_dagger():
    site = np.random.random(3)
    sx = SpinOperator("x", site=site)
    sy = SpinOperator("y", site=site)
    sz = SpinOperator("z", site=site)
    sp = SpinOperator("p", site=site)
    sm = SpinOperator("m", site=site)

    assert sx.dagger() == sx
    assert sy.dagger() == sy
    assert sz.dagger() == sz
    assert sp.dagger() == sm
    assert sm.dagger() == sp


def test_conjugate_of():
    site = np.random.random(3)
    sx = SpinOperator("x", site=site)
    sy = SpinOperator("y", site=site)
    sz = SpinOperator("z", site=site)
    sp = SpinOperator("p", site=site)
    sm = SpinOperator("m", site=site)

    assert sx.conjugate_of(sx)
    assert sy.conjugate_of(sy)
    assert sz.conjugate_of(sz)
    assert sp.conjugate_of(sm)
    assert sm.conjugate_of(sp)

    assert not sp.conjugate_of(sp)
    assert not sx.conjugate_of(sy)


def test_Schwinger():
    site = np.random.random(3)
    sx = SpinOperator("x", site=site)
    sy = SpinOperator("y", site=site)
    sz = SpinOperator("z", site=site)
    sp = SpinOperator("p", site=site)
    sm = SpinOperator("m", site=site)
    C_UP = AoC(CREATION, site=site, spin=SPIN_UP)
    C_DOWN = AoC(CREATION, site=site, spin=SPIN_DOWN)
    A_UP = AoC(ANNIHILATION, site=site, spin=SPIN_UP)
    A_DOWN = AoC(ANNIHILATION, site=site, spin=SPIN_DOWN)

    terms = sx.Schwinger()
    assert len(terms) == 2
    assert terms[0].coeff == 0.5
    assert terms[1].coeff == 0.5
    assert terms[0].components == (C_UP, A_DOWN)
    assert terms[1].components == (C_DOWN, A_UP)

    terms = sy.Schwinger()
    assert len(terms) == 2
    assert terms[0].coeff == -0.5j
    assert terms[1].coeff == 0.5j
    assert terms[0].components == (C_UP, A_DOWN)
    assert terms[1].components == (C_DOWN, A_UP)

    terms = sz.Schwinger()
    assert len(terms) == 2
    assert terms[0].coeff == 0.5
    assert terms[1].coeff == -0.5
    assert terms[0].components == (C_UP, A_UP)
    assert terms[1].components == (C_DOWN, A_DOWN)

    terms = sp.Schwinger()
    assert len(terms) == 1
    assert terms[0].coeff == 1
    assert terms[0].components == (C_UP, A_DOWN)

    terms = sm.Schwinger()
    assert len(terms) == 1
    assert terms[0].coeff == 1
    assert terms[0].components == (C_DOWN, A_UP)


def test_matrix_repr():
    sites = np.array([[0, 0], [1, 1]])
    site_indices_table = IndexTable(SiteID(site=site) for site in sites)

    sx = SpinOperator("x", site=sites[0])
    M = sx.matrix_repr(site_indices_table).toarray()
    M_ref = np.array(
        [[0, 0.5, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 0.5], [0, 0, 0.5, 0]]
    )
    assert np.all(M == M_ref)

    sy = SpinOperator("y", site=sites[1])
    M = sy.matrix_repr(site_indices_table).toarray()
    M_ref = np.array(
        [
            [0, 0, -0.5j, 0], [0, 0, 0, -0.5j],
            [0.5j, 0, 0, 0], [0, 0.5j, 0, 0]
        ]
    )
    assert np.all(M == M_ref)