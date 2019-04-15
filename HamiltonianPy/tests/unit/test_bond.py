"""
A test script for the bond module
"""


import numpy as np
import pytest

from HamiltonianPy.bond import Bond, set_float_point_precision


POINTS = np.array([[0, np.sqrt(2)], [1, np.sqrt(3)]])


def test_set_float_point_precision():
    bond = Bond(p0=POINTS[0], p1=POINTS[1], directional=True)
    assert bond._tuple_form == ((0, 14142), (10000, 17320), True)

    set_float_point_precision(precision=6)
    bond = Bond(p0=POINTS[0], p1=POINTS[1], directional=True)
    assert bond._tuple_form == ((0, 1414213), (1000000, 1732050), True)


def test_init():
    with pytest.raises(AssertionError):
        Bond(p0=(0, 0), p1=(1, 1))

    with pytest.raises(AssertionError):
        Bond(p0=np.array([0, 0, 0, 0]), p1=np.array([1, 1, 1, 1]))

    with pytest.raises(AssertionError):
        Bond(p0=np.array([0, 0]), p1=np.array([1, 1, 1]))

    with pytest.raises(AssertionError):
        Bond(p0=POINTS[0], p1=POINTS[1], directional="False")

    bond = Bond(p0=POINTS[0], p1=POINTS[1], directional=True)
    assert np.all(bond._p0 == POINTS[0])
    assert np.all(bond._p1 == POINTS[1])
    assert bond._dim == 2
    assert bond._directional
    assert hasattr(bond, "_tuple_form")


def test_directional_property():
    for directional in [True, False]:
        bond = Bond(p0=POINTS[0], p1=POINTS[1], directional=directional)
        assert bond.directional == directional


def test_getEndPoints():
    bond = Bond(p0=POINTS[0], p1=POINTS[1], directional=True)
    p0, p1 = bond.getEndpoints()
    assert np.all(p0 == POINTS[0])
    assert np.all(p1 == POINTS[1])


def test_getLength():
    bond = Bond(p0=np.array([0, 0]), p1=np.array([1, 1]))
    assert bond.getLength() == np.sqrt(2)


def test_getDisplace():
    bond = Bond(p0=POINTS[0], p1=POINTS[1], directional=True)
    dr = bond.getDisplace()
    assert np.all(dr == np.array([1, np.sqrt(3) - np.sqrt(2)]))

    bond = Bond(p0=POINTS[0], p1=POINTS[1], directional=False)
    with pytest.raises(NotImplementedError):
        bond.getDisplace()


def test_getAzimuth():
    bond = Bond(p0=POINTS[0], p1=POINTS[1], directional=True)
    theta = np.arctan2(np.sqrt(3) - np.sqrt(2), 1)
    assert np.all(bond.getAzimuth(radian=True) == theta)

    bond = Bond(p0=POINTS[0], p1=POINTS[1], directional=False)
    with pytest.raises(NotImplementedError):
        bond.getAzimuth()


def test_comparison():
    bond0 = Bond(p0=POINTS[0], p1=POINTS[1], directional=True)
    bond1 = Bond(p0=POINTS[0], p1=POINTS[1], directional=False)
    bond2 = Bond(p0=POINTS[1], p1=POINTS[0], directional=True)
    bond3 = Bond(p0=POINTS[1], p1=POINTS[0], directional=False)

    assert bond0 != bond1
    assert bond0 != bond2
    assert bond0 != bond3
    assert bond1 != bond2
    assert bond1 == bond3
    assert bond2 != bond3


def test_oppositeTo():
    bond0 = Bond(p0=POINTS[0], p1=POINTS[1], directional=True)
    bond1 = Bond(p0=POINTS[1], p1=POINTS[0], directional=True)
    assert bond0.oppositeTo(bond1)
    assert bond1.oppositeTo(bond0)

    bond0 = Bond(p0=POINTS[0], p1=POINTS[1], directional=False)
    bond1 = Bond(p0=POINTS[1], p1=POINTS[0], directional=False)
    with pytest.raises(NotImplementedError):
        bond0.oppositeTo(bond1)
