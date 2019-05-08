"""
A test script for the bond module
"""


import numpy as np
import pytest

from HamiltonianPy.bond import *


def test_set_float_point_precision():
    p0 = (0, np.sqrt(2))
    p1 = (1, np.sqrt(3))
    bond = Bond(p0=p0, p1=p1, directional=True)
    assert bond._tuple_form == ((0, 14142), (10000, 17320), True)

    set_float_point_precision(precision=6)
    bond = Bond(p0=p0, p1=p1, directional=True)
    assert bond._tuple_form == ((0, 1414213), (1000000, 1732050), True)

def test_init():
    with pytest.raises(AssertionError, match="Invalid shape"):
        Bond(p0=[0, 0, 0, 0], p1=[1, 1, 1, 1])

    with pytest.raises(AssertionError, match="Shape does not match"):
        Bond(p0=[0, 0], p1=[1, 1, 1])

    p0 = (0, np.sqrt(2))
    p1 = (1, np.sqrt(3))
    bond = Bond(p0=p0, p1=p1, directional=True)
    assert np.all(bond._p0 == np.array(p0))
    assert np.all(bond._p1 == np.array(p1))
    assert bond._dim == 2
    assert bond._directional
    assert hasattr(bond, "_tuple_form")

def test_directional_property():
    p0 = (0, np.sqrt(2))
    p1 = (1, np.sqrt(3))
    for directional in [True, False]:
        bond = Bond(p0=p0, p1=p1, directional=directional)
        assert bond.directional == directional

def test_endpoints_property():
    p0 = (0, np.sqrt(2))
    p1 = (1, np.sqrt(3))
    bond = Bond(p0=p0, p1=p1, directional=True)
    p0_new, p1_new = bond.endpoints
    assert np.all(p0_new == np.array(p0))
    assert np.all(p1_new == np.array(p1))

def test_getLength():
    bond = Bond(p0=[0, 0], p1=[1, 1])
    assert bond.getLength() == np.sqrt(2)

def test_getDisplace():
    p0 = (0, np.sqrt(2))
    p1 = (1, np.sqrt(3))
    bond = Bond(p0=p0, p1=p1, directional=True)
    dr = bond.getDisplace()
    assert np.all(dr == np.array([1, np.sqrt(3) - np.sqrt(2)]))

    bond = Bond(p0=p0, p1=p1, directional=False)
    with pytest.raises(NotImplementedError):
        bond.getDisplace()

def test_getAzimuth():
    p0 = (0, np.sqrt(2))
    p1 = (1, np.sqrt(3))
    bond = Bond(p0=p0, p1=p1, directional=True)
    theta = np.arctan2(np.sqrt(3) - np.sqrt(2), 1)
    assert np.all(bond.getAzimuth(radian=True) == theta)

    bond = Bond(p0=p0, p1=p1, directional=False)
    with pytest.raises(NotImplementedError):
        bond.getAzimuth()

def test_comparison():
    p0 = (0, np.sqrt(2))
    p1 = (1, np.sqrt(3))
    bond0 = Bond(p0=p0, p1=p1, directional=True)
    bond1 = Bond(p0=p0, p1=p1, directional=False)
    bond2 = Bond(p0=p1, p1=p0, directional=True)
    bond3 = Bond(p0=p1, p1=p0, directional=False)

    assert bond0 != bond1
    assert bond0 != bond2
    assert bond0 != bond3
    assert bond1 != bond2
    assert bond1 == bond3
    assert bond2 != bond3

def test_oppositeTo():
    p0 = (0, np.sqrt(2))
    p1 = (1, np.sqrt(3))
    bond0 = Bond(p0=p0, p1=p1, directional=True)
    bond1 = Bond(p0=p1, p1=p0, directional=True)
    assert bond0.oppositeTo(bond1)
    assert bond1.oppositeTo(bond0)

    bond0 = Bond(p0=p0, p1=p1, directional=False)
    bond1 = Bond(p0=p1, p1=p0, directional=False)
    with pytest.raises(NotImplementedError):
        bond0.oppositeTo(bond1)
