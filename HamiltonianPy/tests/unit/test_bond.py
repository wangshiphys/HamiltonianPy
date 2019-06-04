"""
A test script for the bond module.
"""


import numpy as np
import pytest

from HamiltonianPy.bond import Bond, set_float_point_precision


@pytest.fixture(scope="module")
def endpoints():
    p0 = (0, 1.414)
    p1 = (1, np.sqrt(3))
    return p0, p1

@pytest.fixture(scope="module")
def bonds(endpoints):
    p0, p1 = endpoints
    bond0 = Bond(p0=p0, p1=p1, directional=True)
    bond1 = Bond(p0=p1, p1=p0, directional=True)
    bond2 = Bond(p0=p0, p1=p1, directional=False)
    bond3 = Bond(p0=p1, p1=p0, directional=False)
    separator = "*" * 80
    print(bond0)
    print(separator)
    print(bond1)
    print(separator)
    print(bond2)
    print(separator)
    print(bond3)
    print(separator)
    return {
        "p0-p1-T": bond0,
        "p1-p0-T": bond1,
        "p0-p1-F": bond2,
        "p1-p0-F": bond3,
    }


def test_set_float_point_precision(endpoints):
    p0, p1 = endpoints
    bond = Bond(p0=p0, p1=p1, directional=True)
    assert bond._tuple_form == ((0, 14140), (10000, 17320), True)

    set_float_point_precision(precision=6)
    bond = Bond(p0=p0, p1=p1, directional=True)
    assert bond._tuple_form == ((0, 1414000), (1000000, 1732050), True)

    # Restore to default value
    set_float_point_precision()


class TestBond:
    def test_init(self, endpoints, bonds):
        with pytest.raises(AssertionError, match="Invalid shape"):
            tmp  = Bond(p0=[0, 0, 0, 0], p1=[1, 1, 1, 1])
        with pytest.raises(AssertionError, match="Shape does not match"):
            tmp = Bond(p0=[0, 0], p1=[1, 1, 1])

        p0, p1 = endpoints
        bond = bonds["p0-p1-T"]
        assert np.all(bond._p0 == p0)
        assert np.all(bond._p1 == p1)
        assert bond._dim == 2
        assert bond._directional
        assert hasattr(bond, "_tuple_form")

    def test_directional_property(self, bonds):
        assert bonds["p0-p1-T"].directional
        assert bonds["p1-p0-T"].directional
        assert not bonds["p0-p1-F"].directional
        assert not bonds["p1-p0-F"].directional

    def test_endpoints_property(self, endpoints, bonds):
        p0, p1 = endpoints
        p0_new, p1_new = bonds["p0-p1-T"].endpoints
        assert np.all(p0_new == p0)
        assert np.all(p1_new == p1)

    def test_getLength(self, endpoints, bonds):
        p0, p1 = endpoints
        length_ref = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        assert bonds["p0-p1-T"].getLength() == length_ref
        assert bonds["p0-p1-F"].getLength() == length_ref
        assert bonds["p1-p0-T"].getLength() == length_ref
        assert bonds["p1-p0-F"].getLength() == length_ref

    def test_getDisplace(self, endpoints, bonds):
        p0, p1 = endpoints
        dr_ref = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        assert np.all(bonds["p0-p1-T"].getDisplace() == dr_ref)
        assert np.all(bonds["p1-p0-T"].getDisplace() == -dr_ref)

        with pytest.raises(NotImplementedError):
            bonds["p0-p1-F"].getDisplace()
        with pytest.raises(NotImplementedError):
            bonds["p1-p0-F"].getDisplace()

    def test_getAzimuth(self, endpoints, bonds):
        p0, p1 = endpoints
        theta = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        assert np.all(bonds["p0-p1-T"].getAzimuth(radian=True) == theta)

        with pytest.raises(NotImplementedError):
            bonds["p0-p1-F"].getAzimuth()

    def test_comparison(self, bonds):
        assert bonds["p0-p1-T"] == bonds["p0-p1-T"]
        assert bonds["p0-p1-F"] == bonds["p0-p1-F"]
        assert bonds["p1-p0-T"] == bonds["p1-p0-T"]
        assert bonds["p1-p0-F"] == bonds["p1-p0-F"]

        assert bonds["p0-p1-T"] != bonds["p0-p1-F"]
        assert bonds["p0-p1-T"] != bonds["p1-p0-T"]
        assert bonds["p0-p1-T"] != bonds["p1-p0-F"]
        assert bonds["p0-p1-F"] != bonds["p1-p0-T"]
        assert bonds["p0-p1-F"] == bonds["p1-p0-F"]
        assert bonds["p1-p0-T"] != bonds["p1-p0-F"]

        with pytest.raises(TypeError, match="not supported"):
            tmp = bonds["p1-p0-T"] < bonds["p1-p0-F"]
        with pytest.raises(TypeError, match="not supported"):
            tmp = bonds["p1-p0-T"] <= bonds["p1-p0-F"]
        with pytest.raises(TypeError, match="not supported"):
            tmp = bonds["p1-p0-T"] > bonds["p1-p0-F"]
        with pytest.raises(TypeError, match="not supported"):
            tmp = bonds["p1-p0-T"] >= bonds["p1-p0-F"]

    def test_oppositeTo(self, bonds):
        bond0 = bonds["p0-p1-T"]
        bond1 = bonds["p1-p0-T"]
        assert bond0.oppositeTo(bond1)
        assert bond1.oppositeTo(bond0)

        with pytest.raises(NotImplementedError):
            bonds["p0-p1-F"].oppositeTo(bonds["p1-p0-F"])
