"""
A test script for the AoC class
"""

import numpy as np
import pytest

from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.quantumoperator.constant import ANNIHILATION, CREATION
from HamiltonianPy.quantumoperator.particlesystem import AoC, StateID


def test_init():
    site = np.array([0, 0])
    with pytest.raises(AssertionError):
        AoC(otype=2, site=site)

    creator = AoC(CREATION, site=site)
    assert creator.otype == CREATION
    assert creator.state == StateID(site=site)
    assert creator.coordinate == tuple(site)
    assert np.all(creator.site == site)
    assert creator.spin == 0
    assert creator.orbit == 0

    tmp = "AoC(otype=CREATION, site=(0, 0), spin=0, orbit=0)"
    assert repr(creator) == tmp


def test_getIndex():
    spins = (0, 1)
    orbits = (2, 3)
    sites = np.random.random((3, 3))
    otypes = (CREATION, ANNIHILATION)
    aoc_indices_table = IndexTable(
        AoC(otype, site, spin, orbit)
        for otype in otypes for site in sites
        for spin in spins for orbit in orbits
    )

    for index, aoc in aoc_indices_table:
        assert index == aoc.getIndex(aoc_indices_table)


def test_hash():
    site0 = np.array([0, 0], dtype=np.int64)
    site1 = np.array([1E-10, 1E-9], dtype=np.float64)
    aoc0 = AoC(CREATION, site=site0)
    aoc1 = AoC(CREATION, site=site1)
    assert hash(aoc0) == hash(aoc1)


def test_comparison():
    creator = AoC(CREATION, site=np.random.random(2))
    annihilator = AoC(ANNIHILATION, site=np.random.random(2))
    assert  creator < annihilator
    assert  annihilator > creator

    c0 = AoC(CREATION, site=np.array([0, 0]))
    c1 = AoC(CREATION, site=np.array([0, 1]))
    a0 = AoC(ANNIHILATION, site=np.array([0, 0]))
    a1 = AoC(ANNIHILATION, site=np.array([0, 1]))

    assert c0 < c1
    assert c1 > c0
    assert a0 > a1
    assert a1 < a0


def test_multiply():
    creator = AoC(CREATION, site=np.random.random(2))
    annihilator = AoC(ANNIHILATION, site=np.random.random(2))

    res = creator * annihilator
    assert res.coeff == 1.0
    assert res.components == (creator, annihilator)

    res = creator * 0.5
    assert res.coeff == 0.5
    assert res.components == (creator, )

    res = annihilator * creator
    assert res.coeff == 1.0
    assert res.components == (annihilator, creator)

    res = 0.5 * creator
    assert res.coeff == 0.5
    assert res.components == (creator, )


def test_dagger():
    creator = AoC(CREATION, site=np.array([0, 0]))
    annihilator = AoC(ANNIHILATION, site=np.array([0, 0]))
    assert creator.dagger() == annihilator
    assert annihilator.dagger() == creator


def test_conjugate_of():
    creator = AoC(CREATION, site=np.array([0, 0]))
    annihilator = AoC(ANNIHILATION, site=np.array([0, 0]))
    assert creator.conjugate_of(annihilator)
    assert annihilator.conjugate_of(creator)

    assert not creator.conjugate_of(creator)
    assert not annihilator.conjugate_of(annihilator)

    with pytest.raises(TypeError, match="not instance of this class"):
        creator.conjugate_of(0)


def test_same_state():
    creator = AoC(CREATION, site=np.array([0, 0]))
    annihilator = AoC(ANNIHILATION, site=np.array([0, 0]))

    assert creator.same_state(annihilator)
    assert creator.same_state(creator)
    assert annihilator.same_state(annihilator)
    assert annihilator.same_state(creator)

    assert not creator.same_state(AoC(CREATION, site=np.array([0, 1])))
    with pytest.raises(TypeError, match="not instance of this class"):
        assert creator.same_state(0)


def test_derive():
    creator = AoC(CREATION, site=np.array([0, 0]))
    annihilator = AoC(ANNIHILATION, site=np.array([0, 0]))
    assert creator.derive(otype=ANNIHILATION) == annihilator
    assert annihilator.derive(site=np.array([0, 1])) == AoC(
        ANNIHILATION, site=np.array([0, 1])
    )


def test_matrix_repr():
    # Currently the reliability of the `matrix_repr` method is guaranteed by
    # the `matrix_function` from `matrixrepr` module.
    assert True
