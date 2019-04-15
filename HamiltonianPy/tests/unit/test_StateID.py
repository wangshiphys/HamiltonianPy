"""
A test script for the StateID class
"""

import numpy as np
import pytest

from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import StateID


def test_init():
    site = np.array([1.2, 2.3])
    spin, orbit = 5, 7
    with pytest.raises(AssertionError):
        StateID(site=site, spin=1.0)
    with pytest.raises(AssertionError):
        StateID(site=site, spin=-1)
    with pytest.raises(AssertionError):
        StateID(site=site, orbit=1.0)
    with pytest.raises(AssertionError):
        StateID(site=site, orbit=-1)

    state_id = StateID(site=site, spin=spin, orbit=orbit)
    assert np.all(state_id.site == site)
    assert state_id.spin == spin
    assert state_id.orbit == orbit
    assert repr(state_id) == "StateID(site=array([1.2, 2.3]), spin=5, orbit=7)"


def test_hash():
    state_id0 = StateID(site=np.array([0, 0], dtype=np.int64))
    state_id1 = StateID(site=np.array([1E-10, 1E-8], dtype=np.float64))
    assert hash(state_id0) == hash(state_id1)


def test_comparison():
    site = np.array([0, 0])
    state_id0 = StateID(site=site, spin=0, orbit=0)
    state_id1 = StateID(site=site, spin=0, orbit=1)
    state_id2 = StateID(site=site, spin=1, orbit=0)
    state_id3 = StateID(site=site, spin=1, orbit=1)

    with pytest.raises(TypeError, match="'<' not supported"):
        state_id0< 0
    with pytest.raises(TypeError, match="'>' not supported"):
        state_id0 > 0
    with pytest.raises(TypeError, match="'<=' not supported"):
        state_id0 <= 0
    with pytest.raises(TypeError, match="'>=' not supported"):
        state_id0 >= 0

    assert not (state_id0 == 0)
    assert state_id0 != 0
    assert state_id0 < state_id1 < state_id2 < state_id3
    assert state_id3 > state_id2 > state_id1 > state_id0
    assert state_id0 == StateID(site=np.array([1E-8, 1E-9]))
    assert state_id0 != state_id1


def test_getIndex():
    spins = (0, 1, 2)
    orbits = (3, 4, 5)
    sites = np.random.random((5, 3))
    state_indices_table = IndexTable(
        StateID(site=site, spin=spin, orbit=orbit)
        for site in sites for spin in spins for orbit in orbits
    )
    for index, state_id in state_indices_table:
        assert index == state_id.getIndex(state_indices_table)