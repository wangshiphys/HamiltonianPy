"""
A test script for the quantumstate module in quantumoperator subpackage.
"""


import numpy as np
import pytest

from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.quantumoperator.quantumstate import set_float_point_precision
from HamiltonianPy.quantumoperator.quantumstate import SiteID, StateID


def test_set_float_point_precision():
    with pytest.raises(AssertionError):
        set_float_point_precision(1.0)
    with pytest.raises(AssertionError):
        set_float_point_precision(-1)

    set_float_point_precision(precision=4)
    site_id0 = SiteID(site=[1.2, 2.3])
    assert site_id0._tuple_form == (12000, 23000)

    set_float_point_precision(precision=6)
    site_id1 = SiteID(site=[np.sqrt(2), np.sqrt(3)])
    # Instance that already exists does not affected;
    # Newly created instance use the new precision.
    assert site_id0._tuple_form == (12000, 23000)
    assert site_id1._tuple_form == (1414213, 1732050)

    # Restore the default value
    set_float_point_precision()


class TestSiteID:
    def test_init(self):
        with pytest.raises(AssertionError):
            SiteID(site=[0, 0, 0, 0])
        with pytest.raises(AssertionError):
            SiteID(site=[0, "a"])

        site_id = SiteID(site=[1.2, 2.3])
        assert np.all(site_id.site == (1.2, 2.3))
        assert np.all(site_id.coordinate == (1.2, 2.3))
        assert repr(site_id) == "SiteID(site=(1.2, 2.3))"

    def test_hash(self):
        site0 = [0, 0]
        site1 = [1E-6, 1E-6]
        site2 = [1E-4, 1E-4]

        site_id0 = SiteID(site=site0)
        site_id1 = SiteID(site=site1)
        site_id2 = SiteID(site=site2)
        assert hash(site_id0) == hash(site_id1)
        assert hash(site_id0) != hash(site_id2)
        assert hash(site_id1) != hash(site_id2)

        set_float_point_precision(precision=6)
        site_id0 = SiteID(site=site0)
        site_id1 = SiteID(site=site1)
        site_id2 = SiteID(site=site2)
        assert hash(site_id0) != hash(site_id1)
        assert hash(site_id0) != hash(site_id2)
        assert hash(site_id1) != hash(site_id2)

        # Restore the default value
        set_float_point_precision()

    def test_comparison(self):
        sites = np.array([[0, 0], [0.25, np.sqrt(3)/4], [0.5, 0]])
        site_id0 = SiteID(site=sites[0])
        site_id1 = SiteID(site=sites[1])
        site_id2 = SiteID(site=sites[2])

        with pytest.raises(TypeError, match="'<' not supported"):
            trash = site_id0 < 0
        with pytest.raises(TypeError, match="'<=' not supported"):
            trash = site_id1 <= 0
        with pytest.raises(TypeError, match="'>' not supported"):
            trash = site_id2 > 0
        with pytest.raises(TypeError, match="'>=' not supported"):
            trash = site_id0 >= 0

        assert not (site_id0 == 0)
        assert site_id1 != 0

        assert site_id0 < site_id1 < site_id2
        assert site_id2 > site_id1 > site_id0
        assert site_id0 == SiteID(site=[1E-8, 1E-7])
        assert site_id0 != site_id1

    def test_getIndex(self):
        sites = np.random.random((10, 3))
        site_indices_table = IndexTable(SiteID(site=site) for site in sites)
        for index, site_id in site_indices_table:
            assert index == site_id.getIndex(site_indices_table)


class TestStateID:
    def test_init(self):
        site = (1.2, 2.3)
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
        assert state_id.coordinate == site
        assert state_id.spin == spin
        assert state_id.orbit == orbit
        assert repr(state_id) == "StateID(site=(1.2, 2.3), spin=5, orbit=7)"

    def test_hash(self):
        site0 = [0, 0]
        site1 = [1E-6, 1E-6]
        site2 = [1E-4, 1E-4]
        state_id0 = StateID(site=site0)
        state_id1 = StateID(site=site1)
        state_id2 = StateID(site=site2)
        assert hash(state_id0) == hash(state_id1)
        assert hash(state_id0) != hash(state_id2)
        assert hash(state_id1) != hash(state_id2)

        set_float_point_precision(precision=6)
        state_id0 = StateID(site=site0)
        state_id1 = StateID(site=site1)
        state_id2 = StateID(site=site2)
        assert hash(state_id0) != hash(state_id1)
        assert hash(state_id0) != hash(state_id2)
        assert hash(state_id1) != hash(state_id2)

        # Restore the default value
        set_float_point_precision()

    def test_comparison(self):
        state_id0 = StateID(site=[0, 0], spin=0, orbit=0)
        state_id1 = StateID(site=[0, 0], spin=0, orbit=1)
        state_id2 = StateID(site=[0, 0], spin=1, orbit=0)
        state_id3 = StateID(site=[0, 0], spin=1, orbit=1)

        with pytest.raises(TypeError, match="'<' not supported"):
            trash = state_id0 < 0
        with pytest.raises(TypeError, match="'<=' not supported"):
            trash = state_id1 <= 0
        with pytest.raises(TypeError, match="'>' not supported"):
            trash = state_id2 > 0
        with pytest.raises(TypeError, match="'>=' not supported"):
            trash = state_id3 >= 0

        assert not (state_id0 == 0)
        assert state_id1 != 0
        assert state_id0 < state_id1 < state_id2 < state_id3
        assert state_id3 > state_id2 > state_id1 > state_id0
        assert state_id0 == StateID(site=[1E-8, 1E-9])
        assert state_id0 != state_id1

    def test_getIndex(self):
        spins = (0, 1, 2)
        orbits = (3, 4, 5)
        sites = np.random.random((5, 3))
        state_indices_table = IndexTable(
            StateID(site=site, spin=spin, orbit=orbit)
            for site in sites for spin in spins for orbit in orbits
        )

        for index, state_id in state_indices_table:
            assert index == state_id.getIndex(state_indices_table)
