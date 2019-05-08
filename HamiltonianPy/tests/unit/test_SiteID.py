"""
A test script for the SiteID class
"""


import numpy as np
import pytest

from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.quantumoperator.quantumstate import SiteID


def test_init():
    with pytest.raises(AssertionError):
        SiteID(site=[0, 0, 0, 0])
    with pytest.raises(AssertionError):
        SiteID(site=[0, "a"])

    site_id = SiteID(site=[1.2, 2.3])
    assert np.all(site_id.site == (1.2, 2.3))
    assert repr(site_id) == "SiteID(site=(1.2, 2.3))"


def test_hash():
    site_id0 = SiteID(site=[0, 0])
    site_id1 = SiteID(site=[1E-10, 1E-8])
    assert hash(site_id0) == hash(site_id1)


def test_comparison():
    sites = np.array([[0, 0], [0.25, np.sqrt(3)/4], [0.5, 0]])
    site_id0 = SiteID(site=sites[0])
    site_id1 = SiteID(site=sites[1])
    site_id2 = SiteID(site=sites[2])

    with pytest.raises(TypeError, match="'<' not supported"):
        site_id0 < 0
    with pytest.raises(TypeError, match="'>' not supported"):
        site_id0 > 0
    with pytest.raises(TypeError, match="'<=' not supported"):
        site_id0 <= 0
    with pytest.raises(TypeError, match="'>=' not supported"):
        site_id0 >= 0

    assert not (site_id0 == 0)
    assert site_id0 != 0

    assert site_id0 < site_id1
    assert site_id1 < site_id2
    assert site_id0 < site_id2
    assert site_id0 == SiteID(site=[1e-8, 1e-8])
    assert site_id0 != site_id1


def test_getIndex():
    sites = np.random.random((10, 3))
    site_indices_table = IndexTable(SiteID(site=site) for site in sites)
    for index, site_id in site_indices_table:
        assert index == site_id.getIndex(site_indices_table)
