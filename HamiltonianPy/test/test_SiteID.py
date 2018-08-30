"""
A test script for the SiteID class
"""


import numpy as np

from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import SiteID

# sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
sites = np.array([[0, 0], [0.25, np.sqrt(3)/4], [0.5, 0]])

site_ids = [SiteID(site) for site in sites]
table = IndexTable(site_ids)

for site_id in site_ids:
    print(site_id)
    print("The hash code: {0}".format(hash(site_id)))
    print("The corresponding lattice site: {0}".format(site_id.site))
    print("The index: {0}".format(site_id.getIndex(table)))
    print("=" * 80)

for id0 in site_ids:
    for id1 in site_ids:
        print("{0} < {1} is {2}".format(id0, id1, id0 < id1))
        print("{0} == {1} is {2}".format(id0, id1, id0 == id1))
        print("{0} > {1} is {2}".format(id0, id1, id0 > id1))
        print("{0} <= {1} is {2}".format(id0, id1, id0 <= id1))
        print("{0} >= {1} is {2}".format(id0, id1, id0 >= id1))
        print("{0} != {1} is {2}".format(id0, id1, id0 != id1))
        print("=" * 80)
