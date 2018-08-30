"""
A test script for the AoC class
"""


import numpy as np

from HamiltonianPy.constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import AoC, SiteID, StateID

# sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# sites = np.array([[0, 0], [0.25, np.sqrt(3)/4], [0.5, 0]])
sites = np.array([[0], [1]])
state_num = 2 * sites.shape[0]
base = tuple(range(2 ** state_num))

site_ids = []
state_ids = []
aocs = []
for site in sites:
    site_ids.append(SiteID(site))
    state_ids.append(StateID(site=site, spin=SPIN_DOWN))
    state_ids.append(StateID(site=site, spin=SPIN_UP))
    aocs.append(AoC(otype=CREATION, site=site, spin=SPIN_DOWN))
    aocs.append(AoC(otype=CREATION, site=site, spin=SPIN_UP))
    aocs.append(AoC(otype=ANNIHILATION, site=site, spin=SPIN_DOWN))
    aocs.append(AoC(otype=ANNIHILATION, site=site, spin=SPIN_UP))

site_index_table = IndexTable(site_ids)
state_index_table = IndexTable(state_ids)

for aoc in aocs:
    print(aoc)
    print("The hash code: {}".format(hash(aoc)))
    print("The otype attribute : {}".format(aoc.otype))
    print("The state attribute: {}".format(aoc.state))
    print("The index: {}".format(aoc.getStateIndex(state_index_table)))
    print("The matrix representation:")
    print(aoc.matrix_repr(state_index_table, base))
    print("=" * 80)
