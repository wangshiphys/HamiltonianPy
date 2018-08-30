"""
A test script for the StateID class
"""


import numpy as np

from HamiltonianPy.constant import SPIN_DOWN, SPIN_UP
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import StateID

# sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
sites = np.array([[0, 0], [0.25, np.sqrt(3)/4], [0.5, 0]])

state_ids = []
for site in sites:
    state_ids.append(StateID(site=site, spin=SPIN_DOWN))
    state_ids.append(StateID(site=site, spin=SPIN_UP))
table = IndexTable(state_ids)

for state_id in state_ids:
    print(state_id)
    print("The hash code: {}".format(hash(state_id)))
    print("The site attribute: {}".format(state_id.site))
    print("The spin attribute: {}".format(state_id.spin))
    print("The orbit attribute: {}".format(state_id.orbit))
    print("The index: {}".format(state_id.getIndex(table)))
    print("=" * 80)

for id0 in state_ids:
    for id1 in state_ids:
        print("{0} < {1} is {2}".format(id0, id1, id0 < id1))
        print("{0} == {1} is {2}".format(id0, id1, id0 == id1))
        print("{0} > {1} is {2}".format(id0, id1, id0 > id1))
        print("{0} <= {1} is {2}".format(id0, id1, id0 <= id1))
        print("{0} >= {1} is {2}".format(id0, id1, id0 >= id1))
        print("{0} != {1} is {2}".format(id0, id1, id0 != id1))
        print("=" * 80)
