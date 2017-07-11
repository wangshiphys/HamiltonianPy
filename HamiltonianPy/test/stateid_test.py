import numpy as np

from HamiltonianPy.constant import SPIN_DOWN, SPIN_UP
from HamiltonianPy.indexmap import IndexMap
from HamiltonianPy.termofH import StateID

#sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])

stateids = []
for site in sites:
    stateid_down = StateID(site=site, spin=SPIN_DOWN)
    stateid_up = StateID(site=site, spin=SPIN_UP)
    stateids.append(stateid_down)
    stateids.append(stateid_up)
    print(stateid_down)
    print(stateid_up)
    print("The hash value: ", hash(stateid_down))
    print("The hash value: ", hash(stateid_up))
    print("The tuple form: ", stateid_down._tupleform())
    print("The tuple form: ", stateid_up._tupleform())
    print("The site attr : ", stateid_down.getSite())
    print("The site attr : ", stateid_up.getSite())
    print("The spin attr : ", stateid_down.getSpin())
    print("The spin attr : ", stateid_up.getSpin())
    print("The orbit attr : ", stateid_up.getOrbit())
    print("The orbit attr : ", stateid_up.getOrbit())
    print("=" * 60)

statemap = IndexMap(stateids)
for stateid in stateids:
    print("The index: ", stateid.getIndex(statemap))
    print("=" * 60)

for id0 in stateids:
    for id1 in stateids:
        print("id0 = {0}, id1 = {1}, id0 < id1 is {2}".format(id0, id1, id0 < id1))
        print("id0 = {0}, id1 = {1}, id0 == id1 is {2}".format(id0, id1, id0 == id1))
        print("id0 = {0}, id1 = {1}, id0 > id1 is {2}".format(id0, id1, id0 > id1))
        print("id0 = {0}, id1 = {1}, id0 <= id1 is {2}".format(id0, id1, id0 <= id1))
        print("id0 = {0}, id1 = {1}, id0 >= id1 is {2}".format(id0, id1, id0 >= id1))
        print("id0 = {0}, id1 = {1}, id0 != id1 is {2}".format(id0, id1, id0 != id1))
        print("=" * 80)
