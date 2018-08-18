import sys
import os.path
sys.path.append(os.path.abspath("../"))

import numpy as np

from termofH import AoC, StateID
from indextable import IndexTable
from constant import CREATION, ANNIHILATION, SPIN_UP, SPIN_DOWN

#sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])
sites = np.array([[0], [1]])
state_num = 2 * len(sites)
base = tuple(range(2 ** state_num))

Cs = []
As = []
stateids = []
for site in sites:
    state_up = StateID(site=site, spin=SPIN_UP)
    state_down = StateID(site=site, spin=SPIN_DOWN)
    stateids.append(state_down)
    stateids.append(state_up)
    C_up = AoC(otype=CREATION, site=site, spin=SPIN_UP)
    C_down = AoC(otype=CREATION, site=site, spin=SPIN_DOWN)
    A_up = AoC(otype=ANNIHILATION, site=site, spin=SPIN_UP)
    A_down = AoC(otype=ANNIHILATION, site=site, spin=SPIN_DOWN)
    Cs.append(C_down)
    Cs.append(C_up)
    As.append(A_down)
    As.append(A_up)
    print("C_up:\n", C_up, sep="")
    print("Hash value: ", hash(C_up))
    print("StateID:\n", C_up.state, sep="")
    print("=" * 40)
    print("C_down:\n", C_down, sep="")
    print("Hash value: ", hash(C_down))
    print("StateID:\n", C_down.state, sep="")
    print("=" * 40)
    print("A_up:\n", A_up, sep="")
    print("Hash value: ", hash(A_up))
    print("StateID:\n", A_up.state, sep="")
    print("=" * 40)
    print("A_down:\n", A_down, sep="")
    print("Hash value: ", hash(A_down))
    print("StateID:\n", A_down.state, sep="")
    print("=" * 40)

statemap = IndexTable(stateids)
for C in Cs:
    print("Operator:\n", C, sep="")
    print("Index: ", C.getStateIndex(statemap))
    print(C.matrix_repr(statemap, right_bases=base))
    print("=" * 60)
for A in As:
    print("Operator:\n", A, sep="")
    print("Index: ", A.getStateIndex(statemap))
    print(A.matrix_repr(statemap, right_bases=base))
    print("=" * 60)

for index in range(state_num):
    c = (index, CREATION)
    a = (index, ANNIHILATION)
    print(AoC.matrix_function(c, base))
    print()
    print(AoC.matrix_function(a, base))
    print("=" * 60)

operators = Cs + As

for item0 in operators:
    for item1 in operators:
        print("item0:\n", item0, sep="")
        print("item1:\n", item1, sep="")
        print("item0 < item1", item0<item1)
        print("item0 > item1", item0>item1)
        print("=" * 40)

