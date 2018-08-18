import sys
import os.path
sys.path.append(os.path.abspath("../"))

import numpy as np

from termofH import AoC, StateID, ParticleTerm
from indextable import IndexTable
from constant import CREATION, ANNIHILATION, SPIN_UP, SPIN_DOWN

#sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])
sites = np.array([[0], [1]])
state_num = 2 * len(sites)
base = tuple(range(2 ** state_num))

stateids = []
for site in sites:
    stateids.append(StateID(site=site, spin=SPIN_DOWN))
    stateids.append(StateID(site=site, spin=SPIN_UP))
statemap = IndexTable(stateids)

C0_UP = AoC(otype=CREATION, site=sites[0], spin=SPIN_UP)
C0_DOWN = AoC(otype=CREATION, site=sites[0], spin=SPIN_DOWN)
C1_UP = AoC(otype=CREATION, site=sites[1], spin=SPIN_UP)
C1_DOWN = AoC(otype=CREATION, site=sites[1], spin=SPIN_DOWN)
A0_UP = AoC(otype=ANNIHILATION, site=sites[0], spin=SPIN_UP)
A0_DOWN = AoC(otype=ANNIHILATION, site=sites[0], spin=SPIN_DOWN)
A1_UP = AoC(otype=ANNIHILATION, site=sites[1], spin=SPIN_UP)
A1_DOWN = AoC(otype=ANNIHILATION, site=sites[1], spin=SPIN_DOWN)

term = ParticleTerm((C0_UP, A0_UP))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C0_DOWN, A0_DOWN))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C1_UP, A1_UP))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C1_DOWN, A1_DOWN))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C0_UP, A1_UP))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C1_UP, A0_UP))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C0_DOWN, A1_DOWN))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C1_DOWN, A0_DOWN))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C0_UP, A1_DOWN))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C1_DOWN, A0_UP))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C0_DOWN, A1_UP))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C1_UP, A0_DOWN))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C0_UP, A0_UP, C0_DOWN, A0_DOWN))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C1_UP, A1_UP, C1_DOWN, A1_DOWN))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C0_UP, A0_DOWN, C1_DOWN, A1_UP))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term = ParticleTerm((C0_DOWN, A0_UP, C1_UP, A1_DOWN))
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)

term0 = ParticleTerm((C0_DOWN, A0_UP))
term1 = ParticleTerm((C1_UP, A1_DOWN))
term = term0 * term1
print(term)
print(term.matrix_repr(statemap, base))
print("=" * 40)
term = 5 * term
print(term)
print(term.matrixRepr(statemap, base))
print("=" * 40)
term = term * 5
print(term)
print(term.matrixRepr(statemap, base))
print("=" * 40)
