"""
A test script for the ParticleTerm class
"""


import numpy as np

from HamiltonianPy.constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import AoC,StateID, ParticleTerm


# sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])
sites = np.array([[0], [1]])
state_num = 2 * sites.shape[0]
base = tuple(range(2 ** state_num))

state_ids = []
for site in sites:
    state_ids.append(StateID(site=site, spin=SPIN_DOWN))
    state_ids.append(StateID(site=site, spin=SPIN_UP))
table = IndexTable(state_ids)

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
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C0_DOWN, A0_DOWN))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C1_UP, A1_UP))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C1_DOWN, A1_DOWN))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C0_UP, A1_UP))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C1_UP, A0_UP))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C0_DOWN, A1_DOWN))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C1_DOWN, A0_DOWN))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C0_UP, A1_DOWN))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C1_DOWN, A0_UP))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C0_DOWN, A1_UP))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C1_UP, A0_DOWN))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C0_UP, A0_UP, C0_DOWN, A0_DOWN))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C1_UP, A1_UP, C1_DOWN, A1_DOWN))
print(term)
print(term.matrix_repr(table, base))
print("=" * 40)

term = ParticleTerm((C0_UP, A0_DOWN, C1_DOWN, A1_UP))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term = ParticleTerm((C0_DOWN, A0_UP, C1_UP, A1_DOWN))
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)

term0 = ParticleTerm((C0_DOWN, A0_UP))
term1 = ParticleTerm((C1_UP, A1_DOWN))
term = term0 * term1
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)
term = 5 * term
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)
term = term * 5
print(term)
print(term.matrix_repr(table, base))
print("=" * 80)
