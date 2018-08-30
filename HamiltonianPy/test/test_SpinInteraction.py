"""
A test script for the SpinINteraction class
"""


import numpy as np

from HamiltonianPy.constant import SPIN_UP, SPIN_DOWN
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import SiteID, SpinOperator, SpinInteraction

# sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])
sites = np.array([[0], [1]])
site_num = sites.shape[0]

site_ids = [SiteID(site=site) for site in sites]
table = IndexTable(site_ids)

spin_operators = []
for site in sites:
    spin_operators.append(SpinOperator(otype='x', site=site))
    spin_operators.append(SpinOperator(otype='y', site=site))
    spin_operators.append(SpinOperator(otype='z', site=site))
    spin_operators.append(SpinOperator(otype='p', site=site))
    spin_operators.append(SpinOperator(otype='m', site=site))
for item0 in spin_operators:
    for item1 in spin_operators:
        interaction = SpinInteraction((item0, item1))
        print(interaction)
        print(interaction.dagger())
        print("=" * 80)

spin0 = SpinOperator(otype='y', site=sites[0])
spin1 = SpinOperator(otype='z', site=sites[1])
interaction = SpinInteraction((spin0, spin1))
terms = interaction.Schwinger()
for term in terms:
    print(term)
    print("=" * 80)
spin0 = SpinOperator(otype='p', site=sites[0])
spin1 = SpinOperator(otype='m', site=sites[1])
interaction = SpinInteraction((spin0, spin1))
terms = interaction.Schwinger()
for term in terms:
    print(term)
    print("=" * 80)

otype0 = 'z'
otype1 = 'x'
spin0 = SpinOperator(otype=otype0, site=sites[0])
spin1 = SpinOperator(otype=otype1, site=sites[1])
interaction = SpinInteraction((spin0, spin1))
print(interaction.matrix_repr(table))
print("=" * 80)
operators = [(0, otype0), (1, otype1)]
data = SpinInteraction.matrix_function(operators, total_spin=2)
print(data)
print("=" * 80)


spin0 = SpinOperator(otype='x', site=sites[0])
spin1 = SpinOperator(otype='x', site=sites[1])
spin2 = SpinOperator(otype='z', site=sites[0])
spin3 = SpinOperator(otype='z', site=sites[1])
interaction0 = SpinInteraction((spin0, spin1))
interaction1 = SpinInteraction((spin2, spin3))
print(interaction0 * interaction1)
print("=" * 80)
print(interaction0 * 5)
print("=" * 80)
print(5 * interaction1)
