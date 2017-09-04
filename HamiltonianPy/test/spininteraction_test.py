import sys
import os.path
sys.path.append(os.path.abspath("../"))

import numpy as np

from termofH import SiteID, SpinOperator, SpinInteraction
from indextable import IndexTable
from constant import SPIN_UP, SPIN_DOWN

#sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])
sites = np.array([[0], [1]])
site_num = len(sites)

siteids = []
for site in sites:
    siteid = SiteID(site=site)
    siteids.append(siteid)
sitemap = IndexTable(siteids)

spins = []
for site in sites:
    spins.append(SpinOperator(otype='x', site=site))
    spins.append(SpinOperator(otype='y', site=site))
    spins.append(SpinOperator(otype='z', site=site))
    spins.append(SpinOperator(otype='p', site=site))
    spins.append(SpinOperator(otype='m', site=site))
for item0 in spins:
    for item1 in spins:
        interaction = SpinInteraction((item0, item1))
        print(interaction)
        print()
        print(interaction.dagger())
        print("=" * 40)

spin0 = SpinOperator(otype='y', site=sites[0])
spin1 = SpinOperator(otype='z', site=sites[1])
interaction = SpinInteraction((spin0, spin1))
terms = interaction.Schwinger()
for term in terms:
    print(term)
    print("=" * 20)
spin0 = SpinOperator(otype='p', site=sites[0])
spin1 = SpinOperator(otype='m', site=sites[1])
interaction = SpinInteraction((spin0, spin1))
terms = interaction.Schwinger()
for term in terms:
    print(term)
    print("=" * 20)

otype0 = 'z'
otype1 = 'x'
spin0 = SpinOperator(otype=otype0, site=sites[0])
spin1 = SpinOperator(otype=otype1, site=sites[1])
interaction = SpinInteraction((spin0, spin1))
print(interaction.matrixRepr(sitemap))
print("=" * 20)
operators = [(0, otype0), (1, otype1)]
data = SpinInteraction.matrixFunc(operators, totspin=2)
print(data)
print("=" * 20)


spin0 = SpinOperator(otype='x', site=sites[0])
spin1 = SpinOperator(otype='x', site=sites[1])
spin2 = SpinOperator(otype='z', site=sites[0])
spin3 = SpinOperator(otype='z', site=sites[1])
interaction0 = SpinInteraction((spin0, spin1))
interaction1 = SpinInteraction((spin2, spin3))
print(interaction0 * interaction1)
print("=" * 20)
print(interaction0 * 5)
print("=" * 20)
print(5 * interaction1)
