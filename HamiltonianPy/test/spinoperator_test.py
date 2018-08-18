import sys
import os.path
sys.path.append(os.path.abspath("../"))

import numpy as np

from termofH import SiteID, SpinOperator
from indextable import IndexTable
from constant import SPIN_UP, SPIN_DOWN

#sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])
sites = np.array([[0], [1]])
site_num = len(sites)

siteids = []
for site in sites:
    siteid = SiteID(site=site)
    SX = SpinOperator(otype='x', site=site)
    SY = SpinOperator(otype='y', site=site)
    SZ = SpinOperator(otype='z', site=site)
    SP = SpinOperator(otype='p', site=site)
    SM = SpinOperator(otype='m', site=site)
    siteids.append(siteid)
    print("SX:\n", SX, sep="")
    print("Hash value: ", hash(SX))
    print("Matrix:\n", SX.matrix(), sep="")
    print("=" * 40)
    print("SY:\n", SY, sep="")
    print("Hash value: ", hash(SY))
    print("Matrix:\n", SY.matrix(), sep="")
    print("=" * 40)
    print("SZ:\n", SZ, sep="")
    print("Hash value: ", hash(SZ))
    print("Matrix:\n", SZ.matrix(), sep="")
    print("=" * 40)
    print("SP:\n", SP, sep="")
    print("Hash value: ", hash(SP))
    print("Matrix:\n", SP.matrix(), sep="")
    print("=" * 40)
    print("SM:\n", SM, sep="")
    print("Hash value: ", hash(SM))
    print("Matrix:\n", SM.matrix(), sep="")
    print("=" * 40)
sitemap = IndexTable(siteids)


for otype in ('x', 'y', 'z', 'p', 'm'):
    S = SpinOperator(otype=otype, site=sites[0])
    terms = S.Schwinger()
    for term in terms:
        print(term)
        print('*' * 20)
    print('=' * 40)

for index in range(site_num):
    for otype in ('x', 'y', 'z', 'p', 'm'):
        operator = (index, otype)
        print("Operator: ", operator)
        print(SpinOperator.matrix_function(operator, total_spin=site_num))
        print("=" * 40)


for site in sites:
    for otype in ('x', 'y', 'z', 'p', 'm'):
        operator = SpinOperator(otype=otype, site=site)
        print("Operator:\n", operator, sep="")
        print(operator.matrix_repr(sitemap))
        print("=" * 40)
