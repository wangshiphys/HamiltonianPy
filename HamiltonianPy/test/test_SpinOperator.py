"""
A test script for the SpinOperator class
"""


import numpy as np

from HamiltonianPy.constant import SPIN_DOWN, SPIN_UP
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import SiteID, SpinOperator


# sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# sites = np.array([[0, 0], [1/4, np.sqrt(3)/4], [1/2, 0]])
sites = np.array([[0], [1]])
site_num = sites.shape[0]

site_ids = [SiteID(site=site) for site in sites]
table = IndexTable(site_ids)

operators = []
otypes = ("x", "y", "z", "p", "m")
for site in sites:
    for otype in otypes:
        operators.append(SpinOperator(otype=otype, site=site))

for operator in operators:
    print(operator)
    print("The hash code: {}".format(hash(operator)))
    print("Matrix:")
    print(operator.matrix())
    print("=" * 80)

for otype in otypes:
    S = SpinOperator(otype=otype, site=sites[0])
    terms = S.Schwinger()
    for term in terms:
        print(term)
        print("=" * 80)

for operator in operators:
    print(operator.matrix_repr(table))
    print("=" * 80)