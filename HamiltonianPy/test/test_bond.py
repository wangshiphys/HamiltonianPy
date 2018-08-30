"""
A test script for the Bond class
"""


import numpy as np

from HamiltonianPy.bond import Bond


p0 = np.array([0, 0, 0])
p1 = np.array([1, 1, 1])
bond0 = Bond(p0=p0, p1=p1, directional=True)
bond1 = Bond(p0=p0, p1=p1, directional=False)
bond2 = Bond(p0=p1, p1=p0, directional=True)
bond3 = Bond(p0=p1, p1=p0, directional=False)

for bond in [bond0, bond1, bond2, bond3]:
    print(bond)
    print("hash code: ", hash(bond))
    print(repr(bond))
    print("=" * 80)

assert bond0 != bond1
assert bond0 != bond2
assert bond0 != bond3
assert bond1 != bond2
assert bond1 == bond3
assert bond2 != bond3
assert bond0.oppositeTo(bond2)
assert bond2.oppositeTo(bond0)