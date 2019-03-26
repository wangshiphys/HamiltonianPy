"""
A test script for the `extension` submodule
"""


from itertools import combinations_with_replacement, product
from time import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from HamiltonianPy.constant import CREATION, ANNIHILATION, SPIN_DOWN, SPIN_UP
from HamiltonianPy.hilbertspace import base_vectors
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.termofH import AoC, StateID, ParticleTerm


numx = 4
numy = 4
orbits = (0, 1)
spins = (SPIN_DOWN, SPIN_UP)
site_num = numx * numy
orbit_num = len(orbits)
spin_num = len(spins)
state_num = site_num * orbit_num * spin_num
e_num_total = np.random.randint(1, 5)

points = np.random.random((site_num, 2))
state_indices_table = IndexTable(
    [
        StateID(site=point, orbit=orbit, spin=spin)
        for point in points for orbit in orbits for spin in spins
    ]
)

HTerms = []
entries = []
row_indices = []
column_indices = []
for (p0, p1) in combinations_with_replacement(points, r=2):
    for orbit0, orbit1 in product(orbits, repeat=2):
        for spin0, spin1 in product(spins, repeat=2):
            coeff = np.random.random()
            c = AoC(CREATION, site=p0, spin=spin0, orbit=orbit0)
            a = AoC(ANNIHILATION, site=p1, spin=spin1, orbit=orbit1)
            row_indices.append(c.getStateIndex(state_indices_table))
            column_indices.append(a.getStateIndex(state_indices_table))
            entries.append(coeff)
            HTerms.append(ParticleTerm([c, a], coeff=coeff))

# Solve the Hamiltonian in occupation number representation with C
# implementation
t0 = time()
right_bases = base_vectors((state_num, e_num_total), dstructure="tuple")
t1 = time()

HM_C_Occupy = 0
for term in HTerms:
    HM_C_Occupy += term.matrix_repr(
        state_indices_table, right_bases, which_core="c"
    )
HM_C_Occupy += HM_C_Occupy.getH()
t2 = time()

if HM_C_Occupy.shape[0] > 5000:
    Values_C_Occupy = eigsh(
        HM_C_Occupy, k=1, which="SA", return_eigenvectors=False
    )
else:
    Values_C_Occupy = np.linalg.eigvalsh(HM_C_Occupy.toarray())
GE_C_Occupy = Values_C_Occupy[0]
t3 = time()

msg0 = "The time spend on generating the right_bases: {0}s"
msg1 = "The time spend on matrix representation: {0}s"
msg2 = "The time spend on ground state energy: {0}s"
msg3 = "The ground state energy: {0}"

print("In occupation number representation with C implementation:")
print(msg0.format(t1 - t0))
print(msg1.format(t2 - t1))
print(msg2.format(t3 - t2))
print(msg3.format(GE_C_Occupy))
print("=" * 80)

# Solve the Hamiltonian in occupation number representation with Python
# implementation
t0 = time()
right_bases = base_vectors((state_num, e_num_total), dstructure="array")
t1 = time()

HM_Py_Occupy = 0
for term in HTerms:
    HM_Py_Occupy += term.matrix_repr(
        state_indices_table, right_bases, which_core="py"
    )
HM_Py_Occupy += HM_Py_Occupy.getH()
t2 = time()

if HM_Py_Occupy.shape[0] > 5000:
    Values_Py_Occupy = eigsh(
        HM_Py_Occupy, k=1, which="SA", return_eigenvectors=False
    )
else:
    Values_Py_Occupy = np.linalg.eigvalsh(HM_Py_Occupy.toarray())
GE_Py_Occupy = Values_Py_Occupy[0]
t3 = time()

print("In occupation number representation with Python implementation:")
print(msg0.format(t1 - t0))
print(msg1.format(t2 - t1))
print(msg2.format(t3 - t2))
print(msg3.format(GE_Py_Occupy))
print("=" * 80)

# Solve the Hamiltonian in real space
HM_Real = csr_matrix(
    (entries, (row_indices, column_indices)), shape=(state_num, state_num)
)
HM_Real += HM_Real.getH()
t0 = time()
Values_Real = np.linalg.eigvalsh(HM_Real.toarray())
t1 = time()
GE_Real = np.sum(Values_Real[0:e_num_total])

print(msg2.format(t1 - t0))
print(msg3.format(GE_Real))
print("=" * 80)

assert np.abs(GE_Real - GE_C_Occupy) < 1e-12
assert np.abs(GE_Real - GE_Py_Occupy) < 1e-12
assert (HM_C_Occupy - HM_Py_Occupy).count_nonzero() == 0
