"""
A test script for the Lanczos class
"""


from scipy.sparse.linalg import eigsh
from time import time

from HamiltonianPy.constant import CREATION, ANNIHILATION, SPIN_UP, SPIN_DOWN
from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.lattice import Lattice
from HamiltonianPy.lanczos import Lanczos
from HamiltonianPy.termofH import StateID, AoC, ParticleTerm

import numpy as np


site_num = 6
state_num = 2 * site_num
spins = (SPIN_DOWN, SPIN_UP)
points = np.arange(site_num).reshape((site_num, 1))
tvs = np.array([[site_num]])
cluster = Lattice(points=points, vectors=tvs)
intra, inter = cluster.bonds(nth=1, fold=True)
bonds = intra + inter

state_ids = []
for site in points:
    state_ids.append(StateID(site=site, spin=SPIN_DOWN))
    state_ids.append(StateID(site=site, spin=SPIN_UP))
state_index_table = IndexTable(state_ids)

bases = tuple(range(1<<state_num))

H = 0.0
t0 = time()
for bond in bonds:
    t1 = time()
    p0, p1 = bond.getEndpoints()
    c0_down = AoC(otype=CREATION, site=p0, spin=SPIN_DOWN)
    c0_up = AoC(otype=CREATION, site=p0, spin=SPIN_UP)
    a1_down = AoC(otype=ANNIHILATION, site=p1, spin=SPIN_DOWN)
    a1_up = AoC(otype=ANNIHILATION, site=p1, spin=SPIN_UP)
    H += ParticleTerm([c0_down, a1_down]).matrix_repr(
        state_index_table, right_bases=bases
    )
    H += ParticleTerm([c0_up, a1_up]).matrix_repr(
        state_index_table, right_bases=bases
    )
    t2 = time()
    print(repr(bond), flush=True)
    print("The time spend on this bond: ", t2 - t1, flush=True)
    print("=" * 80)
H += H.getH()
t3 = time()
print("The time spend on H matrix: ", t3 - t0, flush=True)

t0 = time()
(GE, ), GS = eigsh(H, k=1, which="SA")
t1 = time()
print("The time spend on GS: ", t1 - t0, flush=True)
print("The lanczos ground state energy: ", GE)

tmp = 4 * np.cos(2 * np.pi * np.arange(site_num) / site_num)
GE_exact = np.sum(tmp[tmp<0])
print("The exact ground state energy: ", GE_exact)

assert abs(GE - GE_exact) < (1e-8)


t0 = time()
vectors = dict()
for site in points:
    for spin in spins:
        aoc = AoC(otype=ANNIHILATION, site=site, spin=spin)
        vectors[aoc] = aoc.matrix_repr(state_index_table, bases).dot(GS)
t1 = time()
print("The time spend on excitation state: ", t1 - t0, flush=True)

t0 = time()
lanczos = Lanczos(HM=H)
HM_projs, vectors_projs = lanczos(vectors)
t1 = time()
print("The time spend on projection: ", t1 - t0, flush=True)

omega = 0.1 + 0.05j
site0 = points[1]
site1 = points[2]
spin0 = 1
spin1 = 1
key0 = AoC(otype=ANNIHILATION, site=site0, spin=spin0)
key1 = AoC(otype=ANNIHILATION, site=site1, spin=spin1)

HM_proj = HM_projs[key1]
dim = HM_proj.shape[0]
matrix = (omega - GE) * np.identity(dim) + HM_proj
vec = np.zeros((dim, 1))
vec[0] = 1
ket = np.linalg.solve(matrix, vec) * vectors_projs[key1][key1][0]
gf_lanczos = np.vdot(vectors_projs[key1][key0], ket)

print("GF lanczos: ", gf_lanczos, flush=True)
if site_num <= 6:
    matrix = (omega - GE) * np.identity(1<<state_num) + H.toarray()
    ket = np.linalg.solve(matrix, vectors[key1])
    gf_exact = np.vdot(vectors[key0], ket)
    print("GF exact: ", gf_exact)
