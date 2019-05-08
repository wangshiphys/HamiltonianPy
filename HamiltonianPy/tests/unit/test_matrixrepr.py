"""
A test script for the matrixrepr module
"""


import numpy as np

from HamiltonianPy.hilbertspace import base_vectors
from HamiltonianPy.quantumoperator.constant import ANNIHILATION, CREATION
from HamiltonianPy.quantumoperator.matrixrepr import matrix_function


def test_matrix_function():
    state_num = 64
    particle_num = 2
    bases = base_vectors((state_num, particle_num))

    terms = [
        [(i, CREATION), ((i+1) % state_num, ANNIHILATION)]
        for i in range(state_num)
    ]
    HM = 0.0
    for term in terms:
        HM += matrix_function(term, bases)
    HM += HM.getH()
    GE_Occupy = np.linalg.eigvalsh(HM.toarray())[0]

    ks = 2 * np.pi * np.arange(state_num) / state_num
    GE_KSpace = np.sum(np.sort(2 * np.cos(ks))[0:particle_num])

    assert np.allclose(GE_Occupy, GE_KSpace)
