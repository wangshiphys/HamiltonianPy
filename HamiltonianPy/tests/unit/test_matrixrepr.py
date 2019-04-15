"""
A test script for the matrixrepr module
"""

import os

import numpy as np

from HamiltonianPy.constant import ANNIHILATION, CREATION
from HamiltonianPy.hilbertspace import base_vectors
from HamiltonianPy.matrixrepr import matrix_function


def test_matrix_function():
    core_num = int(os.environ["NUMBER_OF_PROCESSORS"])
    threads_num = int(np.random.randint(1, core_num + 1))

    state_num = 64
    particle_num = 2
    bases = base_vectors((state_num, particle_num))

    terms = [
        [(i, CREATION), ((i+1) % state_num, ANNIHILATION)]
        for i in range(state_num)
    ]
    HM = 0.0
    for term in terms:
        HM += matrix_function(term, bases, threads_num=threads_num)
    HM += HM.getH()
    GE_Occupy = np.linalg.eigvalsh(HM.toarray())[0]

    ks = 2 * np.pi * np.arange(state_num) / state_num
    GE_KSpace = np.sum(np.sort(2 * np.cos(ks))[0:particle_num])

    assert np.allclose(GE_Occupy, GE_KSpace)
