"""
A test script for the hilbertspace module
"""


import numpy as np
import pytest

from HamiltonianPy.hilbertspace import *


def test_SimpleHilbertSpace():
    with pytest.raises(ValueError):
        SimpleHilbertSpace(states=-1)
    with pytest.raises(ValueError):
        SimpleHilbertSpace(states=())
    with pytest.raises(ValueError):
        SimpleHilbertSpace(states=(-1, 0, 1))
    with pytest.raises(ValueError):
        SimpleHilbertSpace(states=4, number=6)

    space = SimpleHilbertSpace(4, 2)
    assert space.single_particle_states == (0, 1, 2, 3)
    assert space.state_number == 4
    assert space.particle_number == 2

    assert repr(space) == "SimpleHilbertSpace((0, 1, 2, 3), 2)"
    assert np.all(space() == np.array([3, 5, 6, 9, 10, 12]))


def test_HilbertSpace():
    number = 4
    space = HilbertSpace(number)
    assert np.all(space() == np.arange(1 << number))

    space = HilbertSpace([4, 2])
    assert np.all(space() == np.array([3, 5, 6, 9, 10, 12]))

    space = HilbertSpace([(0, 2), 1], [(1, 3), 1])
    assert np.all(space() == np.array([3, 6, 9, 12]))

    space = HilbertSpace([(0, 1), 1], [(2, 3), 1])
    assert np.all(space() == np.array([5, 6, 9, 10]))


def test_base_vectors():
    number = 4
    kets = base_vectors(number)
    assert np.all(kets == np.arange(1 << number))

    kets = base_vectors((4, 2))
    assert np.all(kets == np.array([3, 5, 6, 9, 10, 12]))

    kets = base_vectors([(0, 2), 1], [(1, 3), 1])
    assert np.all(kets == np.array([3, 6, 9, 12]))

    kets = base_vectors([(0, 1), 1], [(2, 3), 1])
    assert np.all(kets == np.array([5, 6, 9, 10]))