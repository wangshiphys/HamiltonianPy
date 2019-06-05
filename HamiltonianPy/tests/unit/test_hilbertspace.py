"""
A test script for the hilbertspace module
"""


import numpy as np
import pytest

from HamiltonianPy.hilbertspace import is_valid_states_collection
from HamiltonianPy.hilbertspace import is_valid_subspace_specifier
from HamiltonianPy.hilbertspace import subspace_specifier_preprocess
from HamiltonianPy.hilbertspace import SimpleHilbertSpace
from HamiltonianPy.hilbertspace import HilbertSpace
from HamiltonianPy.hilbertspace import base_vectors


def test_is_valid_states_collection():
    assert not is_valid_states_collection(5)
    assert not is_valid_states_collection("abc")
    assert not is_valid_states_collection(list())
    assert not is_valid_states_collection(tuple())
    assert not is_valid_states_collection(set())
    assert not is_valid_states_collection([0, 1, "2"])
    assert not is_valid_states_collection([0, 1, -2])

    assert is_valid_states_collection([0, 1, 2, 3])

def test_is_valid_subspace_specifier():
    assert is_valid_subspace_specifier(4)
    assert not is_valid_subspace_specifier(-4)
    assert not is_valid_subspace_specifier("4")

    assert is_valid_subspace_specifier([4, 2])
    assert not is_valid_subspace_specifier([4, 2, 1])
    assert not is_valid_subspace_specifier([-4, 2])
    assert not is_valid_subspace_specifier([4, 5])

    assert is_valid_subspace_specifier([(0, 1, 2, 3), 2])
    assert not is_valid_subspace_specifier([(0, 1, 2, 3), 2, 1])
    assert not is_valid_subspace_specifier([(0, 1, 2, 3), 5])

def test_subspace_specifier_preprocess():
    assert subspace_specifier_preprocess(4) == ((0, 1, 2, 3), -1)
    with pytest.raises(ValueError, match="Invalid subspace-specifier"):
        subspace_specifier_preprocess(-4)
    with pytest.raises(ValueError, match="Invalid subspace-specifier"):
        subspace_specifier_preprocess("4")

    assert subspace_specifier_preprocess([4, 2]) == ((0, 1, 2, 3), 2)
    with pytest.raises(ValueError, match="Invalid subspace-specifier"):
        subspace_specifier_preprocess([4, 2, 1])
    with pytest.raises(ValueError, match="Invalid subspace-specifier"):
        subspace_specifier_preprocess([-4, 2])
    with pytest.raises(ValueError, match="Invalid subspace-specifier"):
        subspace_specifier_preprocess([4, 5])

    assert subspace_specifier_preprocess([(0, 1, 2, 3), 2]) == ((0, 1, 2, 3), 2)
    with pytest.raises(ValueError, match="Invalid subspace-specifier"):
        subspace_specifier_preprocess([(0, 1, 2, 3), 2, 1])
    with pytest.raises(ValueError, match="Invalid subspace-specifier"):
        subspace_specifier_preprocess([(0, 1, 2, 3), 5])


class TestSimpleHilbertSpace:
    def test_init(self):
        match0 = r"`states` parameter .* positive integer .*"
        match1 = r"`states` parameter .* collection of non-negative integers!"
        match2 = r"`number` parameter .* integer"
        with pytest.raises(ValueError, match=match0):
            SimpleHilbertSpace(states="a")
        with pytest.raises(ValueError, match=match0):
            SimpleHilbertSpace(states=0)
        with pytest.raises(ValueError, match=match1):
            SimpleHilbertSpace(states=[0, 1, -2])
        with pytest.raises(ValueError, match=match2):
            SimpleHilbertSpace(4, 5)

        space = SimpleHilbertSpace(4, 2)
        assert space.single_particle_states == (0, 1, 2, 3)
        assert space.state_number == 4
        assert space.particle_number == 2

    def test_repr_and_str(self):
        space = SimpleHilbertSpace(2)
        print(space)
        print("*" * 80)
        assert repr(space) == "SimpleHilbertSpace((0, 1), -1)"

        space = SimpleHilbertSpace(4, 2)
        print(space)
        print("*" * 80)
        assert repr(space) == "SimpleHilbertSpace((0, 1, 2, 3), 2)"

    def test_base_vectors(self):
        space = SimpleHilbertSpace(2)
        assert space.base_vectors(container="list") == [0, 1, 2, 3]

        space = SimpleHilbertSpace(4, 2)
        assert np.all(space() == [3, 5, 6, 9, 10, 12])


class TestHilbertSpace:
    def test_repr_and_str(self):
        separator = "*" * 80

        space = HilbertSpace(2)
        print(space)
        print(separator)
        assert repr(space) == "HilbertSpace(((0, 1), -1))"

        space = HilbertSpace([4, 2])
        print(space)
        print(separator)
        assert repr(space) == "HilbertSpace(((0, 1, 2, 3), 2))"

        space = HilbertSpace([(0, 2), 1], [(1, 3), 1])
        print(space)
        print(separator)
        assert repr(space) == "HilbertSpace(((0, 2), 1), ((1, 3), 1))"

        space = HilbertSpace([(0, 1), 1], [(2, 3), 1])
        print(space)
        print(separator)
        assert repr(space) == "HilbertSpace(((0, 1), 1), ((2, 3), 1))"

    def test_base_vectors(self):
        space = HilbertSpace(2)
        assert np.all(space() == [0, 1, 2, 3])

        space = HilbertSpace([4, 2])
        assert np.all(space() == [3, 5, 6, 9, 10, 12])

        space = HilbertSpace([(0, 2), 1], [(1, 3), 1])
        assert np.all(space() == [3, 6, 9, 12])

        space = HilbertSpace([(0, 1), 1], [(2, 3), 1])
        assert np.all(space() == [5, 6, 9, 10])


def test_base_vectors():
    kets = base_vectors(2)
    assert np.all(kets == [0, 1, 2, 3])

    kets = base_vectors((4, 2))
    assert np.all(kets == [3, 5, 6, 9, 10, 12])

    kets = base_vectors([(0, 2), 1], [(1, 3), 1], container="tuple")
    assert kets == (3, 6, 9, 12)

    kets = base_vectors([(0, 1), 1], [(2, 3), 1], container="list")
    assert kets == [5, 6, 9, 10]
