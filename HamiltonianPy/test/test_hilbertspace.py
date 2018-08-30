"""
A test script for the hilbertspace module
"""


from time import time

from HamiltonianPy.hilbertspace import *


def _pretty_print(array, length=20):
    if len(array) > length:
        half = length // 2
        print("(", end="")
        for item in array[:half]:
            print(item, end=", ")
        print("...", end=", ")
        for item in array[-half:-1]:
            print(item, end=", ")
        print(array[-1], end=")\n")
    else:
        print(array)


def _test_SimpleHilbertSpace():
    number = 4
    space = SimpleHilbertSpace(number)
    print(
        "The single_particle_states attribute:\n    {}".format(
            space.single_particle_states
        )
    )
    print("The state_number attribute: {}".format(space.state_number))
    print("The particle_number attribute: {}".format(space.particle_number))
    print("repr:\n    {}".format(repr(space)))
    print("str:\n    {}".format(str(space)))
    print("=" * 80)

    for number in range(1, 21):
        t0 = time()
        space = SimpleHilbertSpace(number)
        kets = space()
        t1 = time()
        assert (kets == tuple(range(1<<number)))
        print(space)
        print("The time spend on generating base vectors: {}".format(t1 - t0))
        print("Base vectors:")
        _pretty_print(kets)
        print("=" * 80)

    space = SimpleHilbertSpace(4, 2)
    kets = space()
    assert kets == (3, 5, 6, 9, 10, 12)


def _test_HilbertSpace():
    number = 4
    space = HilbertSpace(number)
    kets = space()
    print(repr(space))
    print(space)
    print("=" * 80)
    assert kets == tuple(range(1<<number))

    space = HilbertSpace([4, 2])
    kets = space()
    print(repr(space))
    print(space)
    print("=" * 80)
    assert kets == (3, 5, 6, 9, 10, 12)

    space = HilbertSpace([(0, 2), 1], [(1, 3), 1])
    kets = space()
    print(repr(space))
    print(space)
    print("=" * 80)
    assert kets == (3, 6, 9, 12)

    space = HilbertSpace([(0, 1), 1], [(2, 3), 1])
    kets = space()
    print(repr(space))
    print(space)
    print("=" * 80)
    assert kets == (5, 6, 9, 10)


def _test_base_vectors():
    number = 4
    kets = base_vectors(number)
    assert kets == tuple(range(1<<number))

    kets = base_vectors([4, 2])
    assert kets == (3, 5, 6, 9, 10, 12)

    kets = base_vectors([(0, 2), 1], [(1, 3), 1])
    assert kets == (3, 6, 9, 12)

    kets = base_vectors([(0, 1), 1], [(2, 3), 1])
    assert kets == (5, 6, 9, 10)


if __name__ == "__main__":
    _test_SimpleHilbertSpace()
    _test_HilbertSpace()
    _test_base_vectors()