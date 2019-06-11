"""
A test script for the matrixrepr module in quantumoperator subpackage.
"""


import numpy as np
import pytest

from HamiltonianPy.quantumoperator.matrixrepr import matrix_function


@pytest.fixture(scope="module")
def bases():
    return np.arange(16, dtype=np.uint64)

def matrix_function_wrapper(term, bases, special_tag="general"):
    entries, (row_indices, col_indices) = matrix_function(
        term, bases, to_csr=False, coeff=1, special_tag=special_tag
    )
    return sorted(zip(row_indices, col_indices, entries))


class TestMatrixFunctionGeneral:
    def test_annihilator(self, bases):
        # $c_0$
        term = [(0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            (0, 1, 1), ( 2,  3, 1), ( 4,  5, 1), ( 6,  7, 1),
            (8, 9, 1), (10, 11, 1), (12, 13, 1), (14, 15, 1),
        ]

        # $c_1$
        term = [(1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            (0,  2, 1), (1,  3, -1), ( 4,  6, 1), ( 5,  7, -1),
            (8, 10, 1), (9, 11, -1), (12, 14, 1), (13, 15, -1),
        ]

        # $c_2$
        term = [(2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            (0,  4, 1), (1,  5, -1), ( 2,  6, -1), ( 3,  7, 1),
            (8, 12, 1), (9, 13, -1), (10, 14, -1), (11, 15, 1),
        ]

        # $c_3$
        term = [(3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            (0,  8,  1), (1,  9, -1), (2, 10, -1), (3, 11,  1),
            (4, 12, -1), (5, 13,  1), (6, 14,  1), (7, 15, -1),
        ]

    def test_creation(self, bases):
        # $c_0^{\\dagger}$
        term = [(0, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            (1, 0, 1), ( 3,  2, 1), ( 5,  4, 1), ( 7,  6, 1),
            (9, 8, 1), (11, 10, 1), (13, 12, 1), (15, 14, 1),
        ]

        # $c_1^{\\dagger}$
        term = [(1, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            ( 2, 0, 1), ( 3, 1, -1), ( 6,  4, 1), ( 7,  5, -1),
            (10, 8, 1), (11, 9, -1), (14, 12, 1), (15, 13, -1),
        ]

        # $c_2^{\\dagger}$
        term = [(2, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            ( 4, 0, 1), ( 5, 1, -1), ( 6,  2, -1), ( 7,  3, 1),
            (12, 8, 1), (13, 9, -1), (14, 10, -1), (15, 11, 1),
        ]

        # $c_3^{\\dagger}$
        term = [(3, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            ( 8, 0,  1), ( 9, 1, -1), (10, 2, -1), (11, 3,  1),
            (12, 4, -1), (13, 5,  1), (14, 6,  1), (15, 7, -1),
        ]

    def test_hopping(self, bases):
        # $c_0^{\\dagger} c_0$
        term = [(0, 1), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            (1, 1, 1), ( 3,  3, 1), ( 5,  5, 1), ( 7,  7, 1),
            (9, 9, 1), (11, 11, 1), (13, 13, 1), (15, 15, 1),
        ]

        # $c_0^{\\dagger} c_1$
        term = [(0, 1), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(1, 2, 1), (5, 6, 1), (9, 10, 1), (13, 14, 1)]

        # $c_1^{\\dagger} c_0$
        term = [(1, 1), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(2, 1, 1), (6, 5, 1), (10, 9, 1), (14, 13, 1)]

        # $c_0^{\\dagger} c_2$
        term = [(0, 1), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(1, 4, 1), (3, 6, -1), (9, 12, 1), (11, 14, -1)]

        # $c_2^{\\dagger} c_0$
        term = [(2, 1), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(4, 1, 1), (6, 3, -1), (12, 9, 1), (14, 11, -1)]

        # $c_0^{\\dagger} c_3$
        term = [(0, 1), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(1, 8, 1), (3, 10, -1), (5, 12, -1), (7, 14, 1)]

        # $c_3^{\\dagger} c_0$
        term = [(3, 1), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(8, 1, 1), (10, 3, -1), (12, 5, -1), (14, 7, 1)]

        # $c_1^{\\dagger} c_1$
        term = [(1, 1), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            ( 2,  2, 1), ( 3,  3, 1), ( 6,  6, 1), ( 7,  7, 1),
            (10, 10, 1), (11, 11, 1), (14, 14, 1), (15, 15, 1),
        ]

        # $c_1^{\\dagger} c_2$
        term = [(1, 1), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(2, 4, 1), (3, 5, 1), (10, 12, 1), (11, 13, 1)]

        # $c_2^{\\dagger} c_1$
        term = [(2, 1), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(4, 2, 1), (5, 3, 1), (12, 10, 1), (13, 11, 1)]

        # $c_1^{\\dagger} c_3$
        term = [(1, 1), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(2, 8, 1), (3, 9, 1), (6, 12, -1), (7, 13, -1)]

        # $c_3^{\\dagger} c_1$
        term = [(3, 1), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(8, 2, 1), (9, 3, 1), (12, 6, -1), (13, 7, -1)]

        # $c_2^{\\dagger} c_2$
        term = [(2, 1), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            ( 4,  4, 1), ( 5,  5, 1), ( 6,  6, 1), ( 7,  7, 1),
            (12, 12, 1), (13, 13, 1), (14, 14, 1), (15, 15, 1),
        ]

        # $c_2^{\\dagger} c_3$
        term = [(2, 1), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(4, 8, 1), (5, 9, 1), (6, 10, 1), (7, 11, 1)]

        # $c_3^{\\dagger} c_2$
        term = [(3, 1), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(8, 4, 1), (9, 5, 1), (10, 6, 1), (11, 7, 1)]

        # $c_3^{\\dagger} c_3$
        term = [(3, 1), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            ( 8,  8, 1), ( 9,  9, 1), (10, 10, 1), (11, 11, 1),
            (12, 12, 1), (13, 13, 1), (14, 14, 1), (15, 15, 1),
        ]

    def test_hubbard(self, bases):
        # $n_0 n_0$
        term = [(0, 1), (0, 0), (0, 1), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            (1, 1, 1), ( 3,  3, 1), ( 5,  5, 1), ( 7,  7, 1),
            (9, 9, 1), (11, 11, 1), (13, 13, 1), (15, 15, 1),
        ]

        # $n_0 n_1$
        term = [(0, 1), (0, 0), (1, 1), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(3, 3, 1), (7, 7, 1), (11, 11, 1), (15, 15, 1)]

        # $n_1 n_0$
        term = [(1, 1), (1, 0), (0, 1), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(3, 3, 1), (7, 7, 1), (11, 11, 1), (15, 15, 1)]

        # $n_0 n_2$
        term = [(0, 1), (0, 0), (2, 1), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(5, 5, 1), (7, 7, 1), (13, 13, 1), (15, 15, 1)]

        # $n_2 n_0$
        term = [(2, 1), (2, 0), (0, 1), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(5, 5, 1), (7, 7, 1), (13, 13, 1), (15, 15, 1)]

        # $n_0 n_3$
        term = [(0, 1), (0, 0), (3, 1), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(9, 9, 1), (11, 11, 1), (13, 13, 1), (15, 15, 1)]

        # $n_3 n_0$
        term = [(3, 1), (3, 0), (0, 1), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(9, 9, 1), (11, 11, 1), (13, 13, 1), (15, 15, 1)]

        # $n_1 n_1$
        term = [(1, 1), (1, 0), (1, 1), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            ( 2,  2, 1), ( 3,  3, 1), ( 6,  6, 1), ( 7,  7, 1),
            (10, 10, 1), (11, 11, 1), (14, 14, 1), (15, 15, 1),
        ]

        # $n_1 n_2$
        term = [(1, 1), (1, 0), (2, 1), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(6, 6, 1), (7, 7, 1), (14, 14, 1), (15, 15, 1)]

        # $n_2 n_1$
        term = [(2, 1), (2, 0), (1, 1), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(6, 6, 1), (7, 7, 1), (14, 14, 1), (15, 15, 1)]

        # $n_1 n_3$
        term = [(1, 1), (1, 0), (3, 1), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(10, 10, 1), (11, 11, 1), (14, 14, 1), (15, 15, 1)]

        # $n_3 n_1$
        term = [(3, 1), (3, 0), (1, 1), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(10, 10, 1), (11, 11, 1), (14, 14, 1), (15, 15, 1)]

        # $n_2 n_2$
        term = [(2, 1), (2, 0), (2, 1), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            ( 4,  4, 1), ( 5,  5, 1), ( 6,  6, 1), ( 7,  7, 1),
            (12, 12, 1), (13, 13, 1), (14, 14, 1), (15, 15, 1),
        ]

        # $n_2 n_3$
        term = [(2, 1), (2, 0), (3, 1), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(12, 12, 1), (13, 13, 1), (14, 14, 1), (15, 15, 1)]

        # $n_3 n_2$
        term = [(3, 1), (3, 0), (2, 1), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(12, 12, 1), (13, 13, 1), (14, 14, 1), (15, 15, 1)]

        # $n_3 n_3$
        term = [(3, 1), (3, 0), (3, 1), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [
            ( 8,  8, 1), ( 9,  9, 1), (10, 10, 1), (11, 11, 1),
            (12, 12, 1), (13, 13, 1), (14, 14, 1), (15, 15, 1),
        ]

    def test_hole_pairing(self, bases):
        # $c_0 c_1$
        term = [(0, 0), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 3, -1), (4, 7, -1), (8, 11, -1), (12, 15, -1)]

        # $c_1 c_0$
        term = [(1, 0), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 3, 1), (4, 7, 1), (8, 11, 1), (12, 15, 1)]

        # $c_0 c_2$
        term = [(0, 0), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 5, -1), (2, 7, 1), (8, 13, -1), (10, 15, 1)]

        # $c_2 c_0$
        term = [(2, 0), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 5, 1), (2, 7, -1), (8, 13, 1), (10, 15, -1)]

        # $c_0 c_3$
        term = [(0, 0), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 9, -1), (2, 11, 1), (4, 13, 1), (6, 15, -1)]

        # $c_3 c_0$
        term = [(3, 0), (0, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 9, 1), (2, 11, -1), (4, 13, -1), (6, 15, 1)]

        # $c_1 c_2$
        term = [(1, 0), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 6, -1), (1, 7, -1), (8, 14, -1), (9, 15, -1)]

        # $c_2 c_1$
        term = [(2, 0), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 6, 1), (1, 7, 1), (8, 14, 1), (9, 15, 1)]

        # $c_1 c_3$
        term = [(1, 0), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 10, -1), (1, 11, -1), (4, 14, 1), (5, 15, 1)]

        # $c_3 c_1$
        term = [(3, 0), (1, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 10, 1), (1, 11, 1), (4, 14, -1), (5, 15, -1)]

        # $c_2 c_3$
        term = [(2, 0), (3, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 12, -1), (1, 13, -1), (2, 14, -1), (3, 15, -1)]

        # $c_3 c_2$
        term = [(3, 0), (2, 0)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(0, 12, 1), (1, 13, 1), (2, 14, 1), (3, 15, 1)]

    def test_particle_pairing(self, bases):
        # $c_0^{\\dagger} c_1^{\\dagger}$
        term = [(0, 1), (1, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(3, 0, 1), (7, 4, 1), (11, 8, 1), (15, 12, 1)]

        # $c_1^{\\dagger} c_0^{\\dagger}$
        term = [(1, 1), (0, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(3, 0, -1), (7, 4, -1), (11, 8, -1), (15, 12, -1)]

        # $c_0^{\\dagger} c_2^{\\dagger}$
        term = [(0, 1), (2, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(5, 0, 1), (7, 2, -1), (13, 8, 1), (15, 10, -1)]

        # $c_2^{\\dagger} c_0^{\\dagger}$
        term = [(2, 1), (0, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(5, 0, -1), (7, 2, 1), (13, 8, -1), (15, 10, 1)]

        # $c_0^{\\dagger} c_3^{\\dagger}$
        term = [(0, 1), (3, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(9, 0, 1), (11, 2, -1), (13, 4, -1), (15, 6, 1)]

        # $c_3^{\\dagger} c_0^{\\dagger}$
        term = [(3, 1), (0, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(9, 0, -1), (11, 2, 1), (13, 4, 1), (15, 6, -1)]

        # $c_1^{\\dagger} c_2^{\\dagger}$
        term = [(1, 1), (2, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(6, 0, 1), (7, 1, 1), (14, 8, 1), (15, 9, 1)]

        # $c_2^{\\dagger} c_1^{\\dagger}$
        term = [(2, 1), (1, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(6, 0, -1), (7, 1, -1), (14, 8, -1), (15, 9, -1)]

        # $c_1^{\\dagger} c_3^{\\dagger}$
        term = [(1, 1), (3, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(10, 0, 1), (11, 1, 1), (14, 4, -1), (15, 5, -1)]

        # $c_3^{\\dagger} c_1^{\\dagger}$
        term = [(3, 1), (1, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(10, 0, -1), (11, 1, -1), (14, 4, 1), (15, 5, 1)]

        # $c_2^{\\dagger} c_3^{\\dagger}$
        term = [(2, 1), (3, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(12, 0, 1), (13, 1, 1), (14, 2, 1), (15, 3, 1)]

        # $c_3^{\\dagger} c_2^{\\dagger}$
        term = [(3, 1), (2, 1)]
        res = matrix_function_wrapper(term, bases)
        assert res == [(12, 0, -1), (13, 1, -1), (14, 2, -1), (15, 3, -1)]


class TestMatrixFunctionSpecial:
    def test_hopping(self, bases):
        for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            for term in [[(i, 1), (j , 0)], [(j, 1), (i, 0)]]:
                general = matrix_function_wrapper(term, bases)
                special = matrix_function_wrapper(term, bases, "hopping")
                assert general == special

    def test_particle_number(self, bases):
        for i in range(4):
            term = [(i, 1), (i, 0)]
            general = matrix_function_wrapper(term, bases)
            special = matrix_function_wrapper(term, bases, "number")
            assert general == special

    def test_Coulomb(self, bases):
        for i in range(4):
            for j in range(4):
                term = [(i, 1), (i, 0), (j, 1), (j, 0)]
                general = matrix_function_wrapper(term, bases)
                special = matrix_function_wrapper(term, bases, "Coulomb")
                assert general == special
