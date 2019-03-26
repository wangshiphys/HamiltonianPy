"""
Python implementation of the algorithm for calculating the matrix
representation of a Ferimonic operator/term in occupation number representation
"""


from numba import jit, int64, uint64
from scipy.sparse import csr_matrix

import numpy as np

from ..constant import CREATION


__all__ = [
    "matrix_function_py",
]


# The core function for calculating the matrix representation
# For the case that the right- and left-bases are not the same
@jit(int64[:, :](uint64[:, :], uint64[:], uint64[:]), nopython=True, cache=True)
def matrix_repr_py_api_nonsymmetric(term, right_bases, left_bases):
    term_length = term.shape[0]
    right_dim = right_bases.shape[0]
    left_dim = left_bases.shape[0]

    res = np.zeros((3, right_dim), dtype=np.int64)
    for i in range(right_dim):
        swap = 0
        ket = right_bases[i]
        for j in range(term_length - 1, -1, -1):
            state_index = term[j, 0]
            mask = term[j, 1]
            criterion = term[j, 2]
            # Whether the jth Ferionmic operator acts on the `ket` results zero
            # If not zero, count the number of swap (odd or even) for
            # performing this action; After this action, a new ket is produced
            if (ket & mask) == criterion:
                for pos in range(state_index):
                    if ket & (1 << pos):
                        swap ^= 1
                ket ^= mask
            else:
                break
        # The following  `else` clause belongs to the `for` loop, not the `if`
        # statement, It is executed when the loop terminates through
        # exhaustion of the list (with `for`) or when the condition becomes
        # false (when `while`), but not when the loop is terminated by a
        # `break` statement
        else:
            index = np.searchsorted(left_bases, ket)
            if index != left_dim and left_bases[index] == ket:
                res[0, i] = index
                res[1, i] = i
                res[2, i] = -1 if swap else 1
            else:
                raise KeyError(
                    "The generated ket does not belong the left_bases"
                )
    return res


# The core function for calculating the matrix representation
# For the case that the right- and left-bases are the same
@jit(int64[:, :](uint64[:, :], uint64[:]), nopython=True, cache=True)
def matrix_repr_py_api_symmetric(term, right_bases):
    term_length = term.shape[0]
    right_dim = right_bases.shape[0]

    res = np.zeros((3, right_dim), dtype=np.int64)
    for i in range(right_dim):
        swap = 0
        ket = right_bases[i]
        for j in range(term_length - 1, -1, -1):
            state_index = term[j, 0]
            mask = term[j, 1]
            criterion = term[j, 2]
            if (ket & mask) == criterion:
                for pos in range(state_index):
                    if ket & (1 << pos):
                        swap ^= 1
                ket ^= mask
            else:
                break
        # The following  `else` clause belongs to the `for` loop, not the `if`
        # statement, It is executed when the loop terminates through
        # exhaustion of the list (with `for`) or when the condition becomes
        # false (when `while`), but not when the loop is terminated by a
        # `break` statement
        else:
            index = np.searchsorted(right_bases, ket)
            if index != right_dim and right_bases[index] == ket:
                res[0, i] = index
                res[1, i] = i
                res[2, i] = -1 if swap else 1
            else:
                raise KeyError(
                    "The generated ket does not belong the left_bases"
                )
    return res


def matrix_function_py(
        term, right_bases, *, left_bases=None, coeff=1.0, to_csr=True
):
    """
    Return the matrix representation of the given term

    Parameters
    ----------
    term : list or tuple
        A sequence of length 2 tuples or lists:
            [(index_0, otype_0), ..., (index_n ,otype_n)]
        `index` is the index of the single-particle state;
        `otype` is the type of the operator which can be either CREATION(1)
        or ANNIHILATION(0).
    right_bases : 1D array
        The bases of the Hilbert space before the operation
    left_bases : 1D array, keyword-only, optional
        The bases of the Hilbert space after the operation
        It not given or None, left_bases is the same as right_bases.
        default: None
    coeff : float or complex, keyword-only, optional
        The coefficient of this term
        default: 1.0
    to_csr : boolean, keyword-only, optional
        Whether to construct a csr_matrix as the result
        default: True

    Returns
    -------
    res : csr_matrix or tuple
        The matrix representation of the term in the Hilbert space
        If `to_csr` is set to True, the result is a csr_matrix;
        If set to False, the result is a tuple: (entries, (rows, cols)),
        where `entries` is the non-zero matrix elements, `rows` and
        `cols` are the row and column indices of the none-zero elements.
    """

    term = np.array(
        [
            [index, 1 << index, 0 if otype == CREATION else (1 << index)]
            for index, otype in term
        ], dtype=np.uint64
    )
    if left_bases is None:
        shape = (right_bases.shape[0], right_bases.shape[0])
        res = matrix_repr_py_api_symmetric(term, right_bases)
    else:
        shape = (left_bases.shape[0], right_bases.shape[0])
        res = matrix_repr_py_api_nonsymmetric(term, right_bases, left_bases)

    res = (coeff * res[2], (res[0], res[1]))
    if to_csr:
        res = csr_matrix(res, shape=shape)
        res.eliminate_zeros()
    return res
