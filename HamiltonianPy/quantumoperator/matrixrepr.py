"""
Implementation of the algorithm for calculating the matrix representation of
a Fermionic operator/term in occupation number representation
"""


import numpy as np
from numba import jit, int64, uint64
from scipy.sparse import csr_matrix

from HamiltonianPy.quantumoperator.constant import CREATION

__all__ = [
    "matrix_function",
]


# The core function for calculating the matrix representation of a general term
@jit(int64[:, :](uint64[:, :], uint64[:], uint64[:]), nopython=True, cache=True)
def _core_general(term, right_bases, left_bases):
    term_length = term.shape[0]
    left_dim = left_bases.shape[0]
    right_dim = right_bases.shape[0]

    result = np.zeros((3, right_dim), dtype=np.int64)
    for i in range(right_dim):
        swap = 0
        ket = right_bases[i]
        for j in range(term_length - 1, -1, -1):
            state_index = term[j, 0]
            mask = term[j, 1]
            criterion = term[j, 2]
            # Whether the jth Fermionic operator acts on the `ket` results zero
            # If not zero, count the number of swap (odd or even) for
            # performing this action; After this action, a new ket is produced
            if (ket & mask) == criterion:
                for pos in range(state_index):
                    if ket & (1 << pos):
                        swap ^= 1
                ket ^= mask
            else:
                break
        # The following `else` clause belongs to the
        # `for j in range(term_length - 1, -1, -1)` loop, not the
        # `if (ket & mask) == criterion` statement. It is executed when the
        # loop terminates through exhaustion of the iterator, but not when
        # the loop is terminated by the `break` statement.
        else:
            index = np.searchsorted(left_bases, ket)
            if index != left_dim and left_bases[index] == ket:
                result[0, i] = index
                result[1, i] = i
                result[2, i] = -1 if swap else 1
            else:
                raise KeyError(
                    "The generated ket does not belong the left_bases"
                )
    return result


# The core function for calculating the matrix representation of the hopping
# term: '$c_i^{\\dagger} c_j$', `i != j` is assumed.
# The `term` parameter is assumed of the following form:
# np.array(
#     [[index0, 1<<index0, 0], [index1, 1<<index1, 1<<index1]], dtype=np.uint64
# )
@jit(int64[:, :](uint64[:, :], uint64[:], uint64[:]), nopython=True, cache=True)
def _core_hopping(term, right_bases, left_bases):
    start = np.min(term[:, 0])
    end = np.max(term[:, 0])
    mask0 = term[0, 1]
    mask1 = term[1, 1]
    left_dim = left_bases.shape[0]
    right_dim = right_bases.shape[0]

    result = np.zeros((3, right_dim), dtype=np.int64)
    for i in range(right_dim):
        swap = 0
        ket = right_bases[i]
        if ((ket & mask0) == 0) and ((ket & mask1) == mask1):
            for pos in range(start+1, end):
                if ket & (1 << pos):
                    swap ^= 1
            ket ^= mask1
            ket ^= mask0
            index = np.searchsorted(left_bases, ket)
            if index != left_dim and left_bases[index] == ket:
                result[0, i] = index
                result[1, i] = i
                result[2, i] = -1 if swap else 1
            else:
                raise KeyError(
                    "The generated ket does not belong the left_bases"
                )
    return result


# The core function for calculating the matrix representation of the
# particle-number operator: '$c_i^{\\dagger} c_i$'
# The `term` parameter is assumed of the following form:
# np.array([[index, 1<<index, 0], [index, 1<<index, 1<<index]], dtype=np.uint64)
@jit(int64[:, :](uint64[:, :], uint64[:]), nopython=True, cache=True)
def _core_particle_number(term, bases):
    mask = term[0, 1]
    dim = bases.shape[0]
    result = np.zeros((3, dim), dtype=np.int64)
    for i in range(dim):
        if bases[i] & mask:
            result[0, i] = i
            result[1, i] = i
            result[2, i] = 1
    return result


# The core function for calculating the matrix representation of the Coulomb
# interaction term: '$n_i n_j$'
# The `term` parameter is assumed of the following form:
# np.array(
#     [
#         [index0, 1 << index0, 0],
#         [index0, 1 << index0, 1 << index0],
#         [index1, 1 << index1, 0],
#         [index1, 1 << index1, 1 << index1],
#     ], dtype=np.uint64
# )
@jit(int64[:, :](uint64[:, :], uint64[:]), nopython=True, cache=True)
def _core_coulomb(term, bases):
    mask0 = term[0, 1]
    mask1 = term[2, 1]
    dim = bases.shape[0]
    result = np.zeros((3, dim), dtype=np.int64)
    for i in range(dim):
        ket = bases[i]
        if (ket & mask0) and (ket & mask1):
            result[0, i] = i
            result[1, i] = i
            result[2, i] = 1
    return result


def matrix_function(
        term, right_bases, *, left_bases=None, coeff=1.0, to_csr=True,
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
    right_bases : 1D np.ndarray
        The bases of the Hilbert space before the operation.
        The data-type of the array's elements is np.uint64.
    left_bases : 1D np.ndarray, keyword-only, optional
        The bases of the Hilbert space after the operation.
        If given, the data-type of the array's elements is np.uint64.
        It not given or None, left_bases is the same as right_bases.
        default: None
    coeff : int, float or complex, keyword-only, optional
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

    Note
    ----
    This function does not perform any check on the input arguments, so the
    caller must guarantee the arguments passed to this function fulfill the
    requirements listed above. It is recommended to call the `matrix_repr`
    method of `AoC`, `NumberOperator` and `ParticleTerm` class defined in
    `particlesystem` module instead of calling this function directly.
    """

    # `bitwise-and` between ket and (1<<ith) to judge whether the ith bit is 0
    # or 1, (1<<ith) is called `mask`;
    # If the operator is a creation operator, then the ith bit must be 0 to
    # generate nonzero result. The criterion is (ket & mask) == 0;
    # If the operator is an annihilation operator, then the ith bit must be 1
    # to generate nonzero result. The criterion is (ket & mask) == mask.
    # The term variable on the left-hand side has shape (N, 3), every row
    # correspond to an operator.
    term = np.array(
        [
            # [index, mask, criterion]
            [index, 1 << index, 0 if otype == CREATION else (1 << index)]
            for index, otype in term
        ], dtype=np.uint64
    )

    right_dim = right_bases.shape[0]
    if left_bases is None:
        left_bases = right_bases
        shape = (right_dim, right_dim)
    else:
        shape = (left_bases.shape[0], right_dim)
    data = _core_general(term, right_bases, left_bases)

    if to_csr:
        res = csr_matrix((coeff * data[2], (data[0], data[1])), shape=shape)
        res.eliminate_zeros()
    else:
        tmp = data[:, data[2] != 0]
        res = (coeff * tmp[2], (tmp[0], tmp[1]))
    return res
