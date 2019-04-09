"""
Implementation of the algorithm for calculating the matrix representation of
a Fermionic operator/term in occupation number representation
"""


from threading import Thread

from numba import jit, int64, uint64, void
from scipy.sparse import csr_matrix

import numpy as np

from HamiltonianPy.constant import CREATION


__all__ = [
    "matrix_function",
]


# The core function for calculating the matrix representation
# For the case that the right- and left-bases are different
@jit(
    void(int64, int64, int64[:, :], uint64[:, :], uint64[:], uint64[:]),
    nopython=True, cache=True, nogil=True,
)
def _matrix_repr_core_nonsymmetric(
        low, high, result, term, right_bases, left_bases
):
    term_length = term.shape[0]
    left_dim = left_bases.shape[0]

    for i in range(low, high):
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


# The core function for calculating the matrix representation
# For the case that the right- and left-bases are the same
@jit(
    void(int64, int64, int64[:, :], uint64[:, :], uint64[:]),
    nopython=True, cache=True, nogil=True,
)
def _matrix_repr_core_symmetric(low, high, result, term, right_bases):
    term_length = term.shape[0]
    right_dim = right_bases.shape[0]

    for i in range(low, high):
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
        else:
            index = np.searchsorted(right_bases, ket)
            if index != right_dim and right_bases[index] == ket:
                result[0, i] = index
                result[1, i] = i
                result[2, i] = -1 if swap else 1
            else:
                raise KeyError(
                    "The generated ket does not belong the left_bases"
                )


def matrix_function(
        term, right_bases, *,
        left_bases=None, coeff=1.0, to_csr=True, threads_num=1
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
    threads_num : int, keyword-only, optional
        The number of threads to use
        default: 1

    Returns
    -------
    res : csr_matrix or tuple
        The matrix representation of the term in the Hilbert space
        If `to_csr` is set to True, the result is a csr_matrix;
        If set to False, the result is a tuple: (entries, (rows, cols)),
        where `entries` is the non-zero matrix elements, `rows` and
        `cols` are the row and column indices of the none-zero elements.
    """

    assert isinstance(threads_num, int) and threads_num > 0

    # `Bitwise and` between ket and (1<<ith) to judge whether the ith bit is 0
    # or 1, (1<<ith) is called `mask`;
    # If the operator is a creation operator, then the ith bit must be 0 to
    # generate nonzero result. The criterion is (ket & mask) == 0;
    # If the operator is an annihilation operator, then the ith bit must be 1
    # to generate nonzero result. The criterion is (ket & mask) == mask.
    term = np.array(
        [
            # [index, mask, criterion]
            [index, 1 << index, 0 if otype == CREATION else (1 << index)]
            for index, otype in term
        ], dtype=np.uint64
    )

    right_dim = right_bases.shape[0]
    result = np.zeros((3, right_dim), dtype=np.int64)

    if left_bases is None:
        core_function = _matrix_repr_core_symmetric
        extra_args = (result, term, right_bases)
        shape = (right_dim, right_dim)
    else:
        core_function = _matrix_repr_core_nonsymmetric
        extra_args = (result, term, right_bases, left_bases)
        shape = (left_bases.shape[0], right_dim)

    if threads_num == 1:
        core_function(0, right_dim, *extra_args)
    else:
        endpoints = np.linspace(
            0, right_dim, num=threads_num+1, endpoint=True, dtype=np.int64
        )
        chunks = [
            (endpoints[i], endpoints[i+1]) + extra_args
            for i in range(threads_num)
        ]
        threads = [
            Thread(target=core_function, args=chunk) for chunk in chunks
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    result = (coeff * result[2], (result[0], result[1]))
    if to_csr:
        result = csr_matrix(result, shape=shape)
        result.eliminate_zeros()
    return result


if __name__ == "__main__":
    from time import time
    import os

    num = 24
    bases_list = list(range(1 << num))
    right_bases = np.array(bases_list, dtype=np.uint64)
    left_bases = np.array(bases_list, dtype=np.uint64)
    core_num = int(os.environ["NUMBER_OF_PROCESSORS"])

    msg = "The time spend on the {0}th operator when using {1}-threads: {2}s"
    for ith in range(num):
        Ms = []
        term = [(ith, 1), (ith, 0)]
        for threads_num in range(1, core_num + 1):
            t0 = time()
            M = matrix_function(
                term, right_bases,
                left_bases=left_bases,
                threads_num=threads_num
            )
            t1 = time()
            Ms.append(M)
            print(msg.format(ith, threads_num, t1 - t0))
        print("=" * 80)

        for i in range(1, len(Ms)):
            assert (Ms[0] - Ms[i]).count_nonzero() == 0
