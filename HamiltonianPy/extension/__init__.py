"""
Provide method to calculate the matrix representation of a specific
Hamiltonian term
"""


__all__ = [
    "matrix_function",
]


from .matrix_repr_c_mod import matrix_repr_c_api
from scipy.sparse import csr_matrix

from ..constant import CREATION


def matrix_function(term, right_bases, *, left_bases=None, to_csr=True):
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
    right_bases : tuple
        The bases of the Hilbert space before the operation
    left_bases : tuple, keyword-only, optional
        The bases of the Hilbert space after the operation
        It not given or None, left_bases is the same as right_bases.
        default: None
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

    assert isinstance(term, (tuple, list))
    assert isinstance(right_bases, tuple)
    assert isinstance(left_bases, tuple) or (left_bases is None)

    buff = []
    for index, otype in term:
        mask = 1 << index
        criterion = 0 if otype == CREATION else mask
        buff.append(index)
        buff.append(mask)
        buff.append(criterion)
    buff = tuple(buff)

    right_dim = len(right_bases)
    if left_bases is None:
        shape = (right_dim, right_dim)
        res = matrix_repr_c_api(buff, right_bases)
    else:
        shape = (len(left_bases), right_dim)
        res = matrix_repr_c_api(buff, right_bases, left_bases)

    if to_csr:
        res = csr_matrix(res, shape=shape)
    return res
