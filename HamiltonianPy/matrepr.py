"""
The module provide functions to calculate the matrix representation of creation
and annihilation operator as well as Hamiltonian term(consist of creation and
annihilation operators) in occupation number representation.
"""

from scipy.sparse import csr_matrix, identity

import numpy as np

from HamiltonianPy.extpkg import matreprcext as mrc
from HamiltonianPy.constant import CREATION, ANNIHILATION, VIEW_AS_ZERO


__all__ = ['termmatrix', 'aocmatrix']


def hopping_F(term, base):# {{{
    """
    Calculate the matrix representation of the hopping term in the Hilbert
    space specified by the base parameter!

    This function applys to system which obey Fermi statistics. For function
    applys to Bose system, see function hopping_B below.

    Parameter:
    ----------
    term: tuple
        This parameter should be of this form ((cindex, aindex), coeff). The
        cindex and aindex represent the indices of the states to create and
        annihilate a particle respectively and the coeff is the coefficience of
        this hopping term.
    base: tuple or list
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the hopping term.
    """

    (cindex, aindex), coeff = term
    if coeff <= VIEW_AS_ZERO:
        return 0.0

    dim = len(base)
    row, col, entries = mrc.hopping(cindex, aindex, base)
    data = (entries, (row, col))
    res = coeff * csr_matrix(data, shape=(dim, dim))
    return res
# }}}


def hubbard_F(term, base):# {{{
    """
    Calculate the matrix representation of the hubbard term in the Hilbert
    space specified by the base parameter!

    This function applys to system which obey Fermi statistics. For function
    applys to Bose system, see function hubbard_B below.

    Parameter:
    ----------
    term: tuple
        This parameter should be of this form ((index0, index1), coeff). The
        index0 and index1 represent the indices of the states of the two number
        operator and the coeff is the coefficience of this hubbard term
    base: tuple or list
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the hubbard term.
    """

    (index0, index1), coeff = term
    if coeff <= VIEW_AS_ZERO:
        return 0.0
    
    dim = len(base)
    row, col, entries = mrc.hubbard(index0, index1, base)
    data = (entries, (col, col))
    res = coeff * csr_matrix(data, shape=(dim, dim))
    return res
# }}}


def pairing_F(term, base):# {{{
    """
    Calculate the matrix representation of the pairing term in the Hilbert
    space specified by the base parameter!

    This function applys to system which obey Fermi statistics. For function
    applys to Bose system, see function pairing_B below.

    Parameter:
    ----------
    term: tuple
        This parameter should be of this form ((index0, index1), coeff, otype).
        The index0 and index1 represent the indices of the two pairing states, 
        the coeff is the coefficience of this pairing term and otype is the
        type of the pairing operator.
    base: tuple or list
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the pairing term.
    """

    (index0, index1), coeff, otype = term
    if coeff <= VIEW_AS_ZERO or index0 == index1:
        return 0.0

    dim = len(base)
    row, col, entries = mrc.pairing(index0, index1, otype, base)
    data = (entries, (row, col))
    res = coeff * csr_matrix(data, shape=(dim, dim))
    return res
# }}}


def general(term, base):# {{{
    """
    Calculate the matrix representation of a general term in the Hilbert 
    space specified by the base parameter!

    Parameter:
    ----------
    term: tuple
        A general term.
    base: tuple
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of a general term.
    """

    raise NotImplementedError("The function has not been implemented!")
# }}}


def hopping_B(term, base):# {{{
    """
    Calculate the matrix representation of the hopping term in the Hilbert
    space specified by the base parameter!

    This function applys to system which obey Bose statistics. For function
    applys to Fermi system, see function hopping_F above.

    Parameter:
    ----------
    term: tuple
        This parameter should be of this form ((cindex, aindex), coeff). The
        cindex and aindex represent the indices of the states to create and
        annihilate a particle respectively and the coeff is the coefficience of
        this hopping term.
    base: np.ndarray
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the hopping term.
    """

    (cindex, aindex), coeff = term
    if coeff <= VIEW_AS_ZERO:
        return 0.0

    dim = len(base)
    slct_p = np.bitwise_and(base, 1<<aindex) != 0
    if cindex == aindex:
        col = np.where(slct_p == True)[0]
        row = col
    else:
        slct_h = np.bitwise_and(base, 1<<cindex) == 0
        slct_ket = np.logical_and(slct_p, slct_h)
        col = np.where(slct_ket == True)[0]
        kets = base[slct_ket] + (1<<cindex) - (1<<aindex)
        row = np.searchsorted(base, kets)
        if (np.any(row >= dim)) or (np.any(base[row] != kets)):
            raise ValueError("There must be some kets not found in the base!")
 
    entries = np.ones((len(col),), dtype=np.int64)
    data = (entries, (row, col))
    res = coeff * csr_matrix(data, shape=(dim, dim))
    return res
# }}}


def hubbard_B(term, base):# {{{
    """
    Calculate the matrix representation of the hubbard term in the Hilbert
    space specified by the base parameter!

    This function applys to system which obey Bose statistics. For function
    applys to Fermi system, see function hubbard_F above.

    Parameter:
    ----------
    term: tuple
        This parameter should be of this form ((index0, index1), coeff). The
        index0 and index1 represent the indices of the states of the two number
        operator and the coeff is the coefficience of this hubbard term.
    base: np.ndarray
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the hubbard term.
    """

    (index0, index1), coeff = term
    if coeff <= VIEW_AS_ZERO:
        return 0.0

    dim = len(base)
    slct_p0 = np.bitwise_and(base, 1<<index0) != 0
    if index0 == index1:
        col = np.where(slct_p0 == True)[0]
    else:
        slct_p1 = np.bitwise_and(base, 1<<index1) != 0
        slct_ket = np.logical_and(slct_p0, slct_p1)
        col = np.where(slct_ket ==True)[0]

    entries = np.ones((len(col),), dtype=np.int64)
    data = (entries, (col, col))
    res = coeff * csr_matrix(data, shape=(dim, dim))
    return res
# }}}


def pairing_B(term, base):# {{{
    """
    Calculate the matrix representation of the pairing term in the Hilbert
    space specified by the base parameter!

    This function applys to system which obey Bose statistics. For function
    applys to Fermi system, see function pairing_F below.

    Parameter:
    ----------
    term: tuple
        This parameter should be of this form ((index0, index1), coeff, otype).
        The index0 and index1 represent the indices of the two pairing states, 
        the coeff is the coefficience of this pairing term and otype is the
        type of the pairing operator.
    base: np.ndarray
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the pairing term.
    """

    (index0, index1), coeff, otype = term
    if coeff <= VIEW_AS_ZERO or index0 == index1:
        return 0.0

    dim = len(base)
    if otype == CREATION:
        slct0 = np.bitwise_and(base, 1<<index0) == 0
        slct1 = np.bitwise_and(base, 1<<index1) == 0
        buff = 1
    else:
        slct0 = np.bitwise_and(base, 1<<index0) != 0
        slct1 = np.bitwise_and(base, 1<<index1) != 0
        buff = -1

    slct_ket = np.logical_and(slct0, slct1)
    col = np.where(slct_ket == True)[0]
    kets = base[slct_ket] + buff * ((1<<index0) + (1<<index1))
    row = np.searchsorted(base, kets)
    if (np.any(row >= dim)) or (np.any(base[row] != kets)):
        raise ValueError("There must be some kets not found in the base!")
 
    entries = np.ones((len(col),), dtype=np.int64)
    data = (entries, (row, col))
    res = coeff * csr_matrix(data, shape=(dim, dim))
    return res
# }}}


def aocmatrix_F(index, otype, rbase, lbase=None):# {{{
    """
    Calculate the matrix representation of a creation or annihilation operator
    between the Hilbert space specified by the lbase and rbase parameter!

    This function applys to system which obey Fermi statistics. For function
    applys to Bose system, see function aocmatrix_B below.

    Parameter:
    ----------
    index: int
        The index of the creation or annihilation operator.
    otype: int
        The type of the operator, 1 represents creation and 0 represents
        annihilation.
    rbase: tuple or list
        The base of the Hilbert space before the operation.
    lbase: tuple or list, optional
        The base of the Hilbert space after the operation.
        If not given or None, lbase is the same as rbase.
        default: None

    Return:
    -------
    res: csr_matrix
        The matrix representation of the operator.
    """

    if not isinstance(index, int):
        raise TypeError("The input index is not of type int!")

    if lbase is None:
        lbase = rbase

    ldim = len(lbase)
    rdim = len(rbase)

    row, col, entries = mrc.aoc(index, otype, lbase, rbase) 
    data = (entries, (row, col))
    res = csr_matrix(data, shape=(ldim, rdim))
    return res
# }}}


def aocmatrix_B(index, otype, rbase, lbase=None):# {{{
    """
    Calculate the matrix representation of a creation or annihilation operator
    between the Hilbert space specified by the lbase and rbase parameter!
  
    This function applys to system which obey Bose statistics. For function
    applys to Fermi system, see function aocmatrix_F above.

    Parameter:
    ----------
    index: int
        The index of the creation or annihilation operator.
    otype: int
        The type of the operator, 1 represents creation and 0 represents
        annihilation.
    rbase: np.ndarray
        The base of the Hilbert space before the operation.
    lbase: np.ndarray, optional
        The base of the Hilbert space after the operation.
        If not given or None, lbase is the same as rbase.
        default: None

    Return:
    -------
    res: csr_matrix
        The matrix representation of the operator.
    """

    if not isinstance(index, int):
        raise TypeError("The input index is not of type int!")

    if lbase is None:
        lbase = rbase

    ldim = len(lbase)
    rdim = len(rbase)

    if otype == CREATION:
        slct = np.bitwise_and(rbase, 1<<index) == 0
        buff = 1
    elif otype == ANNIHILATION:
        slct = np.bitwise_and(rbase, 1<<index) != 0
        buff = -1
    else:
        raise ValueError("The invalid otype parameter!")

    col = np.where(slct == True)[0]
    kets = rbase[slct] + buff * (1<<index)
    row = np.searchsorted(lbase, kets)
    if (np.any(row >= ldim)) or (np.any(lbase[row] != kets)):
        raise ValueError("There must be some kets not found in the lbase!")

    entires = np.ones((len(col),), dtype=np.int64)
    data = (entries, (row, col))
    res = csr_matrix(data, shape=(ldim, rdim))
    return res
# }}}


def termmatrix(term, base, termtype,  statistics='F'):# {{{
    """
    Calculate the matrix representation of a general term in the Hilbert
    space specified by the base parameter!

    Parameter:
    ----------
    term: tuple
        A general term.
    base: tuple or list or np.ndarray
        The base of the Hilbert.
    termtype: str
        A string that tells whether the term is a hopping, pairing, hubbard or
        other kind of term.
    statistics: string, optional
        The kind of statistics rule the system obey. This parameter can be 
        only "F" or "B", which represent Fermi and Bose statistics respectively.
        default: "F"

    Return:
    -------
    res: csr_matrix
        The matrix representation of a general term.
    """

    if not isinstance(term, tuple):
        raise TypeError("The input term is not a tuple!")

    if statistics in ('f', 'F'):
        if not isinstance(base, (tuple, list)):
            raise TypeError("The base parameter is not of type tuple or list!")
        hopping = hopping_F
        hubbard = hubbard_F
        pairing = pairing_F
    elif statistics in ('b', 'B'):
        if not isinstance(base, np.ndarray):
            raise TypeError("The base parameter is not of type np.ndarray!")
        hopping = hopping_B
        hubbard = hubbard_B
        pairing = pairing_B
    else:
        raise ValueError("The invalid statistics parameter!")

    if termtype.lower() == "hopping":
        res = hopping(term, base)
    elif termtype.lower() == "hubbard":
        res = hubbard(term, base)
    elif termtype.lower() == "pairing":
        res = pairing(term, base)
    else:
        res = general(term, base)
    return res
# }}}


def aocmatrix(index, otype, rbase, lbase=None, statistics="F"):# {{{
    """
    Calculate the matrix representation of a creation or annihilation operator
    between the Hilbert space specified by the lbase and rbase parameter!
  
    This function applys to system which obey Bose statistics. For function
    applys to Fermi system, see function aocmatrix_F above.

    Parameter:
    ----------
    index: int
        The index of the creation or annihilation operator.
    otype: int
        The type of the operator, 1 represents creation and 0 represents
        annihilation.
    rbase: tuple or list
        The base of the Hilbert space before the operation.
    lbase: tuple or list, optional
        The base of the Hilbert space after the operation.
        If not given or None, lbase is the same as rbase.
        default: None
    statistics: string, optional
        The kind of statistics rule the system obey. This parameter can be 
        only "F" or "B", which represent Fermi and Bose statistics respectively.
        default: "F"

    Return:
    -------
    res: csr_matrix
        The matrix representation of the operator.
    """

    if statistics in ('f', 'F'):
        if not isinstance(rbase, (tuple, list)):
            raise TypeError("The rbase parameter is not of type tuple or list!")
        if lbase is not None:
            if not isinstance(lbase, (tuple, list)):
                raise TypeError("The lbase parameter is "
                                "not of type tuple or list!")
        func = aocmatrix_F
    elif statistics in ('b', 'B'):
        if not isinstance(rbase, np.ndarray):
            raise TypeError("The rbase parameter is not of type np.ndarray")
        if lbase is not None:
            if not isinstance(lbase, np.ndarray):
                raise TypeError("The lbase parameter is "
                                "not of type np.ndarray!")
        func = aocmatrix_B
    else:
        raise ValueError("The invalid statistics parameter!")

    res = func(index, otype, rbase, lbase)
    return res
# }}}
