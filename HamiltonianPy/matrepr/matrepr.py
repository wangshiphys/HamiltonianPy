from scipy.sparse import csr_matrix, identity

import numpy as np

from HamiltonianPy.matrepr import matreprcext as mrc
from HamiltonianPy.matrepr.bisearch import bisearch
from HamiltonianPy.constant import CREATION, ANNIHILATION
from HamiltonianPy.optor import Optor

__all__ = ['termmatrix', 'aocmatrix']

def hopping_F(term, base):# {{{
    """
    Calculate the matrix representation of the hopping term in the Hilbert
    space specified by the base parameter!

    This function applys to system which obey Fermi statistics. For function
    applys to Bose system, see function hopping_B below.

    Parameter:
    ----------
    term: Optor
        The hopping term.
    base: tuple or list
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the hopping term.
    """

    dim = len(base)
    shape = (dim, dim)
    (cindex, aindex), coeff = term()
    revision = 0.0

    if coeff == 0.0:
        return 0.0

    if term.otypes[1] == CREATION:
        buff = cindex
        cindex = aindex
        aindex = buff
        coeff = -coeff
        if cindex == aindex:
            revision = -coeff * identity(dim, format='csr')

    row, col, elmts = mrc.hopping(cindex, aindex, base)
    data = (elmts, (row, col))
    res = revision + coeff * csr_matrix(data, shape=shape)
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
    term: Optor
        The hubbard term.
    base: tuple or list
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the hubbard term.
    """

    dim = len(base)
    shape = (dim, dim)
    (cindex0, aindex0, cindex1, aindex1), coeff = term()

    if coeff == 0.0:
        return 0.0

    row, col, elmts = mrc.hubbard(cindex0, cindex1, base)
    data = (elmts, (col, col))
    res = coeff * csr_matrix(data, shape=shape)
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
    term: Optor
        The pairing term.
    base: tuple or list
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the pairing term.
    """

    dim = len(base)
    shape = (dim, dim)
    (index0, index1), coeff = term()
    otype = term.otypes[0]

    if coeff == 0.0 or index0 == index1:
        return 0.0

    row, col, elmts = mrc.pairing(index0, index1, otype, base)
    data = (elmts, (row, col))
    res = coeff * csr_matrix(data, shape=shape)
    return res
# }}}

def general(term, base):# {{{
    """
    Calculate the matrix representation of a general term in the Hilbert 
    space specified by the base parameter!

    Parameter:
    ----------
    term: Optor
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
    term: Optor
        The hopping term.
    base: np.ndarray
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the hopping term.
    """

    dim = len(base)
    shape = (dim, dim)
    (cindex, aindex), coeff = term()
    revision = 0.0

    if coeff == 0.0:
        return 0.0

    if term.otypes[1] == CREATION:
        buff = cindex
        cindex = aindex
        aindex = buff
        if cindex == aindex:
            revision = coeff * identity(dim, format='csr')

    slct_p = np.bitwise_and(base, 1<<aindex) != 0
    if cindex == aindex:
        col = np.where(slct_p == True)[0]
        row = col
    else:
        slct_h = np.bitwise_and(base, 1<<cindex) == 0
        slct_ket = np.logical_and(slct_p, slct_h)
        col = np.where(slct_ket == True)[0]
        kets = base[slct_ket] + (1<<cindex) - (1<<aindex)
        row = bisearch(list(kets), list(base))
 
    elmts = np.ones((len(col),), dtype=np.int64)
    data = (elmts, (row, col))
    res = revision + coeff * csr_matrix(data, shape=shape)
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
    term: Optor
        The hubbard term.
    base: np.ndarray
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the hubbard term.
    """

    dim = len(base)
    shape = (dim, dim)
    (cindex0, aindex0, cindex1, aindex1), coeff = term()

    if coeff == 0.0:
        return 0.0

    slct_p0 = np.bitwise_and(base, 1<<aindex0) != 0
    if aindex0 == aindex1:
        col = np.where(slct_p0 == True)[0]
    else:
        slct_p1 = np.bitwise_and(base, 1<<aindex1) != 0
        slct_ket = np.logical_and(slct_p0, slct_p1)
        col = np.where(slct_ket ==True)[0]

    elmts = np.ones((len(col),), dtype=np.int64)
    data = (elmts, (col, col))
    res = coeff * csr_matrix(data, shape=shape)
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
    term: Optor
        The pairing term.
    base: np.ndarray
        The base of the Hilbert.

    Return:
    -------
    res: csr_matrix
        The matrix representation of the pairing term.
    """

    dim = len(base)
    shape = (dim, dim)
    (index0, index1), coeff = term()
    otype = term.otypes[0]

    if coeff == 0.0 or index0 == index1:
        return 0.0

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
    row = bisearch(list(kets), list(base))
    data = (elmts, (row, col))
    res = coeff * csr_matrix(data, shape=shape)
    return res
# }}}

def aocmatrix_F(index, otype, lbase, rbase=None):# {{{
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
    lbase: tuple or list
        The base of the Hilbert space after the operation.
    rbase: tuple or list, optional
        The base of the Hilbert space before the operation.
        If not given or None, rbase is the same as lbase.
        default: None

    Return:
    -------
    res: csr_matrix
        The matrix representation of the operator.
    """

    if not isinstance(index, int):
        raise TypeError("The input index is not of type int!")

    if rbase is None:
        rbase = lbase

    ldim = len(lbase)
    rdim = len(rbase)
    shape = (ldim, rdim)

    row, col, elmts = mrc.aoc(index, otype, lbase, rbase) 
    data = (elmts, (row, col))
    res = csr_matrix(data, shape=shape)
    return res
# }}}

def aocmatrix_B(index, otype, lbase, rbase=None):# {{{
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
    lbase: np.ndarray
        The base of the Hilbert space after the operation.
    rbase: np.ndarray, optional
        The base of the Hilbert space before the operation.
        If not given or None, rbase is the same as lbase.
        default: None

    Return:
    -------
    res: csr_matrix
        The matrix representation of the operator.
    """

    if not isinstance(index, int):
        raise TypeError("The input index is not of type int!")

    if rbase is None:
        rbase = lbase

    ldim = len(lbase)
    rdim = len(rbase)
    shape = (ldim, rdim)

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
    row = bisearch(list(kets), list(lbase))
    elmts = np.ones((len(col),), dtype=np.int64)
    data = (elmts, (row, col))
    res = csr_matrix(data, shape=shape)
    return res
# }}}

def termmatrix(term, base, statistics="F"):# {{{
    """
    Calculate the matrix representation of a general term in the Hilbert
    space specified by the base parameter!

    Parameter:
    ----------
    term: Optor
        A general term.
    base: tuple or list or np.ndarray
        The base of the Hilbert.
    statistics: string, optional
        The kind of statistics rule the system obey. This parameter can be 
        only "F" or "B", which represent Fermi and Bose statistics respectively.
        default: "F"

    Return:
    -------
    res: csr_matrix
        The matrix representation of a general term.
    """

    if statistics == "F":
        if not isinstance(base, (tuple, list)):
            raise TypeError("The base parameter is not of type tuple or list!")
        hopping = hopping_F
        hubbard = hubbard_F
        pairing = pairing_F
    elif statistics == "B":
        if not isinstance(base, np.ndarray):
            raise TypeError("The base parameter is not of type np.ndarray!")
        hopping = hopping_B
        hubbard = hubbard_B
        pairing = pairing_B
    else:
        raise ValueError("The invalid statistics parameter!")

    if not isinstance(term, Optor):
        raise TypeError("The input term is not instance of Optor!")
    elif term.ishopping():
        res = hopping(term, base)
    elif term.ishubbard():
        res = hubbard(term, base)
    elif term.ispairing():
        res = pairing(term, base)
    else:
        res = general(term, base)
    return res
# }}}

def aocmatrix(index, otype, lbase, rbase=None, statistics="F"):# {{{
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
    lbase: np.ndarray or tuple or list
        The base of the Hilbert space after the operation.
    rbase: np.ndarray or tuple or list, optional
        The base of the Hilbert space before the operation.
        If not given or None, rbase is the same as lbase.
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

    if statistics == "F":
        if not isinstance(lbase, (tuple, list)):
            raise TypeError("The lbase parameter is not of type tuple or list!")
        if rbase is not None:
            if not isinstance(rbase, (tuple, list)):
                raise TypeError("The rbase parameter is "
                                "not of type tuple or list!")
        func = aocmatrix_F
    elif statistics == "B":
        if not isinstance(lbase, np.ndarray):
            raise TypeError("The lbase parameter is not of type np.ndarray")
        if rbase is not None:
            if not isinstance(rbase, np.ndarray):
                raise TypeError("The rbase parameter is "
                                "not of type np.ndarray!")
        func = aocmatrix_B
    else:
        raise ValueError("The invalid statistics parameter!")

    res = func(index, otype, lbase, rbase)
    return res
# }}}
