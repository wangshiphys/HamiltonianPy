"""Components to construct a model Hamiltonian
"""

from __future__ import absolute_import

__all__ = ["SiteID", "StateID", "AoC", "SpinOperator",
        "SpinInteraction", "ParticleTerm"]

from itertools import product
from scipy.sparse import csr_matrix, identity, kron

import numpy as np

from HamiltonianPy.constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP

import HamiltonianPy.extpkg.matrixrepr as cextmr


# Useful constants
ZOOM = 10000
SWAP_FACTOR_F = -1
SPIN_OTYPES = ('x', 'y', 'z', 'p', 'm')
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.int64)
SIGMA_Y = np.array([[0, -1], [1, 0]], dtype=np.int64) * 1j
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.int64)
SIGMA_P = np.array([[0, 2], [0, 0]], dtype=np.int64)
SIGMA_M = np.array([[0, 0], [2, 0]], dtype=np.int64)
SIGMA_MATRIX = {
        'x': SIGMA_X, 'y': SIGMA_Y, 'z': SIGMA_Z, 'p': SIGMA_P, 'm':SIGMA_M}


class SwapFermionError(Exception):# {{{
    """
    Raised when swap creation and annihilation operators of the same state
    """

    def __init__(self, aoc0, aoc1):
        self.aoc0 = aoc0
        self.aoc1 = aoc1

    def __str__(self):
        info = str(self.aoc0) + "\n" + str(self.aoc1) + "\n"
        info += "Swap these two operators would generate extra "
        info += "identity operator, which can not be processed properly."
        return info
# }}}


class SiteID:# {{{
    """
    A wrapper of 1D np.ndarray which represents the coordinates of a point

    The reason to define this wrapper is to make the coordinates hashable as
    well as comparable as a whole.

    Attributes
    ----------
    site : np.ndarray
        The coordinates of the point
        The shape of this array should be (1,), (2,) or (3,).

    Examples
    --------
    >>> import numpy as np
    >>> siteid0 = SiteID(site=np.array([0, 0]))
    >>> siteid1 = SiteID(site=np.array([1, 1]))
    >>> siteid0 < siteid1
    True
    >>> siteid0
    SiteID(site=array([0, 0]))
    """

    def __init__(self, site):# {{{
        """Customize the newly created instance

        Parameters
        ----------
        site : np.ndarray
            The coordinates of the point
            The shape of this array should be (1,), (2,) or (3,).
        """

        if isinstance(site, np.ndarray) and site.shape in [(1,), (2,), (3,)]:
            self._site = np.array(site, copy=True)
            self._site.setflags(write=False)
        else:
            raise TypeError("The 'site' parameter should be np.ndarray with"
                    "shape (1,), (2,) or (3,)")

        # The tuple form of the coordinates.
        # This internal attribute is useful in calculating the hash value of
        # the instance as well as defining compare logic between instances of
        # this class
        self._tupleform = tuple(int(i) for i in ZOOM * site)
    # }}}

    @property
    def site(self):# {{{
        """The site attribute of the instance
        """

        return np.array(self._site, copy=True)
    # }}}

    def __repr__(self):# {{{
        """The official string representation of the instance
        """

        return "SiteID(site={!r})".format(self._site)
    # }}}

    __str__ = __repr__

    def __hash__(self):# {{{
        """Return the hash code of the instance
        """

        return hash(self._tupleform)
    # }}}

    def __lt__(self, other):# {{{
        """Define the < operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tupleform < other._tupleform
        else:
            return NotImplemented
    # }}}

    def __eq__(self, other):# {{{
        """Define the == operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tupleform == other._tupleform
        else:
            return NotImplemented
    # }}}

    def __gt__(self, other):# {{{
        """Define the > operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tupleform > other._tupleform
        else:
            return NotImplemented
    # }}}

    def __le__(self, other):# {{{
        """Define the <= operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tupleform <= other._tupleform
        else:
            return NotImplemented
    # }}}

    def __ne__(self, other):# {{{
        """Define the != operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tupleform != other._tupleform
        else:
            return NotImplemented
    # }}}

    def __ge__(self, other):# {{{
        """Define the >= operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tupleform >= other._tupleform
        else:
            return NotImplemented
    # }}}

    def getIndex(self, indices_table):# {{{
        """Return the index associated with this SiteID

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of SiteID with an integer index

        Returns
        -------
        res : int
            The index of 'self' in the table
        """

        return indices_table(self)
    # }}}
# }}}


class StateID(SiteID):# {{{
    """A unified description of a single particle state

    Attributes
    ----------
    site : ndarray
        The coordinate of the localized state
        The site attribute should be 1D array with shape (1,), (2,) or (3,).
    spin : int
        The spin index of the state
    orbit : int
        The orbit index of the state

    Examples
    --------
    >>> import numpy as np
    >>> id0 = StateID(site=np.array([0, 0]), spin=1)
    >>> id1 = StateID(site=np.array([1, 1]), spin=1)
    >>> id0
    StateID(site=array([0, 0]), spin=1, orbit=0)
    >>> id0 < id1
    True
    """

    def __init__(self, site, spin=0, orbit=0):# {{{
        """Customize the newly created instance

        Parameters
        ----------
        site : ndarray
            The coordinate of the localized state
            The 'site' parameter should be 1D array with length 1, 2 or 3.
        spin : int, optional
            The spin index of the state
            default: 0
        orbit : int, optional
            The orbit index of the state
            default: 0
        """

        super().__init__(site=site)

        if isinstance(spin, int) and spin >= 0:
            self._spin = spin
        else:
            raise ValueError(
                    "The 'spin' parameter should be none negative integer")

        if isinstance(orbit, int) and orbit >= 0:
            self._orbit = orbit
        else:
            raise ValueError(
                    "The 'orbit' parameter should be none negative integer")

        # The self._tupleform on the right hand is already set properly
        # by calling 'super().__init__(site=site)'
        self._tupleform = (self._tupleform, spin, orbit)
    # }}}

    @property
    def spin(self):# {{{
        """The spin attribute of the instance
        """

        return self._spin
    # }}}

    @property
    def orbit(self):# {{{
        """The orbit attribute of the instance
        """

        return self._orbit
    # }}}

    def __repr__(self):# {{{
        """The offical string representation of the instance
        """

        info = "StateID(site={0!r}, spin={1}, orbit={2})"
        return info.format(self._site, self._spin, self._orbit)
    # }}}

    __str__ = __repr__
# }}}


class AoC:# {{{
    """A unified description of the creation and annihilation operator

    Attributes
    ----------
    otype : int
        The type of this operator
        It can be either 0 or 1, corresponding to annihilation or creation.
    state : StateID
        The state of this operator
    site : ndarray
        The coordinate of the localized state
        The site attribute should be 1D array with shape (1,), (2,) or (3,).
    spin : int
        The spin index of the state
    orbit : int
        The orbit index of the state

    Examples
    --------
    >>> import numpy as np
    >>> from termofH import AoC
    >>> c = AoC(otype=1, site=np.array([0, 0]), spin=0)
    >>> a = AoC(otype=0, site=np.array([0, 0]), spin=1)
    >>> c
    AoC(otype=CREATION, site=array([0, 0]), spin=0, orbit=0)
    >>> a
    AoC(otype=ANNIHILATION, site=array([0, 0]), spin=1, orbit=0)

    >>> c < a
    True

    >>> print(2 * c * a)
    coeff: 2
    AoC(otype=CREATION, site=array([0, 0]), spin=0, orbit=0)
    AoC(otype=ANNIHILATION, site=array([0, 0]), spin=1, orbit=0)
    >>> print(0.5 * c)
    coeff: 0.5
    AoC(otype=CREATION, site=array([0, 0]), spin=0, orbit=0)
    >>> print(a * (1+2j))
    coeff: (1+2j)
    AoC(otype=ANNIHILATION, site=array([0, 0]), spin=1, orbit=0)
    """

    def __init__(self, otype, site, spin=0, orbit=0):# {{{
        """Customize the newly created instance

        Parameters
        ----------
        otype : int
            The type of this operator
            It can be either 0 or 1, corresponding to annihilation or creation.
        site : ndarray
            The coordinate of the localized state
            The site attribute should be 1D array with shape (1,), (2,) or (3,).
        spin : int, optional
            The spin index of the state
            default: 0
        orbit : int, optional
            The orbit index of the state
            default: 0
        """

        if otype in (ANNIHILATION, CREATION):
            self._otype = otype
        else:
            raise ValueError(
                    "The 'otype' should be either CREATION or ANNIHILATION")

        state = StateID(site=site, spin=spin, orbit=orbit)
        self._state = state

        # The tuple form of the operator
        # It is a tuple: (otype, (site, spin,  orbit)) and site itself is a
        # tuple with length 1, 2, or 3.
        self._tupleform = (otype, state._tupleform)
    # }}}

    @property
    def otype(self):# {{{
        """The otype attribute of the instance
        """

        return self._otype
    # }}}

    @property
    def state(self):# {{{
        """The state attribute of the instance
        """

        return self._state
    # }}}

    @property
    def site(self):# {{{
        """The site attribute of the instance
        """

        return self._state.site
    # }}}

    @property
    def spin(self):# {{{
        """The spin attribute of the instance
        """

        return self._state.spin
    # }}}

    @property
    def orbit(self):# {{{
        """The orbit attribute of the instance
        """

        return self._state.orbit
    # }}}

    def getIndex(self, indices_table):# {{{
        """Return the index associated with this operator

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate this operator with an integer index

        Returns
        -------
        res : int
            The index of 'self' in the table

        See also
        --------
        getStateIndex
        """

        return indices_table(self)
    # }}}

    def getStateIndex(self, indices_table):# {{{
        """Return the index of the state associated with this operator

        Notes:
            This method is different from the getIndex method.
            This method return the index of the 'state' attribute of the
            operator and the getIndex method return the index of the operator
            itself.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of StateID with an integer index

        Returns
        -------
        res : int
            The index of the state attribute
        """

        return indices_table(self._state)
    # }}}

    def __repr__(self):# {{{
        """The offical string representation of the instance
        """

        otype = "CREATION" if self._otype == CREATION else "ANNIHILATION"
        info = "AoC(otype={0}, site={1!r}, spin={2}, orbit={3})"
        return info.format(otype, self.site, self.spin, self.orbit)
    # }}}

    __str__ = __repr__

    def __hash__(self):# {{{
        """Return the hash code of the instance
        """

        return hash(self._tupleform)
    # }}}

    def __lt__(self, other):# {{{
        """Define the < operator between self and other

        The comparsion logic is as follow:
        Creation operator is always compare less than annihilation operator;
        The smaller the stateid, the samller the creation operator;
        The larger the stateid, the smaller the annihilation operator.
        """

        if isinstance(other, self.__class__):
            otype0 = self._otype
            otype1 = other._otype
            state0 = self._state
            state1 = other._state
            if otype0 == CREATION and otype1 == CREATION:
                return state0 < state1
            elif otype0 == CREATION and otype1 == ANNIHILATION:
                return True
            elif otype0 == ANNIHILATION and otype1 == CREATION:
                return False
            else:
                return state0 > state1
        else:
            return NotImplemented
    # }}}

    def __gt__(self, other):# {{{
        """Define the > operator between self and other

        See also
        --------
        __lt__
        """

        if isinstance(other, self.__class__):
            otype0 = self._otype
            otype1 = other._otype
            state0 = self._state
            state1 = other._state
            if otype0 == CREATION and otype1 == CREATION:
                return state0 > state1
            elif otype0 == CREATION and otype1 == ANNIHILATION:
                return False
            elif otype0 == ANNIHILATION and otype1 == CREATION:
                return True
            else:
                return state0 < state1
        else:
            return NotImplemented
    # }}}

    def __eq__(self, other):# {{{
        """Define the == operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tupleform == other._tupleform
        else:
            return NotImplemented
    # }}}

    def __ne__(self, other):# {{{
        """Define the != operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tupleform != other._tupleform
        else:
            return NotImplemented
    # }}}

    def __le__(self, other):# {{{
        """Define the <= operator between self and other

        See also
        --------
        __lt__, __eq__
        """

        if isinstance(other, self.__class__):
            return self.__lt__(other) or self.__eq__(other)
        else:
            return NotImplemented
    # }}}

    def __ge__(self, other):# {{{
        """Define the >= operator between self and other

        See also
        --------
        __lt__, __gt__, __eq__
        """

        if isinstance(other, self.__class__):
            return self.__gt__(other) or self.__eq__(other)
        else:
            return NotImplemented
    # }}}

    def __mul__(self, other):# {{{
        """Implement the binary arithmetic operation: '*'

        'self' is the left operand and 'other' is the right operand
        Return an instance of ParticleTerm
        """

        if isinstance(other, self.__class__):
            term = ParticleTerm((self, other), coeff=1.0)
        elif isinstance(other, (int, float, complex)):
            term = ParticleTerm((self,), coeff=other)
        else:
            term = NotImplemented
        return term
    # }}}

    def __rmul__(self, other):# {{{
        """Implement the binary arithmetic operation '*'

        'self' is the right operand and 'other' is the left operand
        Return an instance of ParticleTerm
        """

        if isinstance(other, (int, float, complex)):
            term = ParticleTerm((self,), coeff=other)
        else:
            term = NotImplemented
        return term
    # }}}

    def dagger(self):# {{{
        """Return the Hermitian conjugate of self

        Returns
        -------
        res : A new instance of this class.
        """

        otype = ANNIHILATION if self._otype == CREATION else CREATION
        return self.derive(otype=otype)
    # }}}

    def conjugateOf(self, other):# {{{
        """Determine whether 'self' is the Hermitian conjugate of 'other'
        """

        if isinstance(other, self.__class__):
            return self.dagger() == other
        else:
            raise TypeError(
                    "The 'other' parameter is not instance of this class!")
    # }}}

    def sameState(self, other):# {{{
        """Determine whether 'self' and 'other' is defined on the same state
        """

        if isinstance(other, self.__class__):
            return self._state == other._state
        else:
            raise TypeError(
                    "The 'other' parameter is not instance of this class!")
    # }}}

    def derive(self, *, otype=None, site=None, spin=None, orbit=None):# {{{
        """Derive a new instance from 'self' and the given parameters

        This method creates a new instance with the same attribute as 'self'
        except for those given to this method.
        All the parameters should be specified as keyword arguments.

        Returns
        -------
        res : A new instance of AoC
        """

        if otype is None:
            otype = self.otype
        if site is None:
            site = self.site
        if spin is None:
            spin = self.spin
        if orbit is None:
            orbit = self.orbit

        return AoC(otype=otype, site=site, spin=spin, orbit=orbit)
    # }}}

    @staticmethod
    def matrixFunc(operator, rbase, *, lbase=None, to_csr=True):# {{{
        """
        Return the matrix representation of the operator in the Hilbert space

        Parameters
        ----------
        operator : tuple or list
            It is a 2-tuple or 2-list: (index, otype) or [index , otype]
            'index' is the index of the single-particle state
            'otype' is the "CREATION" or "ANNIHILATION" constant
        rbase : tuple or list
            The bases of the Hilbert space before the operation
        lbase : tuple or list, keyword-only, optional
            The bases of the Hilbert space after the operation
            If not given or None, lbase is the same as rbase
            default: None
        to_csr : boolean, keyword-only, optional
            Whether to construct a csr_matrix as the result
            default: True

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of the operator
            If to_csr is set to True, the result is a csr_matrix.
            If set to False, the result is a tuple: (entries, rows, cols).
            'entries' is the non-zero matrix elements, 'rows' and 'cols' are
            the row and column indices of the none-zero elements.
        """

        rdim = len(rbase)
        if lbase is None:
            shape = (rdim, rdim)
            data = cextmr.matrixRepr([operator], rbase)
        else:
            shape = (len(lbase), rdim)
            data = cextmr.matrixRepr([operator], rbase, lbase)

        res = csr_matrix(data, shape=shape) if to_csr else data
        return res
    # }}}

    def matrixRepr(self, indices_table, rbase, *, lbase=None, to_csr=True):# {{{
        """
        Return the matrix representation of this operator in the Hilbert space

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of StateID with an integer index
        rbase : tuple or list
            The bases of the Hilbert space before the operation
        lbase : tuple or list, keyword-only, optional
            The bases of the Hilbert space after the operation
            If not given or None, lbase is the same as rbase
            default: None
        to_csr : boolean, keyword-only, optional
            Whether to construct a csr_matrix as the result
            default: True

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of this operator
            If to_csr is set to True, the result is a csr_matrix.
            If set to False, the result is a tuple: (entries, rows, cols).
            'entries' is the non-zero matrix elements, 'rows' and 'cols' are
            the row and column indices of the none-zero elements.
        """

        operator = (self.getStateIndex(indices_table), self._otype)
        return self.matrixFunc(operator, rbase=rbase, lbase=lbase, to_csr=to_csr)
    # }}}
# }}}


class SpinOperator(SiteID):# {{{
    """A unified description of a spin opeartor

    Attributes
    ----------
    site : ndarray
        The site on which the spin operator is defined
        The site attribute should be 1D array with shape (1,), (2,) or (3,).
    otype : string
        The type of this spin operator
        It can be one of "x", "y", "z", "p" or "m", which represents the five
        type spin operator respectively.
    pauli : boolean
        Determine whether to use Pauli matrix or spin matrix
        default: False
    """

    def __init__(self, otype, site, *, pauli=False):# {{{
        """Customize the newly created instance

        Parameters
        ----------
        otype : string
            The type of this spin operator
            It can be only one of "x", "y", "z", "p" or "m", which represents
            the five type spin operator respectively.
        site : ndarray
            The site on which the spin operator is defined
            The site attribute should be 1D array with shape (1,), (2,) or (3,).
        pauli : boolean, keyword-only, optional
            Determine whether to use Pauli matrix or spin matrix
            default: False
        """

        super().__init__(site=site)

        if otype in SPIN_OTYPES:
            self._otype = otype
        else:
            raise TypeError(
                    "The 'otype' should be in ('x', 'y', 'z', 'p', 'm')")

        self.pauli = pauli

        # The self._tupleform on the right hand is already set properly
        # by calling the super().__init__(site=site)
        self._tupleform = (otype, self._tupleform)
    # }}}

    @property
    def otype(self):# {{{
        """The otype attribute of instance
        """

        return self._otype
    # }}}

    def getSiteID(self):# {{{
        """Extract the site information of this operator
        """

        return SiteID(site=self._site)
    # }}}

    def getSiteIndex(self, indices_table):# {{{
        """Return the index of the site on which this operator is defined

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of SiteID with an integer index

        Returns
        -------
        res : int
            The index of the site attribute of this instance
        """

        return self.getSiteID().getIndex(indices_table)
    # }}}

    def __repr__(self):# {{{
        """The offical string representation of the instance
        """

        info = "SpinOperator(otype={0}, site={1!r}, pauli={2})"
        return info.format(self._otype, self._site, self.pauli)
    # }}}

    __str__ = __repr__

    def __mul__(self, other):# {{{
        """Implement the binary arithmetic operation '*'

        'self' is the left operand and 'other' is the right operand
        Return an instance of SpinInteraction
        """

        if isinstance(other, self.__class__):
            term = SpinInteraction((self, other), coeff=1.0)
        elif isinstance(other, (int, float, complex)):
            term = SpinInteraction((self,), coeff=other)
        else:
            term = NotImplemented
        return term
    # }}}

    def __rmul__(self, other):# {{{
        """Implement the binary arithmetic operation '*'

        'self' parameter is the right operand and 'other' is the left operand
        Return an instance of SpinInteraction
        """

        if isinstance(other, (int, float, complex)):
            term = SpinInteraction((self,), coeff=other)
        else:
            term = NotImplemented
        return term
    # }}}

    def matrix(self):# {{{
        """Return the matrix representation of the spin operator

        The matrix representation is calculated in the single spin Hilbert
        space, i.e. 2 dimension.
        """

        factor = 1 if self.pauli else 0.5
        return factor * np.array(SIGMA_MATRIX[self._otype], copy=True)
    # }}}

    def dagger(self):# {{{
        """Return the Hermit conjuagte of this operator
        """

        if self._otype == 'p':
            return self.derive(otype='m')
        elif self._otype == 'm':
            return self.derive(otype='p')
        else:
            return self.derive()
    # }}}

    def conjugateOf(self, other):# {{{
        """Return whether 'self' is Hermit conjugate of 'other'
        """

        if isinstance(other, self.__class__):
            return self.dagger() == other
        else:
            raise TypeError(
                    "The 'other' parameter is not instance of this class!")
    # }}}

    def sameSite(self, other):# {{{
        """Return whether 'self' and 'other' is defined on the same site
        """

        if isinstance(other, self.__class__):
            return self.getSiteID() == other.getSiteID()
        else:
            raise TypeError(
                    "The 'other' parameter is not instance of this class!")
    # }}}

    def derive(self, *, otype=None, site=None):# {{{
        """Derive a new instance from 'self' and the given parameters

        This method creates a new instance with the same attribute as 'self'
        except for those given to this method.
        All the parameters should be specified as keyword arguments.

        Returns
        -------
        res : A new instance of SpinOperator
        """

        if otype is None:
            otype = self._otype
        if site is None:
            site = self._site
        return SpinOperator(otype=otype, site=site, pauli=self.pauli)
    # }}}

    def Schwinger(self):# {{{
        """Return the Schwinger Fermion representation of this spin operator

        Notes:
            The coeff of the generating ParticleTerm is related to the pauli
            attribute of self.
        """

        M = self.matrix()
        dim = M.shape[0]
        spins = [SPIN_UP, SPIN_DOWN]
        terms = []
        for spin0, row in zip(spins, range(dim)):
            for spin1, col in zip(spins, range(dim)):
                coeff = M[row, col]
                if coeff != 0:
                    C = AoC(otype=CREATION, site=self._site, spin=spin0)
                    A = AoC(otype=ANNIHILATION, site=self._site, spin=spin1)
                    term = ParticleTerm((C, A), coeff=coeff)
                    terms.append(term)
        return terms
    # }}}

    @staticmethod
    def matrixFunc(operator, totspin, *, pauli=False):# {{{
        """
        Calculate the matrix representation of the spin operator

        Parameters
        ----------
        operator : tuple or list
            It is a 2-tuple or 2-list: (index, otype) or [index, otype]
            'index' is the index of the site on which the spin operator is
            defined; 'otype' is the type of the spin operator which should be
            only one of ('x', 'y', 'z', 'p', m).
        totspin : int
            The total number of spins
        pauli : boolean, keyword-only, optional
            Determine whether to use Pauli matrix or spin matrix
            default: False

        Returns
        -------
        res : csr_matrix
            The matrix representation of this spin operator
        """

        factor = 1 if pauli else 0.5
        index, otype = operator
        I0 = identity(1<<index, dtype=np.int64, format="csr")
        I1 = identity(1<<(totspin - index - 1), dtype=np.int64, format="csr")
        S = factor * csr_matrix(SIGMA_MATRIX[otype])
        return kron(I1, kron(S, I0, format="csr"), format="csr")
    # }}}

    def matrixRepr(self, indices_table):# {{{
        """
        Return the matrix representation of this spin operator

        For every specific spin operator, its matrix representation in the
        Hilbert space is defined as follow:
                E_n * ... * E_(i+1) * S_i * E_(i-1) * .. * E_0
        where E is 2 dimension identity matrix, * represents tensor product,
        n is the total number of spins and i is the site index of this
        operator.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of SiteID with an integer index

        Returns
        -------
        res : csr_matrix
            The matrix representation of the spin operator
        """

        totspin = len(indices_table)
        operator = (self.getSiteIndex(indices_table), self._otype)
        return self.matrixFunc(operator, totspin=totspin, pauli=self.pauli)
    # }}}
# }}}


class SpinInteraction:# {{{
    """A unified description of spin interaction term
    """

    def __init__(self, operators, coeff=1.0):# {{{
        """Customize the newly created instance

        Parameters
        ----------
        operators : A sequence of SpinOperator objects
        coeff : int, float, complex, optional
            The coefficience of this term
            default: 1.0
        """

        if isinstance(coeff, (int, float, complex)):
            self._coeff = coeff
        else:
            raise TypeError("The input coeff is not a number.")

        # Sorting the spin operators in ascending order according to their
        # SiteID. The relative position of two operators with the same SiteID
        # will not change and the exchange of two spin operators on different
        # site never change the interaction term.
        self._operators = sorted(operators, key=lambda item: item.getSiteID())
    # }}}

    def __str__(self):# {{{
        """Return a string that describes the content of the instance
        """

        info = "coeff: {0}\n".format(self._coeff)
        for operator in self._operators:
            info += str(operator) + '\n'
        return info
    # }}}

    def __mul__(self, other):# {{{
        """
        Implement the binary arithmetic operation '*'

        'self' is the left operand and 'other' is the right operand
        Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            operators = self._operators + other._operators
            coeff = self._coeff * other._coeff
        elif isinstance(other, SpinOperator):
            operators = self._operators + [other]
            coeff = self._coeff
        elif isinstance(other, (int, float, complex)):
            operators = self._operators
            coeff = self._coeff * other
        else:
            return NotImplemented

        return SpinInteraction(operators, coeff=coeff)
    # }}}

    def __rmul__(self, other):# {{{
        """
        Implement the binary arithmetic operation '*'

        'self' is the right operand and 'other' is the left operand
        This method return a new instance of this class.
        If you just want to update the coeff, use updateCoeff() method instead.
        """

        if isinstance(other, SpinOperator):
            operators = [other] + self._operators
            coeff = self._coeff
        elif isinstance(other, (int, float, complex)):
            operators = self._operators
            coeff = other * self._coeff
        else:
            return NotImplemented

        return SpinInteraction(operators, coeff=coeff)
    # }}}

    def dagger(self):# {{{
        """Return the Hermit conjugate of this term
        """

        operators = [operator.dagger() for operator in self._operators[::-1]]
        return SpinInteraction(operators, coeff=self._coeff.conjugate())
    # }}}

    def setCoeff(self, coeff):# {{{
        """Set the 'coeff' attribute of the instance
        """

        if isinstance(coeff, (int, float, complex)):
            self._coeff = coeff
        else:
            raise TypeError("The 'coeff' parameter should be a number.")
    # }}}

    def Schwinger(self):# {{{
        """Return the Schwinger Fermion representation of this term
        """

        fermion_reprs = [operator.Schwinger() for operator in self._operators]
        terms = []
        for term in product(*fermion_reprs):
            res_term = self._coeff
            for sub_term in term:
                res_term = res_term * sub_term
            terms.append(res_term)
        return terms
    # }}}

    @staticmethod
    def matrixFunc(operators, totspin, coeff=1.0, *, pauli=False):# {{{
        """Return the matrix representation of the spin interaction term

        Parameters
        ----------
        operators : sequence
            A sequence of 2-tuple: [(index_0, otype_0), ..., (index_n, otype_n)]
            'index_n' is the index of the site on which the spin operator is
            defined; and 'otype_n' is the type of the spin operator which
            should be only one of 'x', 'y', 'z', 'p' or 'm'.
        totspin: int
            The total number of spins
        coeff : int, float or complex, optional
            The coefficience of the term
            default: 1.0
        pauli: boolean, keyword-only, optional
            Determine whether to use sigma matrix or spin matrix
            default: False

        Returns
        -------
        res : csr_matrix
            The matrix representation of this term
        """

        factor = 1 if pauli else 0.5
        operators = sorted(operators, key=lambda item: item[0])
        if len(operators) == 2 and operators[0][0] != operators[1][0]:
            (index0, otype0), (index1, otype1) = operators
            # The coeff of this term is multiplied to S0 operator
            S0 = (coeff * factor) * csr_matrix(SIGMA_MATRIX[otype0])
            S1 = factor * csr_matrix(SIGMA_MATRIX[otype1])
            dim0 = 1<<index0
            dim1 = 1<<(index1 - index0 - 1)
            dim2 = 1<<(totspin - index1 - 1)
            if dim0 == 1:
                res = S0
            else:
                I = identity(dim0, dtype=np.int64, format="csr")
                res = kron(S0, I, format="csr")
            if dim1 == 1:
                res = kron(S1, res, format="csr")
            else:
                I = identity(dim1, dtype=np.int64, format="csr")
                res = kron(S1, kron(I, res, format="csr"), format="csr")
            if dim2 != 1:
                I = identity(dim2, dtype=np.int64, format="csr")
                res = kron(I, res, format="csr")
        else:
            res = coeff * identity(1<<totspin, dtype=np.int64, format="csr")
            for index, otype in operators:
                I0 = identity(1<<index, dtype=np.int64, format="csr")
                I1 = identity(1<<(totspin-index-1), dtype=np.int64, format="csr")
                S = factor * csr_matrix(SIGMA_MATRIX[otype])
                res = res.dot(kron(I1, kron(S, I0, format="csr"), format="csr"))
        return res
    # }}}

    def matrixRepr(self, indices_table, coeff=None, *, pauli=False):# {{{
        """Return the matrix representation of this spin interaction term

        Parameters
        ----------
        indices_table : IndexTable
            The table that associate integer index with SiteID
        coeff : int, float or complex, optional
            The given coefficience of this term.
            default: None
        pauli: boolean, keyword-only, optional
            Determine whether to use sigma matrix or spin matrix
            default: False

        Returns
        -------
        res : csr_matrix
            The matrix representation of this term
        """

        if coeff is not None:
            self.setCoeff(coeff)

        operators = [(operator.getSiteIndex(indices_table), operator.otype)
                for operator in self._operators]
        res = self.matrixFunc(operators=operators, totspin=len(indices_table),
                coeff=self._coeff, pauli=pauli)
        return res
    # }}}
# }}}


class ParticleTerm:# {{{
    """A unified description of any operator composed of fermion operators
    """

    def __init__(self, aocs, coeff=1.0):# {{{
        """Customize the newly created instance

        Paramters
        ---------
        aocs : tuple or list
            The creation and/or annihilation operators that compose this term
        coeff : float, int or complex, optional
            The coefficience of this term
            default: 1.0
        """

        # The "normalized" attribute indicates whether this term is normalized
        # See docstring of the normalize method for the definition of
        # a normalized term
        try:
            normal_aocs, swap_num = self.normalize(aocs)
            self.normalized = True
        except SwapFermionError:
            normal_aocs = list(aocs)
            swap_num = 0
            self.normalized = False

        self._aocs = normal_aocs
        self._coeff = (SWAP_FACTOR_F ** swap_num) * coeff
    # }}}

    def __str__(self):# {{{
        """Return a string that describes the content of this instance
        """

        info = "coeff: {0}\n".format(self._coeff)
        for aoc in self._aocs:
            info += str(aoc) + '\n'
        return info
    # }}}

    def __mul__(self, other):# {{{
        """Implement the binary arithmetic operation '*'

        'self' is the left operand and 'other' is the right operand
        Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            aocs = self._aocs + other._aocs
            coeff = self._coeff * other._coeff
        elif isinstance(other, AoC):
            aocs = self._aocs + [other]
            coeff = self._coeff
        elif isinstance(other, (int, float, complex)):
            aocs = self._aocs
            coeff = self._coeff * other
        else:
            return NotImplemented

        return ParticleTerm(aocs=aocs, coeff=coeff)
    # }}}

    def __rmul__(self, other):# {{{
        """Implement the binary arithmetic operation '*'

        'self' is the right operand and 'other' is the left operand
        This method return a new instance of this class
        """

        if isinstance(other, AoC):
            aocs = [other] + self._aocs
            coeff = self._coeff
        if isinstance(other, (int, float, complex)):
            aocs = self._aocs
            coeff = other * self._coeff
        else:
            return NotImplemented

        return ParticleTerm(aocs=aocs, coeff=coeff)
    # }}}

    @staticmethod
    def normalize(aoc_seq):# {{{
        """Reordering the given 'aco_seq' into norm form

        For a composite operator consist of creation and annihilation operators,
        the norm form means that all the creation operators appear to the left
        of all the annihilation operators. Also, the creation and annihilation
        operators are sorted in ascending and descending order respectively
        according to the single particle state associated with the operator.

        See the document of __lt__ method of AoC for the comparsion logic.

        Parameters
        ----------
        aoc_seq : list or tuple
            A collection of creation and/or annihilation operators

        Returns
        -------
        seq : list
            The norm form of the operator
        swap_num : int
            The number swap to get the normal form

        Raises
        ------
        SwapFermionError :
            Exceptions raised when swap creation and annihilation operator
            that defined on the same single particle state.
        """

        seq = list(aoc_seq[:])
        seq_len = len(aoc_seq)
        swap_num = 0

        for length in range(seq_len, 1, -1):
            for i in range(0, length-1):
                aoc0, aoc1 = seq[i:i+2]
                id0 = aoc0.state
                id1 = aoc1.state
                if aoc0 > aoc1:
                    if id0 != id1:
                        seq[i] = aoc1
                        seq[i+1] = aoc0
                        swap_num += 1
                    else:
                        raise SwapFermionError(aoc0, aoc1)
        return seq, swap_num
    # }}}

    def isPairing(self, tag=None):# {{{
        """Return whether this term is a pairing term

        This method is only implemented for the instance which the 'normailzed'
        attribute set to True. For these instances that the 'normalized'
        attribute is False, this method will raise NotImplementedError.

        Parameters
        ----------
        tag : string, optional
            Determine the judgment criterion.
            If 'tag' is none, both particle and hole pairing is ok.
            If 'tag' is 'p' then only particle is ok and if 'tag' is 'h'
            only hole pairing is ok.
            default: None
        """

        if self.normalized:
            if len(self._aocs) == 2:
                otype0 = self._aocs[0].otype
                otype1 = self._aocs[1].otype
                if tag in ('p', 'P'):
                    judge = (otype0==CREATION and otypes==CREATION)
                elif tag in ('h', 'H'):
                    judge = (otype0==ANNIHILATION and otypes==ANNIHILATION)
                else:
                    judge = otype0 == otype1
                return True if judge else False
            else:
                return False
        else:
            raise NotImplementedError(
                    "This method is not implemented for unnormailzed term.")
    # }}}

    def isHopping(self, tag=None):# {{{
        """Return whether this term is a hopping term

        This method is only implemented for the instance which the 'normailzed'
        attribute is set to True. For these instances that the 'normalized'
        attribute is False, this method will raise NotImplementedError.

        Parameters
        ----------
        tag : string, optional
            Determine the judgment criterion.
            If 'tag' is none, arbitrary hopping term is ok
            If tag is 'n' only the number operator is ok
            default: None
        """

        if self.normalized:
            if len(self._aocs) == 2:
                c0 = self._aocs[0].otype == CREATION
                c1 = self._aocs[1].otype == ANNIHILATION
                if tag in ('n', 'N'):
                    c2 = self._aocs[0].sameState(self._aocs[1])
                    judge = c0 and c1 and c2
                else:
                    judge = c0 and c1
                return True if judge else False
            else:
                return False
        else:
            raise NotImplementedError(
                    "This method is not implemented for unnormailzed term.")
    # }}}

    def isHubbard(self):# {{{
        """Return whether this term is a hopping term

        This method is only implemented for the instance which the 'normailzed'
        attribute is set to True. For these instances that the 'normalized'
        attribute is False, this method will raise NotImplementedError.
        """

        if self.normalized:
            if len(self._aocs) == 4:
                judge = (self._aocs[0].otype == CREATION
                        and self._aocs[1].otype == CREATION
                        and self._aocs[2].otype == ANNIHILATION
                        and self._aocs[3].otype == ANNIHILATION
                        and self._aocs[0].sameState(self._aocs[3])
                        and self._aocs[1].sameState(self._aocs[2]))
                return True if judge else False
            else:
                return False
        else:
            raise NotImplementedError(
                    "This method is not implemented for unnormailzed term.")
    # }}}

    def dagger(self):# {{{
        """Return the Hermit conjugate of this term
        """

        aocs = [aoc.dagger() for aoc in self._aocs[::-1]]
        return ParticleTerm(aocs=aocs, coeff=self._coeff.conjugate())
    # }}}

    def setCoeff(self, coeff):# {{{
        """Set the coefficience of this term
        """

        if isinstance(coeff, (int, float, complex)):
            self._coeff = coeff
        else:
            raise TypeError("The input 'coeff' should be a number!")
    # }}}

    @staticmethod
    def matrixFunc(operators, rbase, *, lbase=None, coeff=1.0):# {{{
        """Return the matrix representation of the term

        Parameters
        ----------
        operators : list or tuple
            A sequence of 2-tuple: [(index_0, otype_0), ..., (index_n, otype_n)]
            'index_n' is the index of the state and 'otype_n' is the type of
            the operator which can be either CREATION(1) or ANNIHILATION(0).
        rbase : tuple or list
            The bases of the Hilbert space before the operation
        lbase : tuple or list, keyword-only, optional
            The bases of the Hilbert space after the operation
            It not given or None, lbase is the same as rbase.
            default: None
        coeff : int, float or complex, keyword-only, optional
            The coefficience of the term
            default: 1.0

        Returns
        -------
        res : csr_martrix
            The matrix representation of the term
        """

        rdim = len(rbase)
        if lbase is None:
            shape = (rdim, rdim)
            data = cextmr.matrixRepr(operators, rbase)
        else:
            shape = (len(lbase), rdim)
            data = cextmr.matrixRepr(operators, rbase, lbase)

        return coeff * csr_matrix(data, shape=shape)
    # }}}

    def matrixRepr(self, indices_table, rbase, *, lbase=None, coeff=None):# {{{
        """Return the matrix representation of this term

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of StateID with an integer index
        rbase : tuple or list
            The bases of the Hilbert space before the operation
        lbase : tuple or list, keyword-only, optional
            The bases of the Hilbert space after the operation
            It not given or None, lbase is the same as rbase.
            default: None
        coeff : int, float or complex, keyword-only, optional
            The coefficience of the term
            default: None

        Returns
        -------
        res : csr_martrix
            The matrix representation of the term
        """

        if coeff is not None:
            self.setCoeff(coeff)

        operators = [(aoc.getStateIndex(indices_table), aoc.otype)
                for aoc in self._aocs]
        return self.matrixFunc(operators, rbase, lbase=lbase, coeff=self._coeff)
    # }}}
# }}}
