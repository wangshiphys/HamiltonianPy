"""
Components for constructing a model Hamiltonian
"""


__all__ = [
    "SiteID",
    "StateID",
    "AoC",
    "SpinOperator",
    "SpinInteraction",
    "ParticleTerm",
]


from itertools import product
from scipy.sparse import csr_matrix, identity, kron

import numpy as np

from HamiltonianPy.constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP
# Matrix representation extension
import HamiltonianPy.extension as ext


# Useful constants
ZOOM = 10000
SPIN_OTYPES = ("x", "y", "z", "p", "m")
NUMERIC_TYPES = (int, float, complex, np.number)
SPIN_MATRICES = {
    "x": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
    "y": np.array([[0.0, -0.5], [0.5, 0.0]], dtype=np.float64) * 1j,
    "z": np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64),
    "p": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
    "m": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
}


class SwapError(Exception):
    """
    Raised when swapping creation and annihilation operators defined on the
    same single-particle state
    """

    def __init__(self, aoc0, aoc1):
        self.aoc0 = aoc0
        self.aoc1 = aoc1

    def __str__(self):
        msg = "Swapping the following two operators would generate extra " \
              "identity operator which can not be processed properly:\n"
        msg += "    {0!r}\n    {1!r}".format(self.aoc0, self.aoc1)
        return msg


class SiteID:
    """
    A wrapper over 1D np.ndarray which is the coordinate of a lattice site

    The reason to define this wrapper is to make the coordinate hashable as
    well as comparable as a whole.

    Attributes
    ----------
    site : np.ndarray
        The coordinate of the lattice site

    Examples
    --------
    >>> import numpy as np
    >>> site0 = SiteID(site=np.array([0, 0]))
    >>> site1 = SiteID(site=np.array([1, 1]))
    >>> site0 < site1
    True
    >>> site0
    SiteID(site=array([0, 0]))
    """

    def __init__(self, site):
        """
        Customize the newly created instance

        Parameters
        ----------
        site : np.ndarray
            The coordinate of the lattice site
            The shape of this array should be (1,), (2,) or (3,).
        """

        assert isinstance(site, np.ndarray) and site.shape in [(1,), (2,), (3,)]

        self._site = np.array(site, copy=True)
        self._site.setflags(write=False)

        # The tuple form of this instance
        # This internal attribute is useful in calculating the hash value of
        # the instance as well as defining compare logic between instances of
        # this class
        self._tuple_form = tuple(int(i) for i in ZOOM * site)

    @property
    def site(self):
        """
        The `site` attribute
        """

        return np.array(self._site, copy=True)

    def __repr__(self):
        """
        The official string representation of the instance
        """

        return "SiteID(site={!r})".format(self._site)

    __str__ = __repr__

    def __hash__(self):
        """
        Return the hash code of the instance
        """

        return hash(self._tuple_form)

    def __lt__(self, other):
        """
        Define the `<` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form < other._tuple_form
        else:
            return NotImplemented

    def __eq__(self, other):
        """
        Define the `==` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form == other._tuple_form
        else:
            return NotImplemented

    def __gt__(self, other):
        """
        Define the `>` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form > other._tuple_form
        else:
            return NotImplemented

    def __le__(self, other):
        """
        Define the `<=` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form <= other._tuple_form
        else:
            return NotImplemented

    def __ne__(self, other):
        """
        Define the `!=` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form != other._tuple_form
        else:
            return NotImplemented

    def __ge__(self, other):
        """
        Define the `>=` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form >= other._tuple_form
        else:
            return NotImplemented

    def getIndex(self, indices_table):
        """
        Return the index associated with this instance

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of SiteID with an integer index

        Returns
        -------
        res : int
            The index of this instance in the table
        """

        return indices_table(self)


class StateID(SiteID):
    """
    A unified description of a single-particle state

    Attributes
    ----------
    site : ndarray
        The coordinate of the localized single-particle state
    spin : int
        The spin index of the single-particle state
    orbit : int
        The orbit index of the single-particle state

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

    def __init__(self, site, spin=0, orbit=0):
        """
        Customize the newly created instance

        Parameters
        ----------
        site : ndarray
            The coordinate of the localized single-particle state
            The `site` parameter should be 1D array with length 1, 2 or 3.
        spin : int, optional
            The spin index of the single-particle state
            default: 0
        orbit : int, optional
            The orbit index of the single-particle state
            default: 0
        """

        assert (spin, int) and spin >= 0
        assert (orbit, int) and orbit >= 0

        super().__init__(site=site)
        self._spin = spin
        self._orbit = orbit

        # The self._tuple_form on the right hand is already set properly
        # by calling `super().__init__(site=site)`
        self._tuple_form = (self._tuple_form, spin, orbit)

    @property
    def spin(self):
        """
        The `spin` attribute
        """

        return self._spin

    @property
    def orbit(self):
        """
        The `orbit` attribute
        """

        return self._orbit

    def __repr__(self):
        """
        The official string representation of the instance
        """

        info = "StateID(site={0!r}, spin={1}, orbit={2})"
        return info.format(self._site, self._spin, self._orbit)

    __str__ = __repr__


class AoC:
    """
    A unified description of the creation and annihilation operator

    Attributes
    ----------
    otype : int
        The type of this operator
        It can be either 0 or 1, corresponding to annihilation and creation
        respectively.
    state : StateID
        The single-particle state on which this operator is defined
    site : ndarray
        The coordinate of the localized single-particle state
    spin : int
        The spin index of the single-particle state
    orbit : int
        The orbit index of the single-particle state

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
    The coefficient of this term: 2
    The component operators:
        AoC(otype=CREATION, site=array([0, 0]), spin=0, orbit=0)
        AoC(otype=ANNIHILATION, site=array([0, 0]), spin=1, orbit=0)

    >>> print(0.5 * c)
    The coefficient of this term: 0.5
    The component operators:
        AoC(otype=CREATION, site=array([0, 0]), spin=0, orbit=0)

    >>> print(a * (1+2j))
    The coefficient of this term: (1+2j)
    The component operators:
        AoC(otype=ANNIHILATION, site=array([0, 0]), spin=1, orbit=0)

    """

    def __init__(self, otype, site, spin=0, orbit=0):
        """
        Customize the newly created instance

        Parameters
        ----------
        otype : int
            The type of this operator
            It can be either 0 or 1, corresponding to annihilation and
            creation respectively.
        site : ndarray
            The coordinate of the localized single-particle state
            The `site` parameter should be 1D array with length 1,2 or 3.
        spin : int, optional
            The spin index of the single-particle state
            default: 0
        orbit : int, optional
            The orbit index of the single-particle state
            default: 0
        """

        assert otype in (ANNIHILATION, CREATION)

        self._otype = otype

        state = StateID(site=site, spin=spin, orbit=orbit)
        self._state = state

        # The tuple form of this instance
        # It is a tuple: (otype, (site, spin, orbit)) and site itself is a
        # tuple with length 1, 2, or 3.
        self._tuple_form = (otype, state._tuple_form)

    @property
    def otype(self):
        """
        The `otype` attribute
        """

        return self._otype

    @property
    def state(self):
        """
        The `state` attribute
        """

        return self._state

    @property
    def site(self):
        """
        The `site` attribute
        """

        return self._state.site

    @property
    def spin(self):
        """
        The `spin` attribute
        """

        return self._state.spin

    @property
    def orbit(self):
        """
        The `orbit` attribute
        """

        return self._state.orbit

    def getIndex(self, indices_table):
        """
        Return the index of this operator

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of AoC with an integer index

        Returns
        -------
        res : int
            The index of this instance in the table

        See also
        --------
        getStateIndex
        """

        return indices_table(self)

    def getStateIndex(self, indices_table):
        """
        Return the index of the single-particle state on which this operator is
        defined

        Notes:
            This method is different from the getIndex method.
            This method return the index of the `state` attribute of the
            operator and the getIndex method return the index of the operator
            itself.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of StateID with an integer index

        Returns
        -------
        res : int
            The index of the `state` attribute
        """

        return indices_table(self._state)

    def __repr__(self):
        """
        The official string representation of the instance
        """

        otype = "CREATION" if self._otype == CREATION else "ANNIHILATION"
        info = "AoC(otype={0}, site={1!r}, spin={2}, orbit={3})"
        return info.format(otype, self.site, self.spin, self.orbit)

    __str__ = __repr__

    def __hash__(self):
        """
        Return the hash code of the instance
        """

        return hash(self._tuple_form)

    def __lt__(self, other):
        """
        Define the `<` operator between self and other

        The comparison logic is as follow:
        Creation operator is always compare less than annihilation operator;
        The smaller the single-particle state, the smaller the creation
        operator; The larger the single-particle state, the smaller the
        annihilation operator.
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

    def __gt__(self, other):
        """
        Define the `>` operator between self and other

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

    def __eq__(self, other):
        """
        Define the `==` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form == other._tuple_form
        else:
            return NotImplemented

    def __ne__(self, other):
        """
        Define the `!=` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form != other._tuple_form
        else:
            return NotImplemented

    def __le__(self, other):
        """
        Define the `<=` operator between self and other

        See also
        --------
        __lt__, __eq__
        """

        if isinstance(other, self.__class__):
            return self.__lt__(other) or self.__eq__(other)
        else:
            return NotImplemented

    def __ge__(self, other):
        """
        Define the `>=` operator between self and other

        See also
        --------
        __lt__, __gt__, __eq__
        """

        if isinstance(other, self.__class__):
            return self.__gt__(other) or self.__eq__(other)
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` is the left operand and `other` is the right operand
        Return an instance of ParticleTerm
        """

        if isinstance(other, self.__class__):
            return ParticleTerm((self, other), coeff=1.0)
        elif isinstance(other, NUMERIC_TYPES):
            return ParticleTerm((self,), coeff=other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` is the right operand and `other` is the left operand
        Return an instance of ParticleTerm
        """

        if isinstance(other, NUMERIC_TYPES):
            return ParticleTerm((self,), coeff=other)
        else:
            return NotImplemented

    def dagger(self):
        """
        Return the Hermitian conjugate of this operator

        Returns
        -------
        res : A new instance of this class
        """

        otype = ANNIHILATION if self._otype == CREATION else CREATION
        return self.derive(otype=otype)

    def conjugate_of(self, other):
        """
        Determine whether `self` is the Hermitian conjugate of `other`
        """

        if isinstance(other, self.__class__):
            return self._otype != other._otype and self._state == other._state
        else:
            raise TypeError(
                "The `other` parameter is not instance of this class!"
            )

    def same_state(self, other):
        """
        Determine whether `self` and `other` is defined on the same
        single-particle state
        """

        if isinstance(other, self.__class__):
            return self._state == other._state
        else:
            raise TypeError(
                "The `other` parameter is not instance of this class!"
            )

    def derive(self, *, otype=None, site=None, spin=None, orbit=None):
        """
        Derive a new instance from `self` and the given parameters

        This method creates a new instance with the same attribute as `self`
        except for these given to this method.
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

    def matrix_repr(self, indices_table, right_bases, *, left_bases=None,
                    to_csr=True):
        """
        Return the matrix representation of this operator in the Hilbert space

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of StateID with an integer index
        right_bases : tuple
            The bases of the Hilbert space before the operation
        left_bases : tuple, keyword-only, optional
            The bases of the Hilbert space after the operation
            If not given or None, left_bases is the same as right_bases
            default: None
        to_csr : boolean, keyword-only, optional
            Whether to construct a csr_matrix as the result
            default: True

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of the operator in the Hilbert space
            If `to_csr` is set to True, the result is a csr_matrix;
            If set to False, the result is a tuple: (entries, (rows, cols)),
            where `entries` is the non-zero matrix elements, `rows` and
            `cols` are the row and column indices of the none-zero elements.
        """

        operator = (self.getStateIndex(indices_table), self._otype)
        res = ext.matrix_function(
            [operator], right_bases, left_bases=left_bases, to_csr=to_csr
        )
        return res


class SpinOperator(SiteID):
    """
    A unified description of a spin operator

    Attributes
    ----------
    site : ndarray
        The coordinate of the lattice site on which the spin operator is defined
    otype : string
        The type of this spin operator
        Valid value: "x" | "y" | "z" | "p" | "m"
    """

    def __init__(self, otype, site):
        """
        Customize the newly created instance

        Parameters
        ----------
        otype : str
            The type of this spin operator
            Valid value: "x" | "y" | "z" | "p" | "m"
        site : ndarray
            The coordinate of the lattice site on which the spin operator is
            defined. The `site` parameter should be 1D array with shape (1,),
            (2,) or (3,).
        """

        assert otype in SPIN_OTYPES

        super().__init__(site=site)
        self._otype = otype

        # The self._tuple_form on the right hand is already set properly
        # by calling the super().__init__(site=site)
        self._tuple_form = (otype, self._tuple_form)

    @property
    def otype(self):
        """
        The `otype` attribute
        """

        return self._otype

    def getSiteID(self):
        """
        Extract the site information of this operator
        """

        return SiteID(site=self._site)

    def getSiteIndex(self, indices_table):
        """
        Return the index of the lattice site on which this operator is defined

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

    def __repr__(self):
        """
        The official string representation of the instance
        """

        info = 'SpinOperator(otype="{0}", site={1!r})'
        return info.format(self._otype, self._site)

    __str__ = __repr__

    def __mul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` is the left operand and `other` is the right operand
        Return an instance of SpinInteraction
        """

        if isinstance(other, self.__class__):
            return SpinInteraction((self, other), coeff=1.0)
        elif isinstance(other, NUMERIC_TYPES):
            return SpinInteraction((self,), coeff=other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` parameter is the right operand and `other` is the left operand
        Return an instance of SpinInteraction
        """

        if isinstance(other, NUMERIC_TYPES):
            return SpinInteraction((self,), coeff=other)
        else:
            return NotImplemented

    def matrix(self):
        """
        Return the matrix representation of the spin operator

        The matrix representation is calculated in the single spin Hilbert
        space, i.e. 2 dimension.
        """

        return np.array(SPIN_MATRICES[self._otype], copy=True)

    def dagger(self):
        """
        Return the Hermitian conjugate of this operator
        """

        if self._otype == "p":
            operator = self.derive(otype="m")
        elif self._otype == "m":
            operator = self.derive(otype="p")
        else:
            operator = self.derive()
        return operator

    def conjugate_of(self, other):
        """
        Return whether `self` is Hermitian conjugate of `other`
        """

        if isinstance(other, self.__class__):
            return self.dagger() == other
        else:
            raise TypeError(
                "The `other` parameter is not instance of this class!"
            )

    def same_site(self, other):
        """
        Return whether `self` and `other` is defined on the same lattice site
        """

        if isinstance(other, self.__class__):
            return self.getSiteID() == other.getSiteID()
        else:
            raise TypeError(
                "The `other` parameter is not instance of this class!"
            )

    def derive(self, *, otype=None, site=None):
        """
        Derive a new instance from `self` and the given parameters

        This method creates a new instance with the same attribute as `self`
        except for these given to this method.
        All the parameters should be specified as keyword arguments.

        Returns
        -------
        res : A new instance of SpinOperator
        """

        if otype is None:
            otype = self._otype
        if site is None:
            site = self._site
        return SpinOperator(otype=otype, site=site)

    def Schwinger(self):
        """
        Return the Schwinger Fermion representation of this spin operator
        """

        C_UP = AoC(otype=CREATION, site=self._site, spin=SPIN_UP)
        C_DOWN = AoC(otype=CREATION, site=self._site, spin=SPIN_DOWN)
        A_UP = AoC(otype=ANNIHILATION, site=self._site, spin=SPIN_UP)
        A_DOWN = AoC(otype=ANNIHILATION, site=self._site, spin=SPIN_DOWN)
        tmp = [(C_UP, A_UP), (C_UP, A_DOWN), (C_DOWN, A_UP), (C_DOWN, A_DOWN)]

        terms = []
        for coeff, term in zip(SPIN_MATRICES[self._otype].flat, tmp):
            if coeff != 0:
                terms.append(ParticleTerm(term, coeff=coeff))
        return terms

    @staticmethod
    def matrix_function(operator, total_spin):
        """
        Calculate the matrix representation of the spin operator

        For a specific spin operator, its matrix representation in the
        Hilbert space is defined as follow:
            I_{n-1} * ... * I_{i+1} * S_i * I_{i-1} * ... * I_0
        where I is 2 * 2 identity matrix, `*` represents tensor product,
        `n` is the total number of spins and `i` is the index of the lattice
        site.

        Parameters
        ----------
        operator : tuple or list
            Length 2 tuple or list: (index, otype) or [index, otype]
            `index` is the index of the lattice site on which the spin
            operator is defined;
            `otype` is the type of the spin operator which should be only one
            of "x" | "y" | "z" | "p" | "m".
        total_spin : int
            The total number of spins

        Returns
        -------
        res : csr_matrix
            The matrix representation of this spin operator
        """

        index, otype = operator
        I = identity(1 << index, dtype=np.float64, format="csr")
        res = kron(SPIN_MATRICES[otype], I, format="csr")
        I = identity(1 << (total_spin-index-1), dtype=np.float64, format="csr")
        return kron(I, res, format="csr")

    def matrix_repr(self, indices_table):
        """
        Return the matrix representation of this spin operator

        For a specific spin operator, its matrix representation in the
        Hilbert space is defined as follow:
            I_{n-1} * ... * I_{i+1} * S_i * I_{i-1} * ... * I_0
        where I is 2 * 2 identity matrix, `*` represents tensor product,
        `n` is the total number of spins and `i` is the index of the lattice
        site.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of SiteID with an integer index

        Returns
        -------
        res : csr_matrix
            The matrix representation of the spin operator
        """

        total_spin = len(indices_table)
        operator = (self.getSiteIndex(indices_table), self._otype)
        return self.matrix_function(operator, total_spin)


class SpinInteraction:
    """
    A unified description of spin interaction term
    """

    def __init__(self, operators, coeff=1.0):
        """
        Customize the newly created instance

        Parameters
        ----------
        operators : A sequence of SpinOperator objects
        coeff : int, float, complex, optional
            The coefficient of this term
            default: 1.0
        """

        assert isinstance(coeff, NUMERIC_TYPES)

        # Sorting the spin operators in ascending order according to their
        # SiteID. The relative position of two operators with the same SiteID
        # will not change and the exchange of two spin operators on different
        # lattice site never change the interaction term.
        self._operators = tuple(
            sorted(operators, key=lambda item: item.getSiteID())
        )
        self._coeff = coeff

    @property
    def coeff(self):
        """
        The coefficient of this term
        """

        return self._coeff

    @coeff.setter
    def coeff(self, value):
        assert isinstance(value, NUMERIC_TYPES)
        self._coeff = value

    def __str__(self):
        """
        Return a string that describes the content of the instance
        """

        info = "The coefficient of this term: {0}\n".format(self._coeff)
        info += "The component spin operators:\n"
        for operator in self._operators:
            info += "    {0}\n".format(operator)
        return info

    def __mul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` is the left operand and `other'` is the right operand
        Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            operators = self._operators + other._operators
            coeff = self._coeff * other._coeff
        elif isinstance(other, SpinOperator):
            operators = self._operators + (other, )
            coeff = self._coeff
        elif isinstance(other, NUMERIC_TYPES):
            operators = self._operators
            coeff = self._coeff * other
        else:
            return NotImplemented

        return SpinInteraction(operators, coeff=coeff)

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` is the right operand and `other` is the left operand
        This method return a new instance of this class.
        """

        if isinstance(other, SpinOperator):
            operators = (other, ) + self._operators
            coeff = self._coeff
        elif isinstance(other, NUMERIC_TYPES):
            operators = self._operators
            coeff = other * self._coeff
        else:
            return NotImplemented

        return SpinInteraction(operators, coeff=coeff)

    def dagger(self):
        """
        Return the Hermitian conjugate of this term
        """

        operators = tuple(
            operator.dagger() for operator in self._operators[::-1]
        )
        return SpinInteraction(operators, coeff=self._coeff.conjugate())

    def Schwinger(self):
        """
        Return the Schwinger Fermion representation of this term
        """

        fermion_reprs = [operator.Schwinger() for operator in self._operators]
        terms = []
        for term in product(*fermion_reprs):
            res_term = self._coeff
            for sub_term in term:
                res_term = res_term * sub_term
            terms.append(res_term)
        return terms

    @staticmethod
    def matrix_function(operators, total_spin, coeff=1.0):
        """
        Return the matrix representation of the spin interaction term

        Parameters
        ----------
        operators : sequence
            A sequence of 2-tuple: [(index_0, otype_0), ..., (index_n, otype_n)]
            `index_i` is the index of the lattice site on which the spin
            operator is defined;
            `otype_i` is the type of the spin operator which should be only
            one of "x" | "y" | "z" | "p" | "m".
        total_spin: int
            The total number of spins
        coeff : int, float or complex, optional
            The coefficient of the term
            default: 1.0

        Returns
        -------
        res : csr_matrix
            The matrix representation of this term
        """

        assert isinstance(total_spin, int) and total_spin > 0
        assert isinstance(coeff, NUMERIC_TYPES)

        operators = sorted(operators, key=lambda item: item[0])
        if len(operators) == 2 and operators[0][0] != operators[1][0]:
            (index0, otype0), (index1, otype1) = operators
            S0 = coeff * SPIN_MATRICES[otype0]
            S1 = SPIN_MATRICES[otype1]
            dim0 = 1 << index0
            dim1 = 1 << (index1 - index0 - 1)
            dim2 = 1 << (total_spin - index1 - 1)

            if dim1 == 1:
                res = kron(S1, S0, format="csr")
            else:
                I = identity(dim1, dtype=np.float64, format="csr")
                res = kron(S1, kron(I, S0, format="csr"), format="csr")
            if dim0 != 1:
                res = kron(res, identity(dim0, np.float64, "csr"), format="csr")
            if dim2 != 1:
                res = kron(identity(dim2, np.float64, "csr"), res, format="csr")
        else:
            res = coeff * identity(
                1 << total_spin, dtype=np.float64, format="csr"
            )
            for index, otype in operators:
                I = identity(1 << index, dtype=np.float64, format="csr")
                tmp = kron(SPIN_MATRICES[otype], I, format="csr")
                I = identity(
                    1 << (total_spin-index-1), dtype=np.float64, format="csr"
                )
                tmp = kron(I, tmp, format="csr")
                res = res.dot(tmp)
        return res

    def matrix_repr(self, indices_table, coeff=None):
        """
        Return the matrix representation of this spin interaction term

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of SiteID with an integer index
        coeff : int, float or complex, optional
            A new coefficient for this spin interaction term
            If not given or None, use the original coefficient.
            default: None

        Returns
        -------
        res : csr_matrix
            The matrix representation of this spin interaction term
        """

        if coeff is not None:
            self.coeff = coeff

        total_spin = len(indices_table)
        operators = [
            (operator.getSiteIndex(indices_table), operator.otype)
            for operator in self._operators
        ]
        return self.matrix_function(operators, total_spin, self._coeff)


class ParticleTerm:
    """
    A unified description of any term composed of creation and/or
    annihilation operators
    """

    def __init__(self, aocs, coeff=1.0):
        """
        Customize the newly created instance

        Parameters
        ----------
        aocs : tuple or list
            A collection of creation and/or annihilation operators that
            composing this term
        coeff : float, int or complex, optional
            The coefficient of this term
            default: 1.0
        """

        assert isinstance(coeff, NUMERIC_TYPES)

        self._aocs = tuple(aocs)
        self._coeff = coeff

    @property
    def coeff(self):
        """
        The coefficient of this term
        """

        return self._coeff

    @coeff.setter
    def coeff(self, coeff):
        assert isinstance(coeff, NUMERIC_TYPES)
        self._coeff = coeff

    def __str__(self):
        """
        Return a string that describes the content of this instance
        """

        info = "The coefficient of this term: {0}\n".format(self._coeff)
        info += "The component operators:\n"
        for aoc in self._aocs:
            info += "    {0}\n".format(aoc)
        return info

    def __mul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` is the left operand and `other` is the right operand
        Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            aocs = self._aocs + other._aocs
            coeff = self._coeff * other._coeff
        elif isinstance(other, AoC):
            aocs = self._aocs + (other, )
            coeff = self._coeff
        elif isinstance(other, NUMERIC_TYPES):
            aocs = self._aocs
            coeff = self._coeff * other
        else:
            return NotImplemented

        return ParticleTerm(aocs=aocs, coeff=coeff)

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` is the right operand and `other` is the left operand
        This method return a new instance of this class
        """

        if isinstance(other, AoC):
            aocs = (other, ) + self._aocs
            coeff = self._coeff
        elif isinstance(other, NUMERIC_TYPES):
            aocs = self._aocs
            coeff = other * self._coeff
        else:
            return NotImplemented

        return ParticleTerm(aocs=aocs, coeff=coeff)

    @staticmethod
    def normalize(aocs):
        """
        Reordering the given `aocs` into norm form

        For a composite operator consisting of creation and/or annihilation
        operators, the norm form means that all the creation operators appear
        to the left of all the annihilation operators. Also, the creation and
        annihilation operators are sorted in ascending and descending order
        respectively according to the single-particle state associated with
        the operator.

        See the document of __lt__ method of AoC for the comparison logic.

        Parameters
        ----------
        aocs : list or tuple
            A collection of creation and/or annihilation operators

        Returns
        -------
        aocs : list
            The norm form of the operator
        swap_count : int
            The number of swap to obtain the normal form

        Raises
        ------
        SwapError :
            Exceptions raised when swap creation and annihilation operator
            that was defined on the same single-particle state.
        """

        aocs = list(aocs)
        length = len(aocs)
        swap_count = 0

        for remaining_length in range(length, 1, -1):
            for i in range(0, remaining_length - 1):
                aoc0, aoc1 = aocs[i:i+2]
                id0 = aoc0.state
                id1 = aoc1.state
                if aoc0 > aoc1:
                    if id0 != id1:
                        aocs[i] = aoc1
                        aocs[i + 1] = aoc0
                        swap_count += 1
                    else:
                        raise SwapError(aoc0, aoc1)
        return aocs, swap_count

    def dagger(self):
        """
        Return the Hermitian conjugate of this term
        """

        aocs = [aoc.dagger() for aoc in self._aocs[::-1]]
        return ParticleTerm(aocs=aocs, coeff=self._coeff.conjugate())

    def matrix_repr(self, indices_table, right_bases, *, left_bases=None,
                    coeff=None):
        """
        Return the matrix representation of this term

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instance of StateID with an integer index
        right_bases : tuple
            The bases of the Hilbert space before the operation
        left_bases : tuple, keyword-only, optional
            The bases of the Hilbert space after the operation
            It not given or None, left_bases is the same as right_bases.
            default: None
        coeff : int, float or complex, keyword-only, optional
            A new coefficient for this term
            If not given or None, use the original coefficient.
            default: None

        Returns
        -------
        res : csr_matrix
            The matrix representation of the operator in the Hilbert space
        """

        if coeff is not None:
            self.coeff = coeff

        operators = [
            (aoc.getStateIndex(indices_table), aoc.otype) for aoc in self._aocs
        ]
        res = self._coeff * ext.matrix_function(
            operators, right_bases, lbase=left_bases, to_csr=True
        )
        return res
