"""
Components for constructing a model Hamiltonian
"""


__all__ = [
    "SiteID",
    "StateID",
    "AoC",
    "NumberOperator",
    "SpinOperator",
    "SpinInteraction",
    "ParticleTerm",
    "set_float_point_precision",
]


from itertools import product
from scipy.sparse import csr_matrix, identity, kron

import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy.constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP
from HamiltonianPy.matrixrepr import matrix_function


# Useful global constants
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
################################################################################


def set_float_point_precision(precision):
    """
    Set the precision for processing float point number

    The coordinates of a point are treated as float point numbers no matter
    they are given as float-point numbers or integers. The float-point
    precision affects the internal implementation of the SiteID class,
    as well as classes inherited from it, especially the hash values of
    instances of these classes. If you want to change the default value,
    you must call this function before creating any instances of these
    classes. The default value is: `precision = 4`.

    Parameters
    ----------
    precision : int
        The number of digits precision after the decimal point
    """

    assert isinstance(precision, int) and precision >= 0

    global  ZOOM
    ZOOM = 10 ** precision


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
    >>> from HamiltonianPy.termofH import SiteID
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
        site : 1D np.ndarray
            The coordinate of the lattice site
            The shape of this array should be (1,), (2,) or (3,).
        """

        assert isinstance(site, np.ndarray) and site.shape in ((1,), (2,), (3,))

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
            A table that associate instances of SiteID with integer indices

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
    >>> from HamiltonianPy.termofH import StateID
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
        site : 1D np.ndarray
            The coordinate of the localized single-particle state
            The `site` parameter should be 1D array with length 1, 2 or 3.
        spin : int, optional
            The spin index of the single-particle state
            default: 0
        orbit : int, optional
            The orbit index of the single-particle state
            default: 0
        """

        assert isinstance(spin, int) and spin >= 0
        assert isinstance(orbit, int) and orbit >= 0

        super().__init__(site=site)
        self._spin = spin
        self._orbit = orbit

        # The self._tuple_form on the right hand has already been set properly
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
    >>> from HamiltonianPy.termofH import AoC
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
        site : 1D np.ndarray
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
            A table that associate instances of AoC with integer indices

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
            A table that associate instances of StateID with integer indices

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

    def _tex(self, indices_table=None):
        # Convert the instance to TeX string
        # `indices_table` is a table that associate instances of SiteID with
        # integer indices

        if indices_table is None:
            site = "(" + ", ".join("{:.4f}".format(i) for i in self.site) + ")"
        else:
            site = SiteID(site=self.site).getIndex(indices_table)
        subscript = "{0},{1},{2}".format(site, self.spin, self.orbit)

        # Format strings contain "replacement fields" surrounded by curly
        # braces `{}`. Anything that is not contained in braces is considered
        # literal text, which is copied unchanged to the output. If you need
        # to include a brace character in the literal text, it can be
        # escaped by doubling: `{{` and `}}`
        if self._otype == CREATION:
            tex = r"$c_{{{0}}}^{{\dag}}$".format(subscript)
        else:
            tex = r"$c_{{{0}}}$".format(subscript)
        return tex

    def show(self, indices_table=None):
        """
        Show the instance in handwriting form

        Parameters
        ----------
        indices_table : IndexTable, optional
            A table that associate instances of SiteID with integer indices
            If not given or None, the `site` is show as it is
            default : None
        """

        fig, ax = plt.subplots()
        ax.set_axis_off()
        tex = self._tex(indices_table)
        ax.text(
            0.5, 0.5, tex, fontname="monospace", fontsize=30,
            ha="center", va="center", transform=ax.transAxes
        )
        plt.show()

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

    def matrix_repr(
            self, state_indices_table, right_bases, *,
            left_bases=None, to_csr=True, threads_num=1
    ):
        """
        Return the matrix representation of this operator in the Hilbert space

        Parameters
        ----------
        state_indices_table : IndexTable
            A table that associate instances of StateID with integer indices
        right_bases : 1D np.ndarray
            The bases of the Hilbert space before the operation
        left_bases : 1D np.ndarray, keyword-only, optional
            The bases of the Hilbert space after the operation
            If not given or None, left_bases is the same as right_bases
            default: None
        to_csr : boolean, keyword-only, optional
            Whether to construct a csr_matrix as the result
            default: True
        threads_num : int, keyword-only, optional
            The number of threads to use for calculating the matrix
            representation
            default: 1

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of the operator in the Hilbert space
            If `to_csr` is set to True, the result is a csr_matrix;
            If set to False, the result is a tuple: (entries, (rows, cols)),
            where `entries` is the non-zero matrix elements, `rows` and
            `cols` are the row and column indices of the none-zero elements.
        """

        term = [(self.getStateIndex(state_indices_table), self._otype)]
        return matrix_function(
            term, right_bases, left_bases=left_bases,
            to_csr=to_csr, threads_num=threads_num
        )


class NumberOperator:
    """
    A unified description of the particle-number operator

    Attributes
    ----------
    state : StateID
        The single-particle state on which this operator is defined
    site : 1D np.ndarray
        The coordinate of the localized single-particle state
    spin : int
        The spin index of the single-particle state
    orbit : int
        The orbit index of the single-particle state

    Examples
    --------
    >>> import numpy as np
    >>> from HamiltonianPy.termofH import NumberOperator
    >>> NumberOperator(site=np.array([0, 0]), spin=0)
    NumberOperator(site=array([0, 0]), spin=0, orbit=0)
    """

    def __init__(self, site, spin=0, orbit=0):
        """
        Customize the newly created instance

        Parameters
        ----------
        site : 1D np.ndarray
            The coordinate of the localized single-particle state
            The `site` parameter should be 1D array with length 1, 2 or 3.
        spin : int, optional
            The spin index of the single-particle state
            default: 0
        orbit : int, optional
            The orbit index of the single-particle state
            default: 0
        """

        state = StateID(site=site, spin=spin, orbit=orbit)
        self._state = state

        # The tuple form of this instance
        # It is a tuple: ("N", (site, spin, orbit)) and site itself is a
        # tuple with length 1, 2 or 3.
        self._tuple_form = ("N", state._tuple_form)

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
            A table that associate instances of NumberOperator with integer
            indices

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
        Return the index of the single-particle state on which this operator
        is defined

        Notes:
            This method is different from the `getIndex` method.
            This method return the index of the `state` attribute of the
            operator and the `getIndex` method return the index of the
            operator itself.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instances of StateID with integer indices

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

        info = "NumberOperator(site={0!r}, spin={1}, orbit={2})"
        return info.format(self.site, self.spin, self.orbit)

    __str__ = __repr__

    def tolatex(self, site_index=None):
        """
        Return the LaTex form of this operator

        Parameters
        ----------
        site_index : int, optional
            The index of the `site` attribute.
            If not given or None, the `site` attribute is shown as it is;
            If given, the `site` attribute is replaced with the given
            site_index.
            default: None

        Returns
        -------
        res : str
            The LaTex form of this operator
        """

        if site_index is None:
            site = "(" + ",".join("{:.4f}".format(i) for i in self.site) + ")"
        else:
            site = site_index
        subscript = "{0},{1},{2}".format(site, self.spin, self.orbit)
        str_latex = r"$n_{{{0}}}$".format(subscript)
        return str_latex

    def show(self, site_index=None):
        """
        Show the operator in handwriting form

        Parameters
        ----------
        site_index : int, optional
            The index of the `site` attribute.
            If not given or None, the `site` attribute is shown as it is;
            If given, the `site` attribute is replaced with the given
            site_index.
            default: None
        """

        fig, ax = plt.subplots()
        ax.set_axis_off()
        str_latex = self.tolatex(site_index)
        ax.text(
            0.5, 0.5, str_latex, fontname="monospace", fontsize="xx-large",
            ha="center", va="center", transform=ax.transAxes
        )
        plt.show()

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

    def __gt__(self, other):
        """
        Define the `>` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form > other._tuple_form
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
        """

        if isinstance(other, self.__class__):
            return self._tuple_form <= other._tuple_form
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

    def toterm(self):
        """
        Convert this operator to ParticleTerm instance

        Returns
        -------
        term : ParticleTerm
            The term corresponding to this operator
        """

        C = AoC(CREATION, site=self.site, spin=self.spin, orbit=self.orbit)
        A = AoC(ANNIHILATION, site=self.site, spin=self.spin, orbit=self.orbit)
        return ParticleTerm((C, A), coeff=1.0)

    def __mul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` is the left operand and `other` is the right operand
        Return an instance of ParticleTerm
        """

        return self.toterm() * other

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`

        `self` is the right operand and `other` is the left operand
        Return an instance of ParticleTerm
        """

        return other * self.toterm()

    def dagger(self):
        """
        Return the Hermitian conjugate of this operator

        Returns
        -------
        res : NumberOperator
            The Hermitian conjugate of this operator
        """

        return self

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

    def derive(self, *, site=None, spin=None, orbit=None):
        """
        Derive a new instance from `self` and the given parameters

        This method creates a new instance with the same attribute as `self`
        except for these given to this method.
        All the parameters should be specified as keyword arguments.

        Returns
        -------
        res : A new instance of NumberOperator
        """

        if site is None:
            site = self.site
        if spin is None:
            spin = self.spin
        if orbit is None:
            orbit = self.orbit
        return NumberOperator(site=site, spin=spin, orbit=orbit)

    def matrix_repr(
            self, state_indices_table, bases, *, to_csr=True,threads_num=1
    ):
        """
        Return the matrix representation of this operator in the Hilbert space

        Parameters
        ----------
        state_indices_table : IndexTable
            A table that associate instances of StateID with integer indices
        bases : 1D np.ndarray
            The bases of the Hilbert space
        to_csr : boolean, keyword-only, optional
            Whether to construct a csr_matrix as the result
            default: True
        threads_num : int, keyword-only, optional
            The number of threads to use for calculating the matrix
            representation
            default: 1

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of the operator in the Hilbert space
            If `to_csr` is set to True, the result is a csr_matrix;
            If set to False, the result is a tuple: (entries, (rows, cols)),
            where `entries` is the non-zero matrix elements, `rows` and
            `cols` are the row and column indices of the none-zero elements.
        """

        index = self.getStateIndex(state_indices_table)
        term = [(index, CREATION), (index, ANNIHILATION)]
        return matrix_function(
            term, bases, to_csr=to_csr, threads_num=threads_num
        )


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

    Examples
    --------
    >>> import numpy as np
    >>> from HamiltonianPy.termofH import SpinOperator
    >>> SX = SpinOperator("x", site=np.array([0, 0]))
    >>> SY = SpinOperator("y", site=np.array([1, 1]))
    >>> SX
    SpinOperator(otype="x", site=array([0, 0]))
    >>> SY.matrix()
    array([[ 0.+0.j , -0.-0.5j],
           [ 0.+0.5j,  0.+0.j ]])
    >>> SY < SX
    False
    >>> print(2 * SX * SY)
    The coefficient of this term: 2
    The component spin operators:
        SpinOperator(otype="x", site=array([0, 0]))
        SpinOperator(otype="y", site=array([1, 1]))
    """

    def __init__(self, otype, site):
        """
        Customize the newly created instance

        Parameters
        ----------
        otype : str
            The type of this spin operator
            Valid value: "x" | "y" | "z" | "p" | "m"
        site : 1D np.ndarray
            The coordinate of the lattice site on which the spin operator is
            defined. The `site` parameter should be 1D array with shape (1,),
            (2,) or (3,).
        """

        assert otype in SPIN_OTYPES

        super().__init__(site=site)
        self._otype = otype

        # The self._tuple_form on the right hand has already been set properly
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
            A table that associate instances of SiteID with integer indices

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

    def _tex(self, indices_table=None):
        # Convert the instance to TeX string
        # `indices_table` is a table that associate instance of SiteID with an
        # integer index

        if indices_table is None:
            site = "(" + ", ".join("{:.4f}".format(i) for i in self._site) + ")"
        else:
            site = self.getSiteIndex(indices_table)
        tex = r"$\mathrm{{S}}_{{{0}}}^{{{1}}}$".format(site, self._otype)
        return tex

    def show(self, indices_table=None):
        """
        Show the instance in handwriting form

        Parameters
        ----------
        indices_table : IndexTable, optional
            A table that associate instances of SiteID with integer indices
            If not given or None, the `site` is show as it is
            default : None
        """

        fig, ax = plt.subplots()
        ax.set_axis_off()
        tex = self._tex(indices_table)
        ax.text(
            0.5, 0.5, tex, fontname="monospace", fontsize=30,
            ha="center", va="center", transform=ax.transAxes
        )
        plt.show()

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

        terms = []
        SMatrix = self.matrix()
        for row_index, row_aoc in enumerate((C_UP, C_DOWN)):
            for col_index, col_aoc in enumerate((A_UP, A_DOWN)):
                coeff = SMatrix[row_index, col_index]
                if coeff != 0.0:
                    terms.append(ParticleTerm([row_aoc, col_aoc], coeff=coeff))
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
            A table that associate instances of SiteID with integer indices

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

        assert isinstance(coeff, NUMERIC_TYPES), "Invalid coefficient"

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
        assert isinstance(value, NUMERIC_TYPES), "Invalid coefficient"
        self._coeff = value

    @property
    def components(self):
        """
        The component spin operators of this term
        """

        return self._operators

    def __str__(self):
        """
        Return a string that describes the content of the instance
        """

        info = [
            "The coefficient of this term: {0}".format(self._coeff),
            "The component spin operators:",
        ]
        for operator in self._operators:
            info.append("    {0}".format(operator))
        return "\n".join(info)

    def _tex(self, indices_table=None):
        # Convert the instance to TeX string
        # `indices_table` is a table that associate instances of SiteID with
        # integer indices

        if indices_table is None:
            tex = "coeff = {0:.4f}\n".format(self._coeff)
            tex += "\n".join(operator._tex() for operator in self._operators)
        else:
            tex = "{0:.4f} ".format(self._coeff)
            tex += "".join(
                operator._tex(indices_table) for operator in self._operators
            )
        return tex

    def show(self, indices_table=None):
        """
        Show the instance in handwriting form

        Parameters
        ----------
        indices_table : IndexTable, optional
            A table that associate instances of SiteID with integer indices
            If not given or None, the `site` is show as it is
            default : None
        """

        fig, ax = plt.subplots()
        ax.set_axis_off()
        tex = self._tex(indices_table)
        ax.text(
            0.5, 0.5, tex, fontname="monospace", fontsize=30,
            ha="center", va="center", transform=ax.transAxes
        )
        plt.show()

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
        assert isinstance(coeff, NUMERIC_TYPES), "Invalid coefficient"

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
            A table that associate instances of SiteID with integer indices
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

        assert isinstance(coeff, NUMERIC_TYPES), "Invalid coefficient"

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
        assert isinstance(coeff, NUMERIC_TYPES), "Invalid coefficient"
        self._coeff = coeff

    @property
    def components(self):
        """
        The component creation and/or annihilation operators of this term
        """

        return self._aocs

    def __str__(self):
        """
        Return a string that describes the content of this instance
        """

        info = [
            "The coefficient of this term: {0}".format(self._coeff),
            "The component operators:",
        ]
        for aoc in self._aocs:
            info.append("    {0}".format(aoc))
        return "\n".join(info)

    def _tex(self, indices_table=None):
        # Convert the instance to TeX string
        # `indices_table` is a table that associate instances of SiteID with
        # integer indices

        if indices_table is None:
            tex = "coeff = {0:.4f}\n".format(self._coeff)
            tex += "\n".join(aoc._tex() for aoc in self._aocs)
        else:
            tex = "{0:.4f} ".format(self._coeff)
            tex += "".join(aoc._tex(indices_table) for aoc in self._aocs)
        return tex

    def show(self, indices_table=None):
        """
        Show the instance in handwriting form

        Parameters
        ----------
        indices_table : IndexTable, optional
            A table that associate instances of SiteID with integer indices
            If not given or None, the `site` is show as it is
            default : None
        """

        fig, ax = plt.subplots()
        ax.set_axis_off()
        tex = self._tex(indices_table)
        ax.text(
            0.5, 0.5, tex, fontname="monospace", fontsize=30,
            ha="center", va="center", transform=ax.transAxes
        )
        plt.show()

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

    def matrix_repr(
            self, state_indices_table, right_bases, *,
            left_bases=None, coeff=None, to_csr=True, threads_num=1
    ):
        """
        Return the matrix representation of this term

        Parameters
        ----------
        state_indices_table : IndexTable
            A table that associate instances of StateID with integer indices
        right_bases : 1D np.ndarray
            The bases of the Hilbert space before the operation
        left_bases : 1D np.ndarray, keyword-only, optional
            The bases of the Hilbert space after the operation
            It not given or None, left_bases is the same as right_bases.
            default: None
        coeff : int, float or complex, keyword-only, optional
            A new coefficient for this term
            If not given or None, use the original coefficient.
            default: None
        to_csr : boolean, keyword-only, optional
            Whether to construct a csr_matrix as the result
            default: True
        threads_num : int, keyword-only, optional
            The number of threads to use for calculating the matrix
            representation
            default: 1

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of the operator in the Hilbert space
            If `to_csr` is set to True, the result is a csr_matrix;
            If set to False, the result is a tuple: (entries, (rows, cols)),
            where `entries` is the non-zero matrix elements, `rows` and
            `cols` are the row and column indices of the none-zero elements.
        """

        if coeff is not None:
            self.coeff = coeff

        operators = [
            (aoc.getStateIndex(state_indices_table), aoc.otype)
            for aoc in self._aocs
        ]
        return matrix_function(
            operators, right_bases, left_bases=left_bases,
            coeff=self._coeff, to_csr=to_csr, threads_num=threads_num
        )


def CPFactory(site, *, spin=0, orbit=0, coeff=1.0):
    """
    Generate chemical potential term: `coeff * c_i^{\dagger} c_i`

    Parameters
    ----------
    site : 1D np.ndarray
        The coordinate of the localized single-particle state
        The `site` parameter should be 1D array with length 1,2 or 3.
    spin : int, keyword-only, optional
        The spin index of the single-particle state
        default: 0
    orbit : int, keyword-only, optional
        The orbit index of the single-particle state
        default: 0
    coeff : int or float, keyword-only, optional
        The coefficient of this term

    Returns
    -------
    res : ParticleTerm
        The corresponding chemical potential term
    """

    C = AoC(CREATION, site=site, spin=spin, orbit=orbit)
    A = AoC(ANNIHILATION, site=site, spin=spin, orbit=orbit)
    return ParticleTerm((C, A), coeff=coeff)


def HoppingFactory(
        site0, site1, *, spin0=0, spin1=None, orbit0=0, orbit1=None, coeff=1.0
):
    """
    Generate hopping term: `coeff * c_i^{\dagger} c_j`

    These parameters ended with "0" are for the creation operator and "1" for
    annihilation operator.

    Parameters
    ----------
    site0, site1 : 1D np.ndarray
        The coordinate of the localized single-particle state
        `site0` and `site1` should be 1D array with length 1, 2 or 3.
    spin0, spin1 : int, keyword-only, optional
        The spin index of the single-particle state
        The default value for `spin0` is 0;
        The default value for `spin1` is None, which implies that `spin1`
        takes the same value as `spin0`.
    orbit0, orbit1 : int, keyword-only, optional
        The orbit index of the single-particle state
        The default value for `orbit0` is 0;
        The default value for `orbit1` is None, which implies that `orbit1`
        takes the same value as `orbit0`.
    coeff : int, float or complex, keyword-only, optional
        The coefficient of this term

    Returns
    -------
    res : ParticleTerm
        The corresponding hopping term
    """

    if spin1 is None:
        spin1 = spin0
    if orbit1 is None:
        orbit1 = orbit0

    C = AoC(CREATION, site=site0, spin=spin0, orbit=orbit0)
    A = AoC(ANNIHILATION, site=site1, spin=spin1, orbit=orbit1)
    return ParticleTerm((C, A), coeff=coeff)


def PairingFactory(
        site0, site1, *, spin0=0, spin1=0, orbit0=0, orbit1=0,
        coeff=1.0, which="h"
):
    """
    Generate pairing term:
        `coeff * c_i^{\dagger} c_j^{\dagger}` or `coeff * c_i c_j`

    These parameters ended with "0" are for the first operator and "1" for
    second operator.

    Parameters
    ----------
    site0, site1 : 1D np.ndarray
        The coordinate of the localized single-particle state
        `site0` and `site1` should be 1D array with length 1, 2 or 3.
    spin0, spin1 : int, keyword-only, optional
        The spin index of the single-particle state
        default: 0
    orbit0, orbit1 : int, keyword-only, optional
        The orbit index of the single-particle state
        default: 0
    coeff : int, float or complex, keyword-only, optional
        The coefficient of this term
    which : str, keyword-only, optional
        Determine whether to generate a particle- or hole-pairing term
        Valid values:
            ["h" | "hole"] for hole-pairing
            ["p" | "particle"] for particle-pairing
        default: "h"

    Returns
    -------
    res : ParticleTerm
        The corresponding pairing term
    """

    assert which in ("h", "hole", "p", "particle")

    otype = ANNIHILATION if which in ("h", "hole") else CREATION
    aoc0 = AoC(otype, site=site0, spin=spin0, orbit=orbit0)
    aoc1 = AoC(otype, site=site1, spin=spin1, orbit=orbit1)
    return ParticleTerm((aoc0, aoc1), coeff=coeff)


def HubbardFactory(site, *, orbit=0, coeff=1.0):
    """
    Generate Hubbard term: `coeff * n_{i,up} n_{i,down}`

    This function is valid only for spin-1/2 system.

    Parameters
    ----------
    site : 1D np.ndarray
        The coordinate of the localized single-particle state
        `site` should be 1D array with length 1,2 or 3.
    orbit : int, keyword-only, optional
        The orbit index of the single-particle state
        default: 0
    coeff : int or float, keyword-only, optional
        The coefficient of this term

    Returns
    -------
    res : ParticleTerm
        The corresponding Hubbard term
    """

    C_UP = AoC(CREATION, site=site, spin=SPIN_UP, orbit=orbit)
    C_DOWN = AoC(CREATION, site=site, spin=SPIN_DOWN, orbit=orbit)
    A_UP = AoC(ANNIHILATION, site=site, spin=SPIN_UP, orbit=orbit)
    A_DOWN = AoC(ANNIHILATION, site=site, spin=SPIN_DOWN, orbit=orbit)
    return ParticleTerm((C_UP, A_UP, C_DOWN, A_DOWN), coeff=coeff)


def CoulombFactory(
        site0, site1, *, spin0=0, spin1=0, orbit0=0, orbit1=0, coeff=1.0
):
    """
    Generate Coulomb interaction term: `coeff * n_i n_j`

    These parameters ended with "0" are for the first operator and "1" for
    second operator.

    Parameters
    ----------
    site0, site1 : 1D np.ndarray
        The coordinate of the localized single-particle state
        `site0` and `site1` should be 1D array with length 1, 2 or 3.
    spin0, spin1 : int, keyword-only, optional
        The spin index of the single-particle state
        default: 0
    orbit0, orbit1 : int, keyword-only, optional
        The orbit index of the single-particle state
        default: 0
    coeff : int or float, keyword-only, optional
        The coefficient of this term

    Returns
    -------
    res : ParticleTerm
        The corresponding Coulomb interaction term
    """

    C0 = AoC(CREATION, site=site0, spin=spin0, orbit=orbit0)
    A0 = AoC(ANNIHILATION, site=site0, spin=spin0, orbit=orbit0)
    C1 = AoC(CREATION, site=site1, spin=spin1, orbit=orbit1)
    A1 = AoC(ANNIHILATION, site=site1, spin=spin1, orbit=orbit1)
    return ParticleTerm((C0, A0, C1, A1), coeff=coeff)
