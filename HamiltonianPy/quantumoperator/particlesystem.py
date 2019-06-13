"""
This module provides classes that describe creation/annihilation operator as
well as term composed of creation and/or annihilation operators.
"""


__all__ = [
    "AoC",
    "NumberOperator",
    "ParticleTerm",
]


import matplotlib.pyplot as plt

from HamiltonianPy.quantumoperator.constant import ANNIHILATION, CREATION, \
    NUMERIC_TYPES_GENERAL
from HamiltonianPy.quantumoperator.matrixrepr import matrix_function
from HamiltonianPy.quantumoperator.quantumstate import StateID


class AoC:
    """
    A unified description of the creation and annihilation operator.

    Attributes
    ----------
    otype : int
        The type of this operator. It can be either 0 or 1, corresponding to
        annihilation and creation respectively.
    state : StateID
        The single-particle state on which this operator is defined.
    coordinate : tuple
        The coordinates of the localized single-particle state in tuple form.
    site : 1D np.ndarray
        The coordinates of the localized single-particle state in np.ndarray
        form.
    spin : int
        The spin index of the single-particle state.
    orbit : int
        The orbit index of the single-particle state.

    Examples
    --------
    >>> from HamiltonianPy.quantumoperator import AoC
    >>> c = AoC(otype=1, site=[0, 0], spin=0)
    >>> a = AoC(otype=0, site=(0.3, 0.75), spin=1)
    >>> c
    AoC(otype=CREATION, site=(0, 0), spin=0, orbit=0)
    >>> a
    AoC(otype=ANNIHILATION, site=(0.3, 0.75), spin=1, orbit=0)
    >>> c.tolatex()
    '$c_{(0,0),\\\\downarrow}^{\\\\dagger}$'
    >>> a.tolatex()
    '$c_{(0.3,0.75),\\\\uparrow}$'
    >>> c < a
    True
    >>> c.dagger()
    AoC(otype=ANNIHILATION, site=(0, 0), spin=0, orbit=0)
    >>> a.dagger()
    AoC(otype=CREATION, site=(0.3, 0.75), spin=1, orbit=0)
    >>> print(2 * c * a)
    The coefficient of this term: 2
    The component operators:
        AoC(otype=CREATION, site=(0, 0), spin=0, orbit=0)
        AoC(otype=ANNIHILATION, site=(0.3, 0.75), spin=1, orbit=0)
    >>> print(0.5 * c)
    The coefficient of this term: 0.5
    The component operators:
        AoC(otype=CREATION, site=(0, 0), spin=0, orbit=0)
    >>> print(a * (1+2j))
    The coefficient of this term: (1+2j)
    The component operators:
        AoC(otype=ANNIHILATION, site=(0.3, 0.75), spin=1, orbit=0)
    """

    def __init__(self, otype, site, spin=0, orbit=0):
        """
        Customize the newly created instance.

        Parameters
        ----------
        otype : int
            The type of this operator.
            It can be either 0 or 1, corresponding to annihilation and
            creation respectively. It is recommended to use the constants
            `CREATION` and `ANNIHILATION` defined in the `constant` module.
        site : list, tuple or 1D np.ndarray
            The coordinates of the localized single-particle state.
            The `site` parameter should be 1D array with length 1,2 or 3.
        spin : int, optional
            The spin index of the single-particle state.
            Default: 0.
        orbit : int, optional
            The orbit index of the single-particle state.
            Default: 0.
        """

        assert otype in (ANNIHILATION, CREATION)

        state = StateID(site=site, spin=spin, orbit=orbit)
        self._otype = otype
        self._state = state
        # The tuple form of this instance
        # It is a tuple: (otype, (site, spin, orbit)) and site itself is a
        # tuple with length 1, 2, or 3.
        self._tuple_form = (otype, state._tuple_form)

    @property
    def otype(self):
        """
        The `otype` attribute.
        """

        return self._otype

    @property
    def state(self):
        """
        The `state` attribute.
        """

        return self._state

    @property
    def coordinate(self):
        """
        The `coordinate` attribute.
        """

        return self._state.coordinate

    @property
    def site(self):
        """
        The `site` attribute.
        """

        return self._state.site

    @property
    def spin(self):
        """
        The `spin` attribute.
        """

        return self._state.spin

    @property
    def orbit(self):
        """
        The `orbit` attribute.
        """

        return self._state.orbit

    def getIndex(self, indices_table):
        """
        Return the index of this operator.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instances of AoC with integer indices.

        Returns
        -------
        index : int
            The index of this instance in the given table.

        See also
        --------
        getStateIndex
        """

        return indices_table(self)

    def getStateIndex(self, indices_table):
        """
        Return the index of the single-particle state on which this operator is
        defined.

        Notes:
            This method is different from the `getIndex` method.
            This method return the index of the `state` attribute of the
            operator and the `getIndex` method return the index of the
            operator itself.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instances of StateID with integer indices.

        Returns
        -------
        index : int
            The index of the `state` attribute in the given table.
        """

        return indices_table(self._state)

    def __repr__(self):
        """
        Official string representation of the instance.
        """

        otype = "CREATION" if self._otype == CREATION else "ANNIHILATION"
        info = "AoC(otype={0}, site={1!r}, spin={2}, orbit={3})"
        return info.format(otype, self.coordinate, self.spin, self.orbit)

    __str__ = __repr__

    def tolatex(self, **kwargs):
        """
        Return the LaTex form of this instance.

        Parameters
        ----------
        kwargs :
            All keyword arguments are passed to the `tolatex` method of the
            `state` attribute.
            See also: `StateID.tolatex`.

        Returns
        -------
        latex : str
            The LaTex form of this instance.
        """

        subscript = self._state.tolatex(**kwargs).replace("$", "")
        if self._otype == CREATION:
            latex_form = r"$c_{{{0}}}^{{\dagger}}$".format(subscript)
        else:
            latex_form = r"$c_{{{0}}}$".format(subscript)
        return latex_form

    def show(self, **kwargs):
        """
        Show the instance in handwriting form.

        Parameters
        ----------
        kwargs :
            All keyword arguments are passed to the `tolatex` method of the
            `state` attribute.
            See also: `StateID.tolatex`.
        """

        fig, ax = plt.subplots()
        ax.text(
            0.5, 0.5, self.tolatex(**kwargs), fontsize="xx-large",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        plt.show()

    def __hash__(self):
        """
        Calculate the hash code of the instance.
        """

        return hash(self._tuple_form)

    def __lt__(self, other):
        """
        Implement the `<` operator between self and other.

        The comparison logic is as follow:
        Creation operator is always compare less than annihilation operator;
        The smaller the single-particle state, the smaller the creation
        operator; The larger the single-particle state, the smaller the
        annihilation operator.

        See also
        --------
        StateID.__lt__
        StateID.__gt__
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
        Implement the `>` operator between self and other.

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
        Implement the `==` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form == other._tuple_form
        else:
            return NotImplemented

    def __ne__(self, other):
        """
        Implement the `!=` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form != other._tuple_form
        else:
            return NotImplemented

    def __le__(self, other):
        """
        Implement the `<=` operator between self and other.

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
        Implement the `>=` operator between self and other.

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
        Implement the binary arithmetic operation: `*`.

        `self` is the left operand and `other` is the right operand;
        Return an instance of ParticleTerm.
        """

        if isinstance(other, self.__class__):
            return ParticleTerm((self, other), coeff=1.0)
        elif isinstance(other, NUMERIC_TYPES_GENERAL):
            return ParticleTerm((self,), coeff=other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`.

        `self` is the right operand and `other` is the left operand;
        Return an instance of ParticleTerm.
        """

        if isinstance(other, NUMERIC_TYPES_GENERAL):
            return ParticleTerm((self,), coeff=other)
        else:
            return NotImplemented

    def dagger(self):
        """
        Return the Hermitian conjugate of this operator.
        """

        otype = ANNIHILATION if self._otype == CREATION else CREATION
        return self.derive(otype=otype)

    def conjugate_of(self, other):
        """
        Determine whether `self` is the Hermitian conjugate of `other`.
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
        single-particle state.
        """

        if isinstance(other, self.__class__):
            return self._state == other._state
        else:
            raise TypeError(
                "The `other` parameter is not instance of this class!"
            )

    def derive(self, *, otype=None, site=None, spin=None, orbit=None):
        """
        Derive a new instance from `self` and the given parameters.

        This method creates a new instance with the same attribute as `self`
        except for these given to this method.
        All the parameters should be specified as keyword arguments.

        Returns
        -------
        res : A new instance of AoC.
        """

        if otype is None:
            otype = self.otype
        if site is None:
            site = self.coordinate
        if spin is None:
            spin = self.spin
        if orbit is None:
            orbit = self.orbit

        return self.__class__(otype=otype, site=site, spin=spin, orbit=orbit)

    def matrix_repr(
            self, state_indices_table, right_bases, *,
            left_bases=None, to_csr=True
    ):
        """
        Return the matrix representation of this operator in the Hilbert space.

        Parameters
        ----------
        state_indices_table : IndexTable
            A table that associate instances of StateID with integer indices.
        right_bases : 1D np.ndarray
            The bases of the Hilbert space before the operation.
            The data-type of the array's elements is np.uint64.
        left_bases : 1D np.ndarray, optional, keyword-only
            The bases of the Hilbert space after the operation.
            If given, the data-type of the array's elements is np.uint64.
            If not given or None, left_bases is the same as right_bases.
            Default: None.
        to_csr : bool, optional, keyword-only
            Whether to construct a csr_matrix as the result.
            Default: True.

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of the operator in the Hilbert space.
            If `to_csr` is set to True, the result is a csr_matrix;
            If set to False, the result is a tuple: (entries, (rows, cols)),
            where `entries` is the non-zero matrix elements, `rows` and
            `cols` are the row and column indices of the none-zero elements.
        """

        term = [(state_indices_table(self._state), self._otype)]
        return matrix_function(
            term, right_bases, left_bases=left_bases, to_csr=to_csr
        )


class NumberOperator:
    """
    A unified description of the particle-number operator.

    Attributes
    ----------
    state : StateID
        The single-particle state on which this operator is defined.
    coordinate : tuple
        The coordinates of the localized single-particle state in tuple form.
    site : 1D np.ndarray
        The coordinates of the localized single-particle state in np.ndarray
        form.
    spin : int
        The spin index of the single-particle state.
    orbit : int
        The orbit index of the single-particle state.

    Examples
    --------
    >>> from HamiltonianPy.quantumoperator import NumberOperator
    >>> N0 = NumberOperator(site=[0, 0], spin=0)
    >>> N1 = NumberOperator(site=(0.3, 0.75), spin=1)
    >>> N0
    NumberOperator(site=(0, 0), spin=0, orbit=0)
    >>> N1
    NumberOperator(site=(0.3, 0.75), spin=1, orbit=0)
    >>> N0.tolatex()
    '$n_{(0,0),\\\\downarrow}$'
    >>> N1.tolatex()
    '$n_{(0.3,0.75),\\\\uparrow}$'
    >>> N0 < N1
    True
    >>> N0.dagger() is N0
    True
    >>> print(N0.toterm())
    The coefficient of this term: 1.0
    The component operators:
        AoC(otype=CREATION, site=(0, 0), spin=0, orbit=0)
        AoC(otype=ANNIHILATION, site=(0, 0), spin=0, orbit=0)
    >>> print(N0 * N1 * 1.5)
    The coefficient of this term: 1.5
    The component operators:
        AoC(otype=CREATION, site=(0, 0), spin=0, orbit=0)
        AoC(otype=ANNIHILATION, site=(0, 0), spin=0, orbit=0)
        AoC(otype=CREATION, site=(0.3, 0.75), spin=1, orbit=0)
        AoC(otype=ANNIHILATION, site=(0.3, 0.75), spin=1, orbit=0)
    """

    def __init__(self, site, spin=0, orbit=0):
        """
        Customize the newly created instance.

        Parameters
        ----------
        site : list, tuple or 1D np.ndarray
            The coordinates of the localized single-particle state.
            The `site` parameter should be 1D array with length 1, 2 or 3.
        spin : int, optional
            The spin index of the single-particle state.
            Default: 0.
        orbit : int, optional
            The orbit index of the single-particle state.
            Default: 0.
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
        The `state` attribute.
        """

        return self._state

    @property
    def coordinate(self):
        """
        The `coordinate` attribute.
        """

        return self._state.coordinate

    @property
    def site(self):
        """
        The `site` attribute.
        """

        return self._state.site

    @property
    def spin(self):
        """
        The `spin` attribute.
        """

        return self._state.spin

    @property
    def orbit(self):
        """
        The `orbit` attribute.
        """

        return self._state.orbit

    def getIndex(self, indices_table):
        """
        Return the index of this operator.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instances of NumberOperator with integer
            indices.

        Returns
        -------
        index : int
            The index of this instance in the given table.

        See also
        --------
        getStateIndex
        """

        return indices_table(self)

    def getStateIndex(self, indices_table):
        """
        Return the index of the single-particle state on which this operator
        is defined.

        Notes:
            This method is different from the `getIndex` method.
            This method return the index of the `state` attribute of the
            operator and the `getIndex` method return the index of the
            operator itself.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instances of StateID with integer indices.

        Returns
        -------
        index : int
            The index of the `state` attribute in the given table.
        """

        return indices_table(self._state)

    def __repr__(self):
        """
        Official string representation of the instance.
        """

        info = "NumberOperator(site={0!r}, spin={1}, orbit={2})"
        return info.format(self.coordinate, self.spin, self.orbit)

    __str__ = __repr__

    def tolatex(self, **kwargs):
        """
        Return the LaTex form of this instance.

        Parameters
        ----------
        kwargs :
            All keyword arguments are passed to the `tolatex` method of the
            `state` attribute.
            See also: `StateID.tolatex`.

        Returns
        -------
        latex : str
            The LaTex form of this instance.
        """

        subscript = self._state.tolatex(**kwargs).replace("$", "")
        return r"$n_{{{0}}}$".format(subscript)

    def show(self, **kwargs):
        """
        Show the instance in handwriting form.

        Parameters
        ----------
        kwargs :
            All keyword arguments are passed to the `tolatex` method of the
            `state` attribute.
            See also: `StateID.tolatex`.
        """

        fig, ax = plt.subplots()
        ax.text(
            0.5, 0.5, self.tolatex(**kwargs), fontsize="xx-large",
            ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        plt.show()

    def __hash__(self):
        """
        Calculate the hash code of the instance.
        """

        return hash(self._tuple_form)

    def __lt__(self, other):
        """
        Implement the `<` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form < other._tuple_form
        else:
            return NotImplemented

    def __gt__(self, other):
        """
        Implement the `>` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form > other._tuple_form
        else:
            return NotImplemented

    def __eq__(self, other):
        """
        Implement the `==` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form == other._tuple_form
        else:
            return NotImplemented

    def __ne__(self, other):
        """
        Implement the `!=` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form != other._tuple_form
        else:
            return NotImplemented

    def __le__(self, other):
        """
        Implement the `<=` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form <= other._tuple_form
        else:
            return NotImplemented

    def __ge__(self, other):
        """
        Implement the `>=` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form >= other._tuple_form
        else:
            return NotImplemented

    def toterm(self):
        """
        Convert this operator to ParticleTerm instance.

        Returns
        -------
        term : ParticleTerm
            The term corresponding to this operator.
        """

        spin = self.spin
        orbit = self.orbit
        site = self.coordinate
        c = AoC(CREATION, site=site, spin=spin, orbit=orbit)
        a = AoC(ANNIHILATION, site=site, spin=spin, orbit=orbit)
        return ParticleTerm((c, a), coeff=1.0)

    def __mul__(self, other):
        """
        Implement the binary arithmetic operation: `*`.

        `self` is the left operand and `other` is the right operand;
        Return an instance of ParticleTerm.
        """

        return self.toterm() * other

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`.

        `self` is the right operand and `other` is the left operand;
        Return an instance of ParticleTerm.
        """

        return other * self.toterm()

    def dagger(self):
        """
        Return the Hermitian conjugate of this operator.
        """

        return self

    def derive(self, *, site=None, spin=None, orbit=None):
        """
        Derive a new instance from `self` and the given parameters.

        This method creates a new instance with the same attribute as `self`
        except for these given to this method.
        All the parameters should be specified as keyword arguments.

        Returns
        -------
        res : A new instance of NumberOperator.
        """

        if site is None:
            site = self.coordinate
        if spin is None:
            spin = self.spin
        if orbit is None:
            orbit = self.orbit
        return self.__class__(site=site, spin=spin, orbit=orbit)

    def matrix_repr(
            self, state_indices_table, bases, *, to_csr=True
    ):
        """
        Return the matrix representation of this operator in the Hilbert space.

        Parameters
        ----------
        state_indices_table : IndexTable
            A table that associate instances of StateID with integer indices.
        bases : 1D np.ndarray
            The bases of the Hilbert space.
            The data-type of the array's elements is np.uint64.
        to_csr : bool, optional, keyword-only
            Whether to construct a csr_matrix as the result.
            Default: True.

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of the operator in the Hilbert space.
            If `to_csr` is set to True, the result is a csr_matrix;
            If set to False, the result is a tuple: (entries, (rows, cols)),
            where `entries` is the non-zero matrix elements, `rows` and
            `cols` are the row and column indices of the none-zero elements.
        """

        index = state_indices_table(self._state)
        term = [(index, CREATION), (index, ANNIHILATION)]
        return matrix_function(term, bases, to_csr=to_csr, special_tag="number")


class SwapFermionError(Exception):
    """
    Raised when swap creation and annihilation operators defined on the same
    single-particle state.
    """

    def __init__(self, aoc0, aoc1):
        self.aoc0 = aoc0
        self.aoc1 = aoc1

    def __str__(self):
        return "\n".join(
            [
                "Swapping the following two operators would generate extra "
                "identity operator which can not be processed properly:",
                "    {0!r}".format(self.aoc0),
                "    {0!r}".format(self.aoc1),
            ]
        )


class ParticleTerm:
    """
    A unified description of any term composed of creation and/or
    annihilation operators.

    Attributes
    ----------
    coeff : float, int or complex
        The coefficient of this term.
    components : tuple
        The component creation and/or annihilation operators of this term.
    classification : {"general", "hopping", "number" or "Coulomb"}
        The classification of the term.

    Examples
    --------
    >>> from HamiltonianPy.quantumoperator import AoC, ParticleTerm
    >>> c = AoC(otype=1, site=[0, 0], spin=0)
    >>> a = AoC(otype=0, site=(0.3, 0.75), spin=1)
    >>> term = ParticleTerm((c, a), coeff=1.2)
    >>> print(term)
    The coefficient of this term: 1.2
    The component operators:
        AoC(otype=CREATION, site=(0, 0), spin=0, orbit=0)
        AoC(otype=ANNIHILATION, site=(0.3, 0.75), spin=1, orbit=0)
    >>> print(2.0 * term)
    The coefficient of this term: 2.4
    The component operators:
        AoC(otype=CREATION, site=(0, 0), spin=0, orbit=0)
        AoC(otype=ANNIHILATION, site=(0.3, 0.75), spin=1, orbit=0)
    """

    def __init__(self, aocs, coeff=1.0, *, classification="general"):
        """
        Customize the newly created instance.

        Parameters
        ----------
        aocs : tuple or list
            A collection of creation and/or annihilation operators that
            composing this term.
        coeff : float, int or complex, optional
            The coefficient of this term.
            Default: 1.0.
        classification : str, optional, keyword-only
            A tag that identify the classification of the instance.
            Supported values: "general", "hopping", "number" and "Coulomb".
            If you are not sure about this parameter, just use the default
            value.
            "general" means that the instance is just a term composed of
            creation and/or annihilation operators;
            "hopping" means that the instance is a hopping term:
            '$c_i^{\\dagger} c_j$' and `i != j`(Note: the `i == j` case does
            not belong to this category);
            "number" means that the instance is a particle-number(chemical
            potential) term: 'c_i^{\\dagger} c_i';
            "Coulomb" means that the instance is a Coulomb interaction term:
            'n_i n_j'.
            The "hopping", "number" and "Coulomb" categories can also be
            classified as "general".
            Currently, this class does not check whether the given `aocs` is
            compatible with the `classification` parameter. The user is
            responsible for the compatibility of these two parameters. If
            these two parameters are incompatible, the corresponding
            instance would behave incorrectly, so use this parameter with
            caution.
        """

        assert isinstance(coeff, NUMERIC_TYPES_GENERAL), "Invalid coefficient"
        assert classification in ("general", "hopping", "number", "Coulomb")

        self._aocs = tuple(aocs)
        self._coeff = coeff
        self._classification = classification

    @property
    def coeff(self):
        """
        The coefficient of this term.
        """

        return self._coeff

    @coeff.setter
    def coeff(self, coeff):
        assert isinstance(coeff, NUMERIC_TYPES_GENERAL), "Invalid coefficient"
        self._coeff = coeff

    @property
    def components(self):
        """
        The component creation and/or annihilation operators of this term.
        """

        return self._aocs

    @property
    def classification(self):
        """
        The `classification` attribute.
        """

        return self._classification

    def __str__(self):
        """
        Return a string that describes the content of this instance.
        """

        return "\n".join(
            [
                "The coefficient of this term: {0}".format(self._coeff),
                "The component operators:",
                *["    {0}".format(aoc) for aoc in self._aocs],
            ]
        )

    def tolatex(self, indices_table=None, **kwargs):
        """
        Return the LaTex form of this term.

        Parameters
        ----------
        indices_table : IndexTable or None, optional
            A table that associate instances of SiteID with integer indices.
            The `indices_table` is passed to the `tolatex` method of
            `StateID` as the `site_index` argument.
            If not given or None, the `site` is show as it is.
            Default : None.
        kwargs :
            All other keyword arguments are passed to the `tolatex` method of
            `StateID`.
            See also: `StateID.tolatex`.

        Returns
        -------
        latex : str
            The LaTex form of this term.
        """

        latex_aocs = [
            aoc.tolatex(site_index=indices_table, **kwargs).replace("$", "")
            for aoc in self._aocs
        ]
        return "".join(["$", str(self._coeff), *latex_aocs, "$"])

    def show(self, indices_table=None, **kwargs):
        """
        Show the term in handwriting form.

        Parameters
        ----------
        indices_table : IndexTable or None, optional
            A table that associate instances of SiteID with integer indices.
            The `indices_table` is passed to the `tolatex` method of
            `StateID` as the `site_index` argument.
            If not given or None, the `site` is show as it is.
            Default : None.
        kwargs :
            All other keyword arguments are passed to the `tolatex` method of
            `StateID`.
            See also: `StateID.tolatex`.
        """

        fig, ax = plt.subplots()
        ax.text(
            0.5, 0.5, self.tolatex(indices_table, **kwargs),
            fontsize="xx-large", ha="center",
            va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        plt.show()

    def __mul__(self, other):
        """
        Implement the binary arithmetic operation: `*`.

        `self` is the left operand and `other` is the right operand;
        Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            aocs = self._aocs + other._aocs
            coeff = self._coeff * other._coeff
        elif isinstance(other, AoC):
            aocs = self._aocs + (other, )
            coeff = self._coeff
        elif isinstance(other, NUMERIC_TYPES_GENERAL):
            aocs = self._aocs
            coeff = self._coeff * other
        else:
            return NotImplemented

        return self.__class__(aocs=aocs, coeff=coeff)

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`.

        `self` is the right operand and `other` is the left operand;
        This method return a new instance of this class.
        """

        if isinstance(other, AoC):
            aocs = (other, ) + self._aocs
            coeff = self._coeff
        elif isinstance(other, NUMERIC_TYPES_GENERAL):
            aocs = self._aocs
            coeff = other * self._coeff
        else:
            return NotImplemented

        return self.__class__(aocs=aocs, coeff=coeff)

    @staticmethod
    def normalize(aocs):
        """
        Reordering the given `aocs` into norm form.

        For a composite operator consisting of creation and/or annihilation
        operators, the norm form means that all the creation operators appear
        to the left of all the annihilation operators. Also, the creation and
        annihilation operators are sorted in ascending and descending order
        respectively according to the single-particle state associated with
        the operator.

        See the document of `__lt__` method of AoC for the comparison logic.

        Parameters
        ----------
        aocs : list or tuple
            A collection of creation and/or annihilation operators.

        Returns
        -------
        aocs : list
            The norm form of the operator.
        swap_count : int
            The number of swap to obtain the normal form.

        Raises
        ------
        SwapFermionError :
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
                        raise SwapFermionError(aoc0, aoc1)
        return aocs, swap_count

    def dagger(self):
        """
        Return the Hermitian conjugate of this term.
        """

        aocs = [aoc.dagger() for aoc in self._aocs[::-1]]
        return self.__class__(aocs=aocs, coeff=self._coeff.conjugate())

    def matrix_repr(
            self, state_indices_table, right_bases, *,
            left_bases=None, coeff=None, to_csr=True
    ):
        """
        Return the matrix representation of this term.

        Parameters
        ----------
        state_indices_table : IndexTable
            A table that associate instances of StateID with integer indices.
        right_bases : 1D np.ndarray
            The bases of the Hilbert space before the operation.
            The data-type of the array's elements is np.uint64.
        left_bases : 1D np.ndarray, optional, keyword-only
            The bases of the Hilbert space after the operation.
            If given, the data-type of the array's elements is np.uint64.
            If not given or None, left_bases is the same as right_bases.
            Default: None.
        coeff : int, float or complex, optional, keyword-only
            A new coefficient for this term.
            If not given or None, use the original coefficient.
            Default: None.
        to_csr : bool, optional, keyword-only
            Whether to construct a csr_matrix as the result.
            Default: True.

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of the operator in the Hilbert space.
            If `to_csr` is set to True, the result is a csr_matrix;
            If set to False, the result is a tuple: (entries, (rows, cols)),
            where `entries` is the non-zero matrix elements, `rows` and
            `cols` are the row and column indices of the none-zero elements.
        """

        if coeff is not None:
            self.coeff = coeff

        term = [
            (aoc.getStateIndex(state_indices_table), aoc.otype)
            for aoc in self._aocs
        ]
        return matrix_function(
            term, right_bases,
            left_bases=left_bases, coeff=self._coeff, to_csr=to_csr
        )
