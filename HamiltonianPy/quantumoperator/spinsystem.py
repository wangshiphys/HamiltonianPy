"""
This module provides classes that describe quantum spin operators as well as
spin interactions.
"""


__all__ = [
    "SpinOperator",
    "SpinInteraction",
]


from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, identity, kron

from HamiltonianPy.quantumoperator.constant import ANNIHILATION, CREATION, \
    NUMERIC_TYPES_GENERAL, SPIN_MATRICES, SPIN_OTYPES, SPIN_DOWN, SPIN_UP
from HamiltonianPy.quantumoperator.particlesystem import AoC, ParticleTerm
from HamiltonianPy.quantumoperator.quantumstate import SiteID


class SpinOperator:
    """
    A unified description of quantum spin operator.

    Attributes
    ----------
    otype : str
        The type of this spin operator.
        Supported value: "x" | "y" | "z" | "p" | "m".
    site_id : SiteID
        The ID of the lattice site on which the spin operator is defined.
    coordinate : tuple
        The coordinates of the lattice site in tuple form.
    site : 1D np.ndarray
        The coordinates of the lattice site in np.ndarray form.

    Examples
    --------
    >>> from HamiltonianPy.quantumoperator import SpinOperator
    >>> SX = SpinOperator("x", site=[0, 0])
    >>> SY = SpinOperator("y", site=[1, 1])
    >>> SX
    SpinOperator(otype="x", site=(0, 0))
    >>> SY.matrix()
    array([[ 0.+0.j , -0.-0.5j],
           [ 0.+0.5j,  0.+0.j ]])
    >>> SY < SX
    False
    >>> SX.dagger() is SX
    True
    >>> print(2 * SX * SY)
    The coefficient of this term: 2
    The component operators:
        SpinOperator(otype="x", site=(0, 0))
        SpinOperator(otype="y", site=(1, 1))
    """

    def __init__(self, otype, site):
        """
        Customize the newly created instance.

        Parameters
        ----------
        otype : {"x", "y", "z", "p" or "m"}
            The type of this spin operator.
        site : list, tuple or 1D np.ndarray
            The coordinates of the lattice site on which the spin operator is
            defined. The `site` parameter should be 1D array with length 1,
            2 or 3.
        """

        assert otype in SPIN_OTYPES, "Invalid operator type"

        site_id = SiteID(site=site)
        self._otype = otype
        self._site_id = site_id
        # The tuple form of this instance
        # It is a tuple: (otype, site) and site itself is a tuple with length
        # 1, 2 or 3.
        self._tuple_form = (otype, site_id._tuple_form)

    @property
    def otype(self):
        """
        The `otype` attribute.
        """

        return self._otype

    @property
    def site_id(self):
        """
        The `site_id` attribute.
        """

        return self._site_id

    @property
    def coordinate(self):
        """
        The `coordinate` attribute.
        """

        return self._site_id.coordinate

    @property
    def site(self):
        """
        The `site` attribute.
        """

        return self._site_id.site

    def getIndex(self, indices_table):
        """
        Return the index of this operator.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instances of SpinOperator with integer
            indices.

        Returns
        -------
        index : int
            The index of this instance in the given table.

        See also
        --------
        getSiteIndex
        """

        return indices_table(self)

    def getSiteIndex(self, indices_table):
        """
        Return the index of the lattice site on which this operator is defined.

        Notes:
            This method is different from the `getIndex` method.
            This method return the index of the site on which this operator
            is defined and the `getIndex` method return the index of the
            operator itself.

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instances of SiteID with integer indices.

        Returns
        -------
        index : int
            The index of the `site_id` attribute of this instance.
        """

        return indices_table(self._site_id)

    def __repr__(self):
        """
        Official string representation of the instance.
        """

        info = 'SpinOperator(otype="{0}", site={1!r})'
        return info.format(self._otype, self.coordinate)

    __str__ = __repr__

    def tolatex(self, **kwargs):
        """
        Return the LaTex form of this instance.

        Parameters
        ----------
        kwargs :
            All keyword arguments are passed to the `tolatex` method of the
            `site_id` attribute.
            See also: `SiteID.tolatex`.

        Returns
        -------
        latex : str
            The LaTex form of this instance.
        """

        subscript = self._site_id.tolatex(**kwargs)
        return r"$S_{{{0}}}^{{{1}}}$".format(subscript, self._otype)

    def show(self, **kwargs):
        """
        Show the instance in handwriting form.

        Parameters
        ----------
        kwargs :
            All keyword arguments are passed to the `tolatex` method of the
            `site_id` attribute.
            See also: `SiteID.tolatex`.
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

    def __eq__(self, other):
        """
        Implement the `==` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form == other._tuple_form
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

    def __le__(self, other):
        """
        Implement the `<=` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form <= other._tuple_form
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

    def __ge__(self, other):
        """
        Implement the `>=` operator between self and other.
        """

        if isinstance(other, self.__class__):
            return self._tuple_form >= other._tuple_form
        else:
            return NotImplemented

    def __mul__(self, other):
        """
        Implement the binary arithmetic operation: `*`.

        `self` is the left operand and `other` is the right operand;
        Return an instance of SpinInteraction/
        """

        if isinstance(other, self.__class__):
            return SpinInteraction((self, other), coeff=1.0)
        elif isinstance(other, NUMERIC_TYPES_GENERAL):
            return SpinInteraction((self,), coeff=other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`.

        `self` parameter is the right operand and `other` is the left operand;
        Return an instance of SpinInteraction.
        """

        if isinstance(other, NUMERIC_TYPES_GENERAL):
            return SpinInteraction((self,), coeff=other)
        else:
            return NotImplemented

    def matrix(self):
        """
        Return the matrix representation of the spin operator.

        The matrix representation is calculated in the single spin Hilbert
        space, i.e. 2 dimension.

        See also
        --------
        matrix_function
        matrix_repr
        """

        return np.array(SPIN_MATRICES[self._otype], copy=True)

    def dagger(self):
        """
        Return the Hermitian conjugate of this operator.
        """

        if self._otype == "p":
            operator = self.derive(otype="m")
        elif self._otype == "m":
            operator = self.derive(otype="p")
        else:
            operator = self
        return operator

    def conjugate_of(self, other):
        """
        Return whether `self` is Hermitian conjugate of `other`.
        """

        if isinstance(other, self.__class__):
            return self.dagger() == other
        else:
            raise TypeError(
                "The `other` parameter is not instance of this class!"
            )

    def same_site(self, other):
        """
        Return whether `self` and `other` is defined on the same lattice site.
        """

        if isinstance(other, self.__class__):
            return self._site_id == other._site_id
        else:
            raise TypeError(
                "The `other` parameter is not instance of this class!"
            )

    def derive(self, *, otype=None, site=None):
        """
        Derive a new instance from `self` and the given parameters.

        This method creates a new instance with the same attribute as `self`
        except for these given to this method.
        All the parameters should be specified as keyword arguments.

        Returns
        -------
        res : A new instance of SpinOperator.
        """

        if otype is None:
            otype = self.otype
        if site is None:
            site = self.coordinate
        return SpinOperator(otype=otype, site=site)

    def Schwinger(self):
        """
        Return the Schwinger Fermion representation of this spin operator.
        """

        coordinate = self.coordinate
        C_UP = AoC(otype=CREATION, site=coordinate, spin=SPIN_UP)
        C_DOWN = AoC(otype=CREATION, site=coordinate, spin=SPIN_DOWN)
        A_UP = AoC(otype=ANNIHILATION, site=coordinate, spin=SPIN_UP)
        A_DOWN = AoC(otype=ANNIHILATION, site=coordinate, spin=SPIN_DOWN)

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
        Calculate the matrix representation of the spin operator.

        For a specific spin operator, its' matrix representation in the
        Hilbert space is defined as follow:
            I_{n-1} * ... * I_{i+1} * S_i * I_{i-1} * ... * I_0
        where I is (2, 2) identity matrix, `*` represents tensor product,
        `n` is the total number of spins and `i` is the index of the lattice
        site.

        Parameters
        ----------
        operator : tuple or list
            Length 2 tuple or list: (index, otype) or [index, otype].
            `index` is the index of the lattice site on which the spin
            operator is defined;
            `otype` is the type of the spin operator which should be only one
            of "x" | "y" | "z" | "p" | "m".
        total_spin : int
            The total number of spins.

        Returns
        -------
        res : csr_matrix
            The matrix representation of this spin operator.
        """

        index, otype = operator
        I = identity(1 << index, dtype=np.float64, format="csr")
        res = kron(SPIN_MATRICES[otype], I, format="csr")
        I = identity(1 << (total_spin-index-1), dtype=np.float64, format="csr")
        return kron(I, res, format="csr")

    def matrix_repr(self, site_indices_table):
        """
        Return the matrix representation of this spin operator.

        For a specific spin operator, its matrix representation in the
        Hilbert space is defined as follow:
            I_{n-1} * ... * I_{i+1} * S_i * I_{i-1} * ... * I_0
        where I is (2, 2) identity matrix, `*` represents tensor product,
        `n` is the total number of spins and `i` is the index of the lattice
        site.

        Parameters
        ----------
        site_indices_table : IndexTable
            A table that associate instances of SiteID with integer indices.

        Returns
        -------
        res : csr_matrix
            The matrix representation of this spin operator.
        """

        total_spin = len(site_indices_table)
        operator = (site_indices_table(self._site_id), self._otype)
        return self.matrix_function(operator, total_spin)


class SpinInteraction:
    """
    A unified description of spin interaction term.

    Attributes
    ----------
    coeff : float, int or complex
        The coefficient of this term.
    components : tuple
        The component spin operators of this term.

    Examples
    --------
    >>> from HamiltonianPy.quantumoperator import SpinOperator, SpinInteraction
    >>> S0X = SpinOperator("x", site=[0, 0])
    >>> S1X = SpinOperator("x", site=[0, 1])
    >>> term = SpinInteraction((S0X, S1X), coeff=1.5)
    >>> print(term)
    The coefficient of this term: 1.5
    The component operators:
        SpinOperator(otype="x", site=(0, 0))
        SpinOperator(otype="x", site=(0, 1))
    >>> print(2 * term)
    The coefficient of this term: 3.0
    The component operators:
        SpinOperator(otype="x", site=(0, 0))
        SpinOperator(otype="x", site=(0, 1))
    """

    def __init__(self, operators, coeff=1.0):
        """
        Customize the newly created instance.

        Parameters
        ----------
        operators : tuple or list
            A collection of `SpinOperator` objects that composing this term.
        coeff : int, float, complex, optional
            The coefficient of this term.
            Default: 1.0.
        """

        assert isinstance(coeff, NUMERIC_TYPES_GENERAL), "Invalid coefficient"

        # Sorting the spin operators in ascending order according to their
        # SiteID. The relative position of two operators with the same SiteID
        # will not change and the exchange of two spin operators on different
        # lattice site never change the interaction term.
        self._operators = tuple(
            sorted(operators, key=lambda item: item.site_id)
        )
        self._coeff = coeff

    @property
    def coeff(self):
        """
        The coefficient of this term.
        """

        return self._coeff

    @coeff.setter
    def coeff(self, value):
        assert isinstance(value, NUMERIC_TYPES_GENERAL), "Invalid coefficient"
        self._coeff = value

    @property
    def components(self):
        """
        The component spin operators of this term.
        """

        return self._operators

    def __str__(self):
        """
        Return a string that describes the content of the instance.
        """

        return "\n".join(
            [
                "The coefficient of this term: {0}".format(self._coeff),
                "The component operators:",
            ] + ["    {0}".format(operator) for operator in self._operators]
        )

    def tolatex(self, indices_table=None, **kwargs):
        """
        Return the LaTex form of this instance.

        Parameters
        ----------
        indices_table : IndexTable or None, optional
            A table that associate instances of SiteID with integer indices.
            The `indices_table` is passed to the `tolatex` method of
            `SiteID` as the `site_index` argument.
            If not given or None, the `site` is show as it is.
            Default: None.
        kwargs :
            All other keyword arguments are passed to the `tolatex` method of
            `SiteID`.
            See also: `SiteID.tolatex`.

        Returns
        -------
        latex : str
            The LaTex form of this instance.
        """

        latex_operators = [
            operator.tolatex(
                site_index=indices_table, **kwargs
            ).replace("$", "") for operator in self._operators
        ]
        return "".join(["$", str(self._coeff), *latex_operators, "$"])

    def show(self, indices_table=None, **kwargs):
        """
        Show the instance in handwriting form.

        Parameters
        ----------
        indices_table : IndexTable or None, optional
            A table that associate instances of SiteID with integer indices.
            The `indices_table` is passed to the `tolatex` method of
            `SiteID` as the `site_index` argument.
            If not given or None, the `site` is show as it is.
            Default: None.
        kwargs :
            All other keyword arguments are passed to the `tolatex` method of
            `SiteID`.
            See also: `SiteID.tolatex`.
        """

        fig, ax = plt.subplots()
        ax.text(
            0.5, 0.5, self.tolatex(indices_table, **kwargs),
            fontsize="xx-large", ha="center", va="center",
            transform=ax.transAxes
        )
        ax.set_axis_off()
        plt.show()

    def __mul__(self, other):
        """
        Implement the binary arithmetic operation: `*`.

        `self` is the left operand and `other'` is the right operand;
        Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            operators = self._operators + other._operators
            coeff = self._coeff * other._coeff
        elif isinstance(other, SpinOperator):
            operators = self._operators + (other, )
            coeff = self._coeff
        elif isinstance(other, NUMERIC_TYPES_GENERAL):
            operators = self._operators
            coeff = self._coeff * other
        else:
            return NotImplemented

        return SpinInteraction(operators, coeff=coeff)

    def __rmul__(self, other):
        """
        Implement the binary arithmetic operation: `*`.

        `self` is the right operand and `other` is the left operand;
        This method return a new instance of this class.
        """

        if isinstance(other, SpinOperator):
            operators = (other, ) + self._operators
            coeff = self._coeff
        elif isinstance(other, NUMERIC_TYPES_GENERAL):
            operators = self._operators
            coeff = other * self._coeff
        else:
            return NotImplemented

        return SpinInteraction(operators, coeff=coeff)

    def dagger(self):
        """
        Return the Hermitian conjugate of this term.
        """

        operators = [operator.dagger() for operator in self._operators[::-1]]
        return SpinInteraction(operators, coeff=self._coeff.conjugate())

    def Schwinger(self):
        """
        Return the Schwinger Fermion representation of this term.
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
        Return the matrix representation of the spin interaction term.

        Parameters
        ----------
        operators : sequence
            A sequence of 2-tuple: [(index_0, otype_0), ..., (index_n, otype_n)]
            `index_i` is the index of the lattice site on which the spin
            operator is defined;
            `otype_i` is the type of the spin operator which should be only
            one of "x" | "y" | "z" | "p" | "m".
        total_spin: int
            The total number of spins.
        coeff : int, float or complex, optional
            The coefficient of the term.
            Default: 1.0.

        Returns
        -------
        res : csr_matrix
            The matrix representation of this term.
        """

        assert isinstance(total_spin, int) and total_spin > 0
        assert isinstance(coeff, NUMERIC_TYPES_GENERAL), "Invalid coefficient"

        operators = sorted(operators, key=lambda item: item[0], reverse=True)
        if len(operators) == 2 and operators[0][0] > operators[1][0]:
            (i, alpha), (j, beta) = operators
            Si = coeff * SPIN_MATRICES[alpha]
            Sj = SPIN_MATRICES[beta]
            dim0 = 1 << j
            dim1 = 1 << (i - j - 1)
            dim2 = 1 << (total_spin - i - 1)

            if dim1 == 1:
                res = kron(Si, Sj, format="csr")
            else:
                I = identity(dim1, dtype=np.float64, format="csr")
                res = kron(Si, kron(I, Sj, format="csr"), format="csr")
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

    def matrix_repr(self, site_indices_table, coeff=None):
        """
        Return the matrix representation of this spin interaction term.

        Parameters
        ----------
        site_indices_table : IndexTable
            A table that associate instances of SiteID with integer indices.
        coeff : int, float or complex, optional
            A new coefficient for this spin interaction term.
            If not given or None, use the original coefficient.
            Default: None.

        Returns
        -------
        res : csr_matrix
            The matrix representation of this spin interaction term.
        """

        if coeff is not None:
            self.coeff = coeff

        total_spin = len(site_indices_table)
        operators = [
            (operator.getSiteIndex(site_indices_table), operator.otype)
            for operator in self._operators
        ]
        return self.matrix_function(operators, total_spin, self.coeff)
