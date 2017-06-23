from itertools import product
from scipy.sparse import csr_matrix, identity, kron

import numpy as np

from HamiltonianPy.constant import CREATION, ANNIHILATION, SWAP_FACTOR_F
from HamiltonianPy.constant import SPIN_MATRIX, SPIN_OTYPE, SPIN_DOWN, SPIN_UP
from HamiltonianPy.constant import NDIGITS, FLOAT_TYPE
from HamiltonianPy.exception import SwapFermionError
from HamiltonianPy.indexmap import IndexMap

import HamiltonianPy.extpkg.matrixrepr as cextmr

__all__ = ["SiteID", "StateID", "AoC", "SpinOperator", 
           "SpinInteraction", "ParticleTerm"]


class SiteID:# {{{
    """
    A wrapper of 1D np.ndarray which represents the coordinates of a point.

    The reason to define this wrapper is to make the coordinates hashable as 
    well as comparable as a whole.

    Attribute:
    ----------
    site: np.ndarray
        The coordinates of the point. The shape of this array should be only 
        one of (1, ), (2, ) or (3, ).

    Method:
    -------
    Special method:
        __init__(site)
        __str__()
        __hash__()
        __lt__(other)
        __eq__(other)
        __gt__(other)
        __le__(other)
        __ne__(other)
        __ge__(other)

    General method:
        tupleform()
        getSite()
        getIndex(objmap)
    """

    def __init__(self, site):# {{{
        """
        Initialize instance of this class.
        """

        if isinstance(site, np.ndarray) and site.shape in [(1,), (2,), (3,)]:
            self.site = np.array(site[:])
        else:
            raise TypeError("The invalid site parameter.")
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        return "site: " + str(self.site)
    # }}}

    def __hash__(self):# {{{
        """
        Return the hash value of instance of this class.
        """

        return hash(self.tupleform())
    # }}}

    def __lt__(self, other):# {{{
        """
        Define the < operator between instance of this class.
        """

        if isinstance(other, self.__class__):
            return self.tupleform() < other.tupleform()
        else:
            raise TypeError("The rigth operand is not instance of this class.")
    # }}}

    def __eq__(self, other):# {{{
        """
        Define the == operator between instance of this class.
        """

        if isinstance(other, self.__class__):
            return self.tupleform() == other.tupleform()
        else:
            raise TypeError("The rigth operand is not instance of this class.")
    # }}}

    def __gt__(self, other):# {{{
        """
        Define the > operator between instance of this class.
        """

        if isinstance(other, self.__class__):
            return self.tupleform() > other.tupleform()
        else:
            raise TypeError("The rigth operand is not instance of this class.")
    # }}}

    def __le__(self, other):# {{{
        """
        Define the <= operator between instance of this class.
        """

        return self.__lt__(other) or self.__eq__(other)
    # }}}

    def __ne__(self, other):# {{{
        """
        Define the != operator between instance of this class.
        """

        return not self.__eq__(other)
    # }}}

    def __ge__(self, other):# {{{
        """
        Define the >= operator between instance of this class.
        """

        return self.__gt__(other) or self.__eq__(other)
    # }}}

    def tupleform(self):# {{{
        """
        The tuple form of the coordinates.

        This method is usful in calculating the hash value of instance of this
        class as well as defining compare logic between instances of this class.
        """

        if self.site.dtype in FLOAT_TYPE:
            factor = 10 ** NDIGITS
            res = tuple([int(factor * i) for i in self.site])
        else:
            res = tuple(self.site)
        return res
    # }}}

    def getSite(self):# {{{
        """
        Access the site attribute of instance of this class.
        """

        return np.array(self.site[:])
    # }}}

    def getIndex(self, objmap):# {{{
        """
        Return the index associated with this SiteID.

        Parameter:
        ----------
        objmap: IndexMap
            A map system that associate instance of SiteID with an integer
            index.

        Return:
        -------
        res: int
            The index of self in the map system.
        """
        
        if not isinstance(objmap, IndexMap):
            raise TypeError("The input objmap is not instance of IndexMap!")

        return objmap(self)
    # }}}
# }}}


class StateID(SiteID):# {{{
    """
    This class provide a unified description of a single particle state.

    Attribute:
    ----------
    site: ndarray
        The coordinate of the localized state. The site attribute should be 
        a 1D array, usually it has length 1, 2 or 3 cooresponding to 1, 2 or 
        3 space dimension.
    spin: int, optional
        The spin index of the state.
        default: 0
    orbit: int, optional
        The orbit index of the state.
        default: 0

    Method:
    -------
    Special methods:
        __init__(site, spin=0, orbit=0)
        __str__()
    General methods:
        getSpin()
        getOrbit()
        tupleform()

    Methods inherited from SiteID:
    __hash__()
    __eq__(other)
    __lt__(other)
    __gt__(other)
    __ne__(other)
    __le__(other)
    __ge__(other)
    getSite()
    getIndex(objmap)
    """

    def __init__(self, site, spin=0, orbit=0):# {{{
        """
        Initialize instance of this class!

        Parameter:
        ----------
        See the documentation of this class.
        """

        SiteID.__init__(self, site=site)

        if isinstance(spin, int) and spin >= 0:
            self.spin = spin
        else:
            raise ValueError("The invalid spin parameter!")

        if isinstance(orbit, int) and orbit >= 0:
            self.orbit = orbit
        else:
            raise ValueError("The invalid orbit parameter!")
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "orbit: {0}\nspin: {1}\n".format(self.orbit, self.spin)
        info += SiteID.__str__(self)
        return info
    # }}}

    def tupleform(self):# {{{
        """
        Combine the above three attributes to a tuple, (orbit, spin, site)
        
        This method is usful in calculating the hash value of instance of this
        class as well as defining compare logic between instances of this class.
        """

        return (self.orbit, self.spin) + SiteID.tupleform(self)
    # }}}

    def getSpin(self):# {{{
        """
        Access the spin attribute of instance of this class.
        """

        return self.spin
    # }}}

    def getOrbit(self):# {{{
        """
        Access the Orbit attribute of instance of this class.
        """

        return self.orbit
    # }}}
# }}}


class AoC(StateID):# {{{
    """
    This class provide a unified description of the creation and annihilation 
    operator.

    Attribute:
    ----------
    otype: int
        The type of this operator. It can be either 0 or 1, wich represents
        annihilation or creation respectively.
    site: ndarray
        The coordinate of the localized state. The site attribute should be
        a 1D array, usually it has length 1, 2 or 3 cooresponding to 1, 2 or 
        space dimension.
    spin: int, optional
        The spin index of the state.
        default: 0
    orbit: int, optional
        The orbit index of the state.
        default: 0

    Method:
    -------
    Special methods:
        __init__(otype, site, spin=0, orbit=0)
        __str__()
        __gt__(other)
        __lt__(other)
    General methods:
        getOtype()
        getStateID()
        tupleform()
        dagger()
        sameState(other)
        conjugateOf(other)
        update(*, otype=None, site=None, spin=None, orbit=None)
        matrixRepr(statemap, rbase, lbase=None, *, to_csr=True)
    Static methods:
        matrixFunc(operator, rbase, lbase=None, *, to_csr=True)
    Methods Inherit from SiteID:
        __hash__()
        __eq__(other)
        __ne__(other)
        __ge__(other)
        __le__(other)
        getSite()
        getIndex(objmap)
    Methods Inherit from StateID:
        getSpin()
        getOrbit()
    """

    def __init__(self, otype, site, spin=0, orbit=0):# {{{
        """
        Initilize instance of this class.

        Paramter:
        ---------
        See the documentation of this class.
        """

        StateID.__init__(self, site=site, spin=spin, orbit=orbit)

        if otype in (ANNIHILATION, CREATION):
            self.otype = otype
        else:
            raise ValueError("The invalid otype!")
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describles the content of the instance.
        """

        info = 'otype: {0}\n'.format(self.otype)
        info += StateID.__str__(self)
        return info
    # }}}

    def __lt__(self, other):# {{{
        """
        Define the < operator between instance of this class.

        The comparsion logic is as follow:
        Creation operator is always less than annihilation operator;
        The smaller the stateid, the samller the creation operator;
        The larger the stateid, the smaller the annihilation operator.
        """

        if isinstance(other, self.__class__):
            otype0 = self.otype
            otype1 = other.otype
            id0 = self.getStateID()
            id1 = other.getStateID()
            if otype0 == CREATION and otype1 == CREATION:
                return id0 < id1
            elif otype0 == CREATION and otype1 == ANNIHILATION:
                return True
            elif otype0 == ANNIHILATION and otype1 == CREATION:
                return False
            else:
                return id0 > id1
        else:
            raise TypeError("The right operand is not instance of this class.")
    # }}}

    def __gt__(self, other):# {{{
        """
        Define the > operator between instance of this class.

        See document of __lt__ special method for the comparsion logic.
        """

        if isinstance(other, self.__class__):
            otype0 = self.otype
            otype1 = other.otype
            id0 = self.getStateID()
            id1 = other.getStateID()
            if otype0 == CREATION and otype1 == CREATION:
                return id0 > id1
            elif otype0 == CREATION and otype1 == ANNIHILATION:
                return False
            elif otype0 == ANNIHILATION and otype1 == CREATION:
                return True
            else:
                return id0 < id1
        else:
            raise TypeError("The right operand is not instance of this class.")
    # }}}

    def tupleform(self):# {{{
        """
        Combine the above four attributes to a tuple, (otype, orbit, spin, site)

        This method is usful in calculating the hash value of instance of this
        class as well as defining compare logic between instances of this class.
        """

        return (self.otype, ) + StateID.tupleform(self)
    # }}}

    def getOtype(self):# {{{
        """
        Access the otype attribute of instance of this class.
        """

        return self.otype
    # }}}

    def getStateID(self):# {{{
        """
        Extract the state information of this creation or annihilation operator.

        Return:
        -------
        res: A new instance of StateID class.
        """

        return StateID(site=self.site, spin=self.spin, orbit=self.orbit)
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermitian conjugate of self.

        Return:
        -------
        res: A new instance of this class.
        """

        if self.otype == CREATION:
            otype = ANNIHILATION
        else:
            otype = CREATION
        res = AoC(otype=otype, site=self.site, spin=self.spin, orbit=self.orbit)
        return res
    # }}}

    def conjugateOf(self, other):# {{{
        """
        Determine whether the instance is conjugate of the other instance.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")
        
        return self.dagger() == other
    # }}}

    def sameState(self, other):# {{{
        """
        Determine whether the self AoC and other AoC is of the same state.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.getStateID() == other.getStateID()
    # }}}

    def update(self, *, otype=None, site=None, spin=None, orbit=None):# {{{
        """
        Create a new aoc with the same parameter as self except for those 
        given to update method.
        
        All the parameters should be specified as keyword argument.

        Return:
        -------
        res: A new instance of AoC.
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
    def matrixFunc(operator, rbase, lbase=None, *, to_csr=True):# {{{
        """
        Return the matrix representation of the operator in the Hilbert space
        specified by the rbase and optional lbase.

        Parameter:
        ----------
        operator: tuple or list
            The parameter should be a tuple or list with two entries. The first
            entries is the index of the single-particle state and the second is
            the "CREATION" or "ANNIHILATION" constant.
        rbase: tuple or list
            The bases of the Hilbert space before the operation.
        lbase: tuple or list, optional
            The bases of the Hilbert space after the operation.
            If not given or None, lbase is the same as rbase.
            default: None
        to_csr: boolean, optional, only keyword argument
            Whether to construct a csr_matrix as the result.
            default: True

        Return:
        -------
        res: csr_matrix or tuple
            The matrix representation of the creation or annihilation operator.
            If to_csr is True, the result is a csr_matrix, if False, the result
            is a tuple. The first is the non-zero matrix entries, the second and
            third are the row and col indices.
        """

        rdim = len(rbase)
        if lbase is None:
            shape = (rdim, rdim)
            data = cextmr.matrixRepr([operator], rbase)
        else:
            shape = (len(lbase), rdim)
            data = cextmr.matrixRepr([operator], rbase, lbase)

        if to_csr:
            res = csr_matrix(data, shape=shape)
        else:
            res = data
        return res
    # }}}

    def matrixRepr(self, statemap, rbase, lbase=None, *, to_csr=True):# {{{
        """
        Return the matrix representation of the operator specified by self in 
        the Hilbert space of the manybody system.
        
        Parameter:
        ----------
        statemap: IndexMap
            A map system that associate instance of StateID with an integer 
            index.
        rbase: tuple or list
            The bases of the Hilbert space before the operation.
        lbase: tuple or list, optional
            The bases of the Hilbert space after the operation.
            If not given or None, lbase is the same as rbase.
            default: None
        to_csr: boolean, optional, only keyword argument
            Whether to construct a csr_matrix as the result.
            default: True

        Return:
        -------
        res: csr_matrix or tuple
            The matrix representation of the creation or annihilation operator.
            If to_csr is True, the result is a csr_matrix, if False, the result
            is a tuple. The first is the non-zero matrix entries, the second and
            third are the row and col indices.
        """

        operator = (self.getStateID().getIndex(statemap), self.otype)
        return self.matrixFunc(operator, rbase=rbase, lbase=lbase, to_csr=to_csr)
    # }}}
# }}}


class SpinOperator(SiteID):# {{{
    """
    The class provide a unified description of a spin opeartor.

    Attribute:
    ----------
    otype: string
        The type of this spin operator. It can be only one of "x", "y", "z", 
        "p" or "m",which represents the five type spin operator respectively.
    site: np.ndarray
        The coordinate of the localized spin operator.
    Pauli: boolean, optional
        The attribute determine whether to use Pauli matrix or spin matrix.
        default: False

    Method:
    -------
    Special methods:
        __init__(otype, site, Pauli=False)
        __str__()
    General methods:
        tupleform()
        matrix()
        dagger()
        getSiteID()
        getOtype()
        sameSite(other)
        conjugateOf(other)
        update(*, otype=None, site=None)
        Schwinger()
        matrixRepr(sitemap)
    Static methods:
        matrixFunc(operator, totspin)
    Methods inherited from SiteID:
        __hash__()
        __lt__()
        __eq__()
        __gt__()
        __ne__()
        __le__()
        __ge__()
        getSite()
        getIndex(objmap)
    """

    def __init__(self, otype, site, Pauli=False):# {{{
        """
        Initialize instance of this class.

        Parameter:
        ----------
        See the documentation of this class.
        """

        SiteID.__init__(self, site=site)
        self.Pauli = Pauli
        if otype in SPIN_OTYPE:
            self.otype = otype
        else:
            raise TypeError("The invalid otype!")
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "otype: {0}\n".format(self.otype)
        info += SiteID.__str__(self)
        return info
    # }}}

    def tupleform(self):# {{{
        """
        Combine the above two attributes to a tuple, (otype, site)
        """

        return (self.otype, ) + SiteID.tupleform(self)
    # }}}

    def matrix(self):# {{{
        """
        Return the matrix representation of the spin operator specified by this 
        ID in the Hilbert space of the single spin, i.e. 2 dimension.
        """

        #SIGMA_MATRIX and SPIN_MATRIX are dict defined in constant module.
        #The keys of the dict are 'x', 'y', 'z', 'p', 'm' and the value are
        #cooresponding Pauli or spin matrix.
        if self.Pauli:
            matrix = SPIN_MATRIX["sigma"]
        else:
            matrix = SPIN_MATRIX['s']

        return matrix[self.otype]
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermitian conjuagte of this operator.
        """

        if self.otype in ('x', 'y', 'z'):
            return self.update()
        elif self.otype == 'p':
            return self.update(otype='m')
        elif self.otype == 'm':
            return self.update(otype='p')
        else:
            raise TypeError("The invalid otype!")
    # }}}

    def getSiteID(self):# {{{
        """
        Return the site id.
        """

        return SiteID(site=self.site)
    # }}}

    def getOtype(self):# {{{
        """
        Access the otype attribute of instance on this class.
        """

        return self.otype
    # }}}

    def conjugateOf(self, other):# {{{
        """
        Determine whether the instance is conjugate of the other instance.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")
        
        return self.dagger() == other
    # }}}

    def sameSite(self, other):# {{{
        """
        Determine whether the self SpinOperator and other SpinOperator is on 
        the same site.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.getSiteID() == other.getSiteID()
    # }}}
    
    def update(self, *, otype=None, site=None):# {{{
        """
        Create a new SpinOperator with the same parameter as self except 
        for those given to update method.

        All the parameters should be specified as keyword argument.

        Return:
        -------
        res: A new instance of SpinOperator.
        """

        if otype is None:
            otype = self.otype
        if site is None:
            site = self.site

        return SpinOperator(otype=otype, site=site, Pauli=self.Pauli)
    # }}}

    def Schwinger(self):# {{{
        """
        Return the Schwinger Fermion representation of this spin operator.
        """

        M = self.matrix()
        spins = [SPIN_UP, SPIN_DOWN]
        terms = []
        for spin0, row in zip(spins, range(2)):
            for spin1, col in zip(spins, range(2)):
                coeff = M[row, col]
                if coeff != 0:
                    C = AoC(otype=CREATION, site=self.site, spin=spin0)
                    A = AoC(otype=ANNIHILATION, site=self.site, spin=spin1)
                    term = ParticleTerm((C, A), coeff=coeff)
                    terms.append(term)
        return terms
    # }}}

    @staticmethod
    def matrixFunc(operator, totspin):# {{{
        """
        The static method to calculate the matrix representation of spin
        operator specified by the operator parameter.

        Parameter:
        ----------
        operator: tuple or list
            The operator should be of length 3. The first entry is the index of
            the site on which the spin operator is defined, the second is the
            type of the spin operator which should be only one of ('x', 'y',
            'z', 'p', m). The third determine whether to use sigma or S matrix
            and it can only be string "sigma" or 's'.
        totspin: int
            The total number of spins.

        Return:
        -------
        res: csr_matrix
            The matrix representation of this spin operator.
        """

        index, otype, sigma = operator
        I0 = identity(2 ** index, dtype=np.int64, format="csr")
        I1 = identity(2 ** (totspin - index - 1), dtype=np.int64, format="csr")
        S = csr_matrix(SPIN_MATRIX[sigma][otype])
        res = kron(I1, kron(S, I0, format="csr"), format="csr")
        res.eliminate_zeros()
        return res
    # }}}

    def matrixRepr(self, sitemap):# {{{
        """
        Return the matrix representation of the spin operator specified by this 
        ID in the Hilbert space of the manybody system.

        For every specific spin operator, its matrix representation in the
        Hilbert space is defined as follow: In *...* Ii₊₁*Si*Ii₋₁ *...* I₀
        where I is 2 dimension identity matrix, * represents tensor product and
        the subscripts are the index of these spin matrix.
        
        Parameter:
        ----------
        sitemap: IndexMap
            A map system that associate instance of SiteID with an integer index.

        Return:
        -------
        res: csr_matrix
            The matrix representation of the creation or annihilation operator.
        """

        totspin = len(sitemap)
        index = self.getSiteID().getIndex(sitemap)
        if self.Pauli:
            sigma = "sigma"
        else:
            sigma = 's'

        operator = (index, self.otype, sigma)
        return self.matrixFunc(operator, totspin)
    # }}}
# }}}


class SpinInteraction:# {{{
    """
    This class provide a unified description of spin interaction term.

    Attribute:
    ----------
    operators: A sequence of instances of SpinOperator class.
    coeff: int, float, complex, optional
        The coefficience of this term.
        default: 1.0

    Method:
    -------
    Special methods:
        __init__(operators, coeff=1.0)
        __str__()
        __mul__(other)
        __rmul__(other)
    General methods:
        dagger()
        sameAs(other)
        conjugateOf(other)
        updateCoeff(coeff)
        matrixRepr(sitemap, coeff)
    Static methods:
        matrixFunc(operators, totspin, coeff=1.0)
    """

    def __init__(self, operators, coeff=1.0):# {{{
        """
        Initialize instance of this class.

        The spin operators is sorted in ascending order according to their 
        SiteID. The relative position of two operators with the same SiteID 
        will not change and the exchange of two spin operators on different 
        site never change the interaction term.

        See also the documentation of this class.
        """
        
        if isinstance(coeff, (int, float, complex)):
            self.coeff = coeff
        else:
            raise TypeError("The input coeff is not a number.")
        
        for operator in operators:
            if not isinstance(operator, SpinOperator):
                raise TypeError("The input operator is not instance of "
                                "SpinOperator.")

        self.operators = sorted(operators, key=lambda item: item.getSiteID())
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "coeff: {0}\n".format(self.coeff)
        for item in self.operators:
            info += str(item)
            info += '\n'
        return info
    # }}}

    def __mul__(self, other):# {{{
        """
        Implement the binary arithmetic operation *, the other parameter is the
        right operand. Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            operators = self.operators + other.operators
            coeff = self.coeff * other.coeff
        elif isinstance(other, (int, float, complex)):
            operators = self.operators
            coeff = other * self.coeff
        else:
            raise TypeError("Multiply with the given parameter is not supported.")
        
        return SpinInteraction(operators, coeff=coeff)
    # }}}

    def __rmul__(self, other):# {{{
        """
        Implement the binary arithmetic operation *, the other parameter is the
        left operand.

        This method return a new instance of this class. If you just want to update the coeff, use updateCoeff() method instead.
        """
        
        if isinstance(other, (int, float, complex)):
            operators = self.operators
            coeff = other * self.coeff
        else:
            raise TypeError("Multiply with the given parameter is not supported.")

        return SpinInteraction(operators, coeff=coeff)
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermit conjugate of this operator.
        """

        coeff = self.coeff.conjugate()
        operators = []
        for operator in self.operators[::-1]:
            operators.append(operator.dagger())
        return SpinInteraction(operators, coeff=coeff)
    # }}}

    def sameAs(self, other):# {{{
        """
        Judge whether two instance of this class equal to each other.

        Here, the equal defines as follow, if the two instance are consisted of
        the same spin opertors, also these operators are in the same order, 
        then we claim the two instance are equal. We never care about the coeff
        attribute of the two instance.

        Warning:
        The logic of this method maybe invalid. Use with caution.
        """

        if isinstance(other, self.__class__):
            return self.operators == other.operators
        else:
            raise TypeError("The input parameter is not instance of this class!")
    # }}}

    def conjugateOf(self, other):# {{{
        """
        Determine whether the instance is conjugate of the other instance.

        Here, we also do not care about the coeff attribute.
        Warning:
        The logic of this method maybe invalid. Use with caution.
        """
        
        if isinstance(other, self.__class__):
            return self.sameAs(other.dagger())
        else:
            raise TypeError("The input parameter is not instance of this class!")
    # }}}

    def updateCoeff(self, coeff):# {{{
        """
        Update the coeff attribute of instance of this class.
        """

        if isinstance(coeff, (int, float, complex)):
            self.coeff = coeff
        else:
            raise TypeError("The wrong type of coeff parameter!")
    # }}}

    def Schwinger(self):# {{{
        """
        Return the Schwinger Fermion representation of this spin interaction
        term.
        """
        
        fermion_reprs = []
        for operator in self.operators:
            fermion_reprs.append(operator.Schwinger())

        terms = []
        for term in product(*fermion_reprs):
            res_term = 1
            for sub_term in term:
                res_term = res_term * sub_term
            terms.append(res_term)
        return terms
    # }}}
    
    @staticmethod
    def matrixFunc(operators, totspin, coeff=1.0):# {{{
        """
        This static method calculate the matrix representation of spin
        interaction specified by the operators parameter.

        Parameter:
        ----------
        operators: sequence
            A sequence of length 3 tuple. The first entry of the tuple is an
            integer index of the spin operator. The second entry is the type
            of the spin operator which should be only one of ('x', 'y', 'z',
            'p', 'm'). The third determine whether to use sigma or S matrix and
            it can only be "sigma" or "s".
        totspin: int
            The total number of spins concerned.
        coeff: int, float or complex, optional
            The coefficience of the term.
            default: 1.0

        Return:
        -------
        res: csr_matrix
            The matrix representation of this term.
        """

        operators = sorted(operators, key=lambda item: item[0])
        if len(operators) == 2 and operators[0][0] != operators[1][0]:
            index0, otype0, sigma0 = operators[0]
            index1, otype1, sigma1 = operators[1]
            S0 = csr_matrix(SPIN_MATRIX[sigma0][otype0])
            S1 = csr_matrix(SPIN_MATRIX[sigma1][otype1])
            dim0 = 2 ** (index0)
            dim1 = 2 ** (index1 - index0 - 1)
            dim2 = 2 ** (totspin - index1 - 1)
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
            res = identity(2**totspin, dtype=np.int64, format="csr")
            for index, otype, sigma in operators:
                I0 = identity(2**index, dtype=np.int64, format="csr")
                I1 = identity(2**(totspin-index-1), dtype=np.int64, format="csr")
                S = csr_matrix(SPIN_MATRIX[sigma][otype])
                res = res.dot(kron(I1, kron(S, I0, format="csr"), format="csr"))
        
        res.eliminate_zeros()
        return coeff * res
    # }}}

    def matrixRepr(self, sitemap, coeff=None):# {{{
        """
        Return the matrix representation of this term in Hilbert space.

        Parameter:
        ----------
        sitemap: class IndexMap
            The map that associate integer index with SiteID.
        coeff: int, float or complex, optional
            The given coefficience of this term.
            default: None
        
        Return:
        -------
        res: csr_matrix
            The matrix representation of this term.
        """

        if coeff is not None:
            self.updateCoeff(coeff)

        totspin = len(sitemap)
        operators = []
        for operator in self.operators:
            index = operator.getSiteID().getIndex(sitemap)
            otype = operator.getOtype()
            if operator.Pauli:
                sigma = "sigma"
            else:
                sigma = 's'
            operators.append((index, otype, sigma))
        res = self.matrixFunc(operators, totspin, self.coeff)
        return res
    # }}}
# }}}


class ParticleTerm:# {{{
    """
    This class provide unified description of any operator 
    composed of creation and/or annihilation operators.

    Attribute:
    ----------
    aocs: tuple
        The creation and annihilation operators that consist of this operator.
    coeff: float, int or complex, optional
        The coefficient of the operator.
        default: 1.0
    normalized: boolean
        A flag that indicates whether the instance is normalized. See document
        of normalize method for the definition of normalized term.
    
    Method:
    -------
    Special methods:
        __init__(aocs, coeff=1.0)
        __str__()
        __mul__(other)
        __rmul__(other)
    General methods:
        isPairing(tag=None)
        isHopping(tag=None)
        isHubbard()
        sameAs(other)
        dagger()
        conjugateOf(other)
        updateCoeff(coeff)
        matrixRepr(statemap, rbase, lbase=None, coeff=None)
    Static methods:
        normalize(aoc_seq)
        matrixFunc(operators, rbase, lbase=None, coeff=1.0)
    """

    def __init__(self, aocs, coeff=1.0):# {{{
        """
        Initialize the instance of this class.

        Paramter:
        --------
        aocs: tuple or list
            The creation and annihilation operators that consist this operator.
        coeff: float, int or complex, optional
            The coefficience of the operator.
            default: 1.0
        """

        for aoc in aocs:
            if not isinstance(aoc, AoC):
                raise TypeError("The input parameter is not instance of AoC.")

        try:
            normal_aocs, swap_num = self.normalize(aocs)
            self.normalized = True
        except SwapFermionError:
            normal_aocs = list(aocs)
            swap_num = 0
            self.normalized = False

        self.aocs = normal_aocs
        self.coeff = (SWAP_FACTOR_F ** swap_num) * coeff
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "coeff: {0}\n".format(self.coeff)
        for aoc in self.aocs:
            info += str(aoc)
            info += '\n'
        return info
    # }}}

    def __mul__(self, other):# {{{
        """
        Implement the binary arithmetic operation *, the other parameter is the
        right operand. Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            aocs = self.aocs + other.aocs
            coeff = self.coeff * other.coeff
        elif isinstance(other, (int, float, complex)):
            aocs = self.aocs
            coeff = other * self.coeff
        else:
            raise TypeError("Multiply with the given parameter is not supported.")

        return ParticleTerm(aocs, coeff=coeff)
    # }}}
    
    def __rmul__(self, other):# {{{
        """
        Implement the binary arithmetic operation *, the other parameter is the
        left operand.

        This method return a new instance of this class. If you just want to 
        update the coeff attribute but not create a new instance, use the 
        updateCoeff(coeff) method.
        """

        if isinstance(other, (int, float, complex)):
            aocs = self.aocs
            coeff = other * self.coeff
        else:
            raise TypeError("Multiply with the given parameter is not supported.")

        return ParticleTerm(aocs, coeff=coeff)
    # }}}

    @staticmethod
    def normalize(aoc_seq):# {{{
        """
        Reordering a sequence of creation and/or annihilation operators into 
        norm form.

        For a composite operator consist of creation and annihilation operators,
        the norm form means that all the creation operators appear to the left
        of all the annihilation operators. Also, the creation and annihilation
        operators are sorted in ascending and descending order respectively
        according to the single particle state associated with the operator.

        See the document of AoC.__lt__(other) method for the comparsion logic.

        Parameter:
        ----------
        aoc_seq: list or tuple
            A collection of creation and/or annihilation operators.

        Return:
        -------
        res: list
            The norm form of the operator.
        
        Raise:
        ------
        SwapFermionError: Exceptions raised when swap creation and annihilation
        operator that with the same single particle state.
        """

        seq = list(aoc_seq[:])
        seq_len = len(aoc_seq)
        swap_num = 0

        for length in range(seq_len, 1, -1):
            for i in range(0, length-1):
                aoc0 = seq[i]
                aoc1 = seq[i+1]
                id0 = aoc0.getStateID()
                id1 = aoc1.getStateID()
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
        """
        Determining whether the operator is a pairing term.

        This method is only valid for these instance that the "normalized"
        attribute is set to be True. For these instance that normalized
        attribute is False, this method will raise NotImplementedError.

        Parameter:
        ----------
        tag: string, optional
            Determine the judgment criteria.
            If tag is none, both particle and hole pairing is ok, if tag is 'p'
            then only particle is ok and if tag is 'h' only hole pairing is ok.
            default: None
        """

        if self.normalized:
            if len(self.aocs) != 2:
                return False
            else:
                otype0 = self.aocs[0].getOtype()
                otype1 = self.aocs[1].getOtype()
                if tag in ('p', 'P'):
                    if otype0 == CREATION and otype1 == CREATION:
                        return True
                    else:
                        return False
                elif tag in ('h', 'H'):
                    if otype0 == ANNIHILATION and otype1 == ANNIHILATION:
                        return True
                    else:
                        return False
                else:
                    if otype0 == otype1:
                        return True
                    else:
                        return False
        else:
            raise NotImplementedError("The term is not normalied, We can't "
                "determine it is a pairing term or not using simple logical.")
    # }}}

    def isHopping(self, tag=None):# {{{
        """
        Determining whether the operator is a hopping term.
        
        This method is only valid for these instance that the "normalized"
        attribute is set to be True. For these instance that normalized
        attribute is False, this method will raise NotImplementedError.

        Parameter:
        ----------
        tag: string, optional
            Determine the judgment criteria.
            If tag is none, arbitrary hopping term is ok, and if tag is 'n' only
            the number operator is ok.
            default: None
        """

        if self.normalized:
            if len(self.aocs) != 2:
                return False
            else:
                c0 = self.aocs[0].getOtype() == CREATION
                c1 = self.aocs[1].getOtype() == ANNIHILATION
                if tag in ('n', 'N'):
                    c2 = self.aocs[0].sameState(self.aocs[1])
                    if c0 and c1 and c2:
                        return True
                    else:
                        return False
                else:
                    if c0 and c1:
                        return True
                    else:
                        return False
        else:
            raise NotImplementedError("The term is not normalied, We can't "
                "determine it is a hopping term or not using simple logical.")
    # }}}
    
    def isHubbard(self):# {{{
        """
        Determining whether the operator is a hubbard term.
        
        This method is only valid for these instance that the "normalized"
        attribute is set to be True. For these instance that normalized
        attribute is False, this method will raise NotImplementedError.

        """

        if self.normalized:
            if len(self.aocs) == 4:
                if (self.aocs[0].getOtype() == CREATION and
                    self.aocs[1].getOtype() == CREATION and
                    self.aocs[2].getOtype() == ANNIHILATION and
                    self.aocs[3].getOtype() == ANNIHILATION and
                    self.aocs[0].sameState(self.aocs[3]) and
                    self.aocs[1].sameState(self.aocs[2])
                    ):
                    return True
                else:
                    return False
            else:
                return False
        else:
            raise NotImplementedError("The term is not normalied, We can't "
                "determine it is a hubbard term or not using simple logical.")
    # }}}

    def sameAs(self, other):# {{{
        """
        Judge whether two instance of this class equal to each other.

        Here, the equal defines as follow, if the two instance are consisted of
        the same creation and/or annihilation opertors, also these operators are
        in the same order, then we claim the two instance are equal. We never
        care about the coeff attribute of the two instance.

        Warnning:
            The logic of this method maybe incorrect, use this method with
            caution.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.aocs == other.aocs
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermit conjugate of this operator.
        """

        aocs = []
        for aoc in self.aocs[::-1]:
            aocs.append(aoc.dagger())
        return ParticleTerm(aocs, coeff=self.coeff.conjugate())
    # }}}

    def conjugateOf(self, other):# {{{
        """
        Determine whether an operator is the Hermit conjugate of it's self.

        Warnning:
            The logic of this method maybe incorrect, use this method with
            caution.
        """
        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.sameAs(other.dagger())
    # }}}

    def updateCoeff(self, coeff):# {{{
        """
        Update the coeff attribute of instance of this class.
        """
        
        if isinstance(coeff, (int, float, complex)):
            self.coeff = coeff
        else:
            raise TypeError("The input coeff parameter is of invalid type!")
    # }}}

    @staticmethod
    def matrixFunc(operators, rbase, lbase=None, coeff=1.0):# {{{
        """
        Return the matrix representation of the term in the Hilbert space 
        specified by the rbase and the optional lbase.

        Parameter:
        ----------
        operators: list or tuple
            It is a sequence of length-2 tuples or lists. The first entry is the
            index of the state and the second is the operator type(CREATION or
            ANNIHILATION).
        rbase: tuple or list
            The bases of the Hilbert space before the operation.
        lbase: tuple or list, optional
            The bases of the Hilbert space after the operation.
            It not given or None, lbase is the same as rbase.
            default: None
        coeff: int, float or complex, optional
            The coefficience of the term.
            default: 1.0

        Return:
        -------
        res: csr_martrix
            The matrix representation of the term.
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

    def matrixRepr(self, statemap, rbase, lbase=None, coeff=None):# {{{
        """
        Return the matrix representation of the operator specified by this
        instance in the Hilbert space of the manybody system.
        
        Parameter:
        ----------
        statemap: IndexMap
            A map system that associate instance of StateID with an integer 
            index.
        rbase: tuple or list
            The bases of the Hilbert space before the operation.
        lbase: tuple or list, optional
            The bases of the Hilbert space after the operation.
            It not given or None, lbase is the same as rbase.
            default: None
        coeff: int, float or complex, optional
            The coefficience of the term.
            default: None

        Return:
        -------
        res: csr_martrix
            The matrix representation of the term.
        """

        if coeff is not None:
            self.updateCoeff(coeff)

        aocs = []
        for aoc in self.aocs:
            index = aoc.getStateID().getIndex(statemap)
            otype = aoc.otype
            aocs.append((index, otype))
        return self.matrixFunc(aocs, rbase, lbase=lbase, coeff=self.coeff)
    # }}}
# }}}