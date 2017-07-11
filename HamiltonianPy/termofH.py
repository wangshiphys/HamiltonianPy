from itertools import product
from scipy.sparse import csr_matrix, identity, kron

import numpy as np

from HamiltonianPy.constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP
from HamiltonianPy.exception import SwapFermionError
from HamiltonianPy.indexmap import IndexMap

import HamiltonianPy.extpkg.matrixrepr as cextmr

__all__ = ["SiteID", "StateID", "AoC", "SpinOperator", 
           "SpinInteraction", "ParticleTerm"]

#Useful constants
ZOOM = 10000
SWAP_FACTOR_F = -1
SPIN_OTYPES = ('x', 'y', 'z', 'p', 'm')
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.int64)
SIGMA_Y = np.array([[0, -1], [1, 0]], dtype=np.int64) * 1j
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.int64)
SIGMA_P = np.array([[0, 2], [0, 0]], dtype=np.int64)
SIGMA_M = np.array([[0, 0], [2, 0]], dtype=np.int64)
SIGMA_MATRIX = {'x': SIGMA_X, 'y': SIGMA_Y, 'z': SIGMA_Z, 
                'p': SIGMA_P, 'm':SIGMA_M}
###########################################################


class SiteID:# {{{
    """
    A wrapper of 1D np.ndarray which represents the coordinates of a point.

    The reason to define this wrapper is to make the coordinates hashable as 
    well as comparable as a whole.

    Attributes
    ----------
    site : np.ndarray
        The coordinates of the point. The shape of this array should be only
        one of (1, ), (2, ) or (3, ).

    Methods
    -------
    Public methods:
        getSite()
        getIndex(sitemap)
    Special methods:
        __init__(site)
        __str__()
        __hash__()
        __lt__(other)
        __eq__(other)
        __gt__(other)
        __le__(other)
        __ne__(other)
        __ge__(other)
    """

    def __init__(self, site):# {{{
        if isinstance(site, np.ndarray) and site.shape in [(1,), (2,), (3,)]:
            self._site = np.array(site, copy=True)
            self._site.setflags(write=False)
        else:
            raise TypeError("The invalid site parameter.")
    # }}}

    @property
    def site(self):# {{{
        """
        The site attribute of instance of this class.
        """

        return np.array(self._site, copy=True)
    # }}}

    def getSite(self):# {{{
        """
        Access the site attribute of instance of this class.
        """

        return np.array(self._site, copy=True)
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        return "site: " + str(self._site)
    # }}}

    def __hash__(self):# {{{
        """
        Return the hash value of instance of this class.
        """

        return hash(self._tupleform())
    # }}}

    def __lt__(self, other):# {{{
        """
        Define the < operator between instance of this class.
        """

        if isinstance(other, self.__class__):
            return self._tupleform() < other._tupleform()
        else:
            raise TypeError("The rigth operand is not instance of this class.")
    # }}}

    def __eq__(self, other):# {{{
        """
        Define the == operator between instance of this class.
        """

        if isinstance(other, self.__class__):
            return self._tupleform() == other._tupleform()
        else:
            raise TypeError("The rigth operand is not instance of this class.")
    # }}}

    def __gt__(self, other):# {{{
        """
        Define the > operator between instance of this class.
        """

        if isinstance(other, self.__class__):
            return self._tupleform() > other._tupleform()
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

    def _tupleform(self):# {{{
        #The tuple form of the coordinates.
        #This method is useful in calculating the hash value of instance of
        #this class as well as defining compare logic between instances of
        #this class. This method should only be used internally.
        
        return tuple([int(i) for i in ZOOM * self._site])
    # }}}

    def getIndex(self, sitemap):# {{{
        """
        Return the index associated with this SiteID.

        Parameters
        ----------
        sitemap : IndexMap
            A map system that associate instance
            of SiteID with an integer index.

        Returns
        -------
        res : int
            The index of self in the map system.
        """

        if not isinstance(sitemap, IndexMap):
            raise TypeError("The input sitemap is not instance an IndexMap!")

        return sitemap(self)
    # }}}
# }}}


class StateID(SiteID):# {{{
    """
    This class provide a unified description of a single particle state.

    Attributes
    ----------
    site : ndarray
        The coordinate of the localized state. The site attribute should be 
        a 1D array, usually it has length 1, 2 or 3 cooresponding to 1, 2 or 
        3 space dimension.
    spin : int, optional
        The spin index of the state.
        default: 0
    orbit : int, optional
        The orbit index of the state.
        default: 0

    Methods
    -------
    Public methods:
        getSpin()
        getOrbit()
    Special methods:
        __init__(site, spin=0, orbit=0)
        __str__()
    Inherited from SiteID:
        __hash__()
        __eq__(other)
        __lt__(other)
        __gt__(other)
        __ne__(other)
        __le__(other)
        __ge__(other)
        getSite()
        getIndex(statemap)
    """

    def __init__(self, site, spin=0, orbit=0):# {{{
        SiteID.__init__(self, site=site)

        if isinstance(spin, int) and spin >= 0:
            self._spin = spin
        else:
            raise ValueError("The invalid spin parameter!")

        if isinstance(orbit, int) and orbit >= 0:
            self._orbit = orbit
        else:
            raise ValueError("The invalid orbit parameter!")
    # }}}

    @property
    def spin(self):# {{{
        """
        The spin attribute of instance of this class.
        """

        return self._spin
    # }}}

    def getSpin(self):# {{{
        """
        Access the spin attribute of instance of this class.
        """

        return self._spin
    # }}}

    @property
    def orbit(self):# {{{
        """
        The orbit attribute of instance of this class.
        """

        return self._orbit
    # }}}
    
    def getOrbit(self):# {{{
        """
        Access the Orbit attribute of instance of this class.
        """

        return self._orbit
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "orbit: {0}\nspin: {1}\n".format(self._orbit, self._spin)
        info += SiteID.__str__(self)
        return info
    # }}}

    def _tupleform(self):# {{{
        #Combine the above three attributes to a tuple, (orbit, spin, site)

        return (self._orbit, self._spin, SiteID._tupleform(self))
    # }}}
# }}}


class AoC(StateID):# {{{
    """
    This class provide a unified description of the creation and annihilation operator.

    Attributes
    ----------
    otype : int
        The type of this operator. It can be either 0 or 1, wich represents
        annihilation or creation respectively.
    site : ndarray
        The coordinate of the localized state. The site attribute should be
        a 1D array, usually it has length 1, 2 or 3 cooresponding to 1, 2 or 
        space dimension.
    spin : int, optional
        The spin index of the state.
        default: 0
    orbit : int, optional
        The orbit index of the state.
        default: 0

    Methods
    -------
    Public methods:
        getOtype()
        getStateID()
        getStateIndex(statemap)
        dagger()
        conjugateOf(other)
        sameState(other)
        derive(*, otype=None, site=None, spin=None, orbit=None)
        matrixRepr(statemap, rbase, *, lbase=None, to_csr=True)
    Static methods:
        matrixFunc(operator, rbase, *, lbase=None, to_csr=True)
    Special methods:
        __init__(otype, site, spin=0, orbit=0)
        __str__()
        __gt__(other)
        __lt__(other)
    Inherit from SiteID:
        __hash__()
        __eq__(other)
        __ne__(other)
        __ge__(other)
        __le__(other)
        getSite()
        getIndex(aocmap)
    Inherit from StateID:
        getSpin()
        getOrbit()
    """

    def __init__(self, otype, site, spin=0, orbit=0):# {{{
        StateID.__init__(self, site=site, spin=spin, orbit=orbit)

        if otype in (ANNIHILATION, CREATION):
            self._otype = otype
        else:
            raise ValueError("The invalid otype!")
    # }}}

    @property
    def otype(self):# {{{
        """
        The otype attribute of instance of this class. 
        """

        return self._otype
    # }}}

    def getOtype(self):# {{{
        """
        Access the otype attribute of instance of this class.
        """

        return self._otype
    # }}}

    def getStateID(self):# {{{
        """
        Extract the state information of this creation or annihilation operator.

        Returns
        -------
        res : A new instance of StateID class.
        """

        return StateID(site=self._site, spin=self._spin, orbit=self._orbit)
    # }}}

    def getStateIndex(self, statemap):# {{{
        """
        Return the index of the state associated with this operator.

        Parameters
        ----------
        statemap : IndexMap
            A map system that associate instance
            of StateID with an integer index.

        Returns
        -------
        res : int
            The index of the state.
        """

        return self.getStateID().getIndex(statemap)
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describles the content of the instance.
        """

        info = 'otype: {0}\n'.format(self._otype)
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
            otype0 = self._otype
            otype1 = other._otype
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
            otype0 = self._otype
            otype1 = other._otype
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

    def _tupleform(self):# {{{
        #Combine the above four attributes to a tuple, (otype, orbit, spin, site)

        return (self._otype, ) + StateID._tupleform(self)
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermitian conjugate of self.

        Returns
        -------
        res : A new instance of this class.
        """

        if self._otype == CREATION:
            otype = ANNIHILATION
        else:
            otype = CREATION
        return self.derive(otype=otype)
    # }}}

    def conjugateOf(self, other):# {{{
        """
        Determine whether self is the Hermitian conjugate of other.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The other parameter is not instance of this class!")
        
        return self.dagger() == other
    # }}}

    def sameState(self, other):# {{{
        """
        Determine whether self and other is of the same state.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The other parameter is not instance of this class!")

        return self.getStateID() == other.getStateID()
    # }}}

    def derive(self, *, otype=None, site=None, spin=None, orbit=None):# {{{
        """
        Derive a new aoc with the same parameter as self except for those 
        given to this method.
        
        All the parameters should be specified as keyword argument.

        Returns
        -------
        res : A new instance of AoC.
        """

        if otype is None:
            otype = self._otype
        if site is None:
            site = self._site
        if spin is None:
            spin = self._spin
        if orbit is None:
            orbit = self._orbit

        return AoC(otype=otype, site=site, spin=spin, orbit=orbit)
    # }}}

    @staticmethod
    def matrixFunc(operator, rbase, *, lbase=None, to_csr=True):# {{{
        """
        Return the matrix representation of the operator in the Hilbert space
        specified by the rbase and optional lbase.

        Parameters
        ----------
        operator : tuple or list
            The parameter should be a tuple or list with two entries. The first
            entries is the index of the single-particle state and the second is
            the "CREATION" or "ANNIHILATION" constant.
        rbase : tuple or list
            The bases of the Hilbert space before the operation.
        lbase : tuple or list, optional
            The bases of the Hilbert space after the operation.
            If not given or None, lbase is the same as rbase.
            default: None
        to_csr : boolean, optional, only keyword argument
            Whether to construct a csr_matrix as the result.
            default: True

        Returns
        -------
        res : csr_matrix or tuple
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
        
        Parameters
        ----------
        statemap : IndexMap
            A map system that associate instance of StateID with an integer index.
        rbase : tuple or list
            The bases of the Hilbert space before the operation.
        lbase : tuple or list, optional
            The bases of the Hilbert space after the operation.
            If not given or None, lbase is the same as rbase.
            default: None
        to_csr : boolean, optional, only keyword argument
            Whether to construct a csr_matrix as the result.
            default: True

        Returns
        -------
        res : csr_matrix or tuple
            The matrix representation of the creation or annihilation operator.
            If to_csr is True, the result is a csr_matrix, if False, the result
            is a tuple. The first is the non-zero matrix entries, the second and
            third are the row and col indices.
        """

        operator = (self.getStateIndex(statemap), self._otype)
        return self.matrixFunc(operator, rbase=rbase, lbase=lbase, to_csr=to_csr)
    # }}}
# }}}


class SpinOperator(SiteID):# {{{
    """
    The class provide a unified description of a spin opeartor.

    Attributes
    ----------
    otype : string
        The type of this spin operator. It can be only one of "x", "y", "z", 
        "p" or "m",which represents the five type spin operator respectively.
    site : np.ndarray
        The site on which the spin operator is defined.
    Pauli : boolean, optional, only keyword argument
        The attribute determine whether to use Pauli matrix or spin matrix.
        default: False

    Methods
    -------
    Public methods:
        getOtype()
        getSiteID()
        getSiteIndex(sitemap)
        matrix()
        dagger()
        conjugateOf(other)
        sameSite(other)
        derive(*, otype=None, site=None)
        Schwinger()
        matrixRepr(sitemap)
    Static methods:
        matrixFunc(operator, totspin, *, Pauli=False)
    Special methods:
        __init__(otype, site, Pauli=False)
        __str__()
    Inherited from SiteID:
        __hash__()
        __lt__()
        __eq__()
        __gt__()
        __ne__()
        __le__()
        __ge__()
        getSite()
        getIndex(operatormap)
    """

    def __init__(self, otype, site, *, Pauli=False):# {{{
        SiteID.__init__(self, site=site)
        self.Pauli = Pauli
        if otype in SPIN_OTYPES:
            self._otype = otype
        else:
            raise TypeError("The invalid otype!")
    # }}}

    @property
    def otype(self):# {{{
        """
        The otype attribute of instance of this class.
        """

        return self._otype
    # }}}

    def getOtype(self):# {{{
        """
        Access the otype attribute of instance on this class.
        """

        return self._otype
    # }}}

    def getSiteID(self):# {{{
        """
        Return the site id.
        """

        return SiteID(site=self._site)
    # }}}

    def getSiteIndex(self, sitemap):# {{{
        """
        Return the index of the site associated with this spin operator.
        
        Parameters
        ----------
        sitemap : IndexMap
            A map system that associate instance of SiteID with an integer index.

        Returns
        -------
        res : int
            The index of the site attribute of the self.
        """
        
        return self.getSiteID().getIndex(sitemap)
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "otype: {0}\n".format(self._otype)
        info += SiteID.__str__(self)
        return info
    # }}}

    def _tupleform(self):# {{{
        #Combine the above two attributes to a tuple, (otype, site)

        return (self._otype, SiteID._tupleform(self)) 
    # }}}

    def matrix(self):# {{{
        """
        Return the matrix representation of the spin operator specified by this 
        ID in the Hilbert space of the single spin, i.e. 2 dimension.
        """

        if self.Pauli:
            Pauli_factor = 1
        else:
            Pauli_factor = 0.5
        return Pauli_factor * np.array(SIGMA_MATRIX[self._otype], copy=True)
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermit conjuagte of this operator.
        """

        if self._otype == 'p':
            return self.derive(otype='m')
        elif self._otype == 'm':
            return self.derive(otype='p')
        else:
            return self.derive()
    # }}}

    def conjugateOf(self, other):# {{{
        """
        Determine whether self is Hermit conjugate of other.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The other parameter is not instance of this class!")
        
        return self.dagger() == other
    # }}}

    def sameSite(self, other):# {{{
        """
        Determine whether the self and other is defined on the same site.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The other parameter is not instance of this class!")

        return self.getSiteID() == other.getSiteID()
    # }}}
    
    def derive(self, *, otype=None, site=None):# {{{
        """
        Create a new SpinOperator with the same parameter as self except 
        for those given to this method.

        All the parameters should be specified as keyword argument.

        Returns
        -------
        res : A new instance of SpinOperator.
        """

        if otype is None:
            otype = self._otype
        if site is None:
            site = self._site

        return SpinOperator(otype=otype, site=site, Pauli=self.Pauli)
    # }}}

    def Schwinger(self):# {{{
        """
        Return the Schwinger Fermion representation of this spin operator.

        Notes:
            The coeff of the generating ParticleTerm is related to the Pauli
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
    def matrixFunc(operator, totspin, *, Pauli=False):# {{{
        """
        The static method to calculate the matrix representation of spin
        operator specified by the operator parameter.

        Parameters
        ----------
        operator : tuple or list
            The operator should be of length 2. The first entry is the index of
            the site on which the spin operator is defined, the second is the
            type of the spin operator which should be only one of ('x', 'y',
            'z', 'p', m). 
        totspin : int
            The total number of spins.
        Pauli : boolean, keyword-only, optional
            This parameter determine whether to use Pauli matrix or spin matrix.
            default: False

        Returns
        -------
        res : csr_matrix
            The matrix representation of this spin operator.
        """

        if Pauli:
            Pauli_factor = 1
        else:
            Pauli_factor = 0.5

        index, otype = operator
        I0 = identity(1<<index, dtype=np.int64, format="csr")
        I1 = identity(1<<(totspin - index - 1), dtype=np.int64, format="csr")
        S = Pauli_factor * csr_matrix(SIGMA_MATRIX[otype])
        return kron(I1, kron(S, I0, format="csr"), format="csr")
    # }}}

    def matrixRepr(self, sitemap):# {{{
        """
        Return the matrix representation of the spin operator specified by this 
        ID in the Hilbert space of the manybody system.

        For every specific spin operator, its matrix representation in the
        Hilbert space is defined as follow: In *...* Ii₊₁*Si*Ii₋₁ *...* I₀
        where I is 2 dimension identity matrix, * represents tensor product and
        the subscripts are the index of these spin matrix.
        
        Parameters
        ----------
        sitemap : IndexMap
            A map system that associate instance of SiteID with an integer index.

        Returns
        -------
        res : csr_matrix
            The matrix representation of the creation or annihilation operator.
        """

        operator = (self.getSiteIndex(sitemap), self._otype)
        return self.matrixFunc(operator, totspin=len(sitemap), Pauli=self.Pauli)
    # }}}
# }}}


class SpinInteraction:# {{{
    """
    This class provide a unified description of spin interaction term.

    Methods
    -------
    Public methods:
        dagger()
        sameAs(other)
        conjugateOf(other)
        setCoeff(coeff)
        Schwinger()
        matrixRepr(sitemap, coeff)
    Special methods:
        __init__(operators, coeff=1.0)
        __str__()
        __mul__(other)
        __rmul__(other)
    Static methods:
        matrixFunc(operators, totspin, coeff=1.0, *, Pauli=False)
    """

    def __init__(self, operators, coeff=1.0):# {{{
        """
        Initialize instance of this class.

        Parameters
        ----------
        operators : A sequence of instances of SpinOperator class.
        coeff : int, float, complex, optional
            The coefficience of this term.
            default: 1.0
        """
        
        if isinstance(coeff, (int, float, complex)):
            self._coeff = coeff
        else:
            raise TypeError("The input coeff is not a number.")

        #Check whether all entries is SpinOperator.
        for operator in operators:
            if not isinstance(operator, SpinOperator):
                errmsg = "The input operator is not instance of SpinOperator."
                raise TypeError(errmsg)

        #Sorting the spin operators in ascending order according to their
        #SiteID. The relative position of two operators with the same SiteID
        #will not change and the exchange of two spin operators on different
        #site never change the interaction term.
        self._operators = sorted(operators, key=lambda item: item.getSiteID())
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "coeff: {0}\n".format(self._coeff)
        for operator in self._operators:
            info += str(operator) + '\n'
        return info
    # }}}

    def __mul__(self, other):# {{{
        """
        Implement the binary arithmetic operation *, the other parameter is the
        right operand. Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            operators = self._operators + other._operators
            coeff = self._coeff * other._coeff
        elif isinstance(other, (int, float, complex)):
            operators = self._operators
            coeff = self._coeff * other
        else:
            raise TypeError("Multiply with the given parameter is not supported.")
        
        return SpinInteraction(operators, coeff=coeff)
    # }}}

    def __rmul__(self, other):# {{{
        """
        Implement the binary arithmetic operation *, the other parameter is the
        left operand.

        This method return a new instance of this class. If you just want to
        update the coeff, use updateCoeff() method instead.
        """
        
        if isinstance(other, (int, float, complex)):
            operators = self._operators
            coeff = other * self._coeff
        else:
            raise TypeError("Multiply with the given parameter is not supported.")

        return SpinInteraction(operators, coeff=coeff)
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermit conjugate of this operator.
        """

        operators = [operator.dagger() for operator in self._operators[::-1]]
        return SpinInteraction(operators, coeff=self._coeff.conjugate())
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
            return self._operators == other._operators
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

    def setCoeff(self, coeff):# {{{
        """
        Set the coeff attribute of instance of this class.
        """

        if isinstance(coeff, (int, float, complex)):
            self._coeff = coeff
        else:
            raise TypeError("The wrong type of coeff parameter!")
    # }}}

    def Schwinger(self):# {{{
        """
        Return the Schwinger Fermion representation of this spin interaction term.
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
    def matrixFunc(operators, totspin, coeff=1.0, *, Pauli=False):# {{{
        """
        This static method calculate the matrix representation of spin
        interaction specified by the operators parameter.

        Parameters
        ----------
        operators : sequence
            A sequence of length 2 tuple. The first entry of the tuple is an
            integer index of the spin operator. The second entry is the type
            of the spin operator which should be only one of ('x', 'y', 'z',
            'p', 'm').
        totspin: int
            The total number of spins concerned.
        coeff : int, float or complex, optional
            The coefficience of the term.
            default: 1.0

        Returns
        -------
        res : csr_matrix
            The matrix representation of this term.
        """

        if Pauli:
            Pauli_factor = 1
        else:
            Pauli_factor = 0.5

        operators = sorted(operators, key=lambda item: item[0])
        if len(operators) == 2 and operators[0][0] != operators[1][0]:
            index0, otype0 = operators[0]
            index1, otype1 = operators[1]
            #The coeff of this term is multiplied to S0 operator.
            S0 = (coeff * Pauli_factor) * csr_matrix(SIGMA_MATRIX[otype0])
            S1 = Pauli_factor * csr_matrix(SIGMA_MATRIX[otype1])
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
                S = Pauli_factor * csr_matrix(SIGMA_MATRIX[otype])
                res = res.dot(kron(I1, kron(S, I0, format="csr"), format="csr"))
        
        return res
    # }}}

    def matrixRepr(self, sitemap, coeff=None):# {{{
        """
        Return the matrix representation of this term in Hilbert space.

        Parameters
        ----------
        sitemap : class IndexMap
            The map that associate integer index with SiteID.
        coeff : int, float or complex, optional
            The given coefficience of this term.
            default: None
        
        Returns
        -------
        res : csr_matrix
            The matrix representation of this term.
        """

        if coeff is not None:
            self.setCoeff(coeff)

        operators = []
        for operator in self._operators:
            index = operator.getSiteIndex(sitemap)
            otype = operator.getOtype()
            operators.append((index, otype))
        Pauli = operator.Pauli
        res = self.matrixFunc(operators=operators, totspin=len(sitemap),
                              coeff=self._coeff, Pauli=Pauli)
        return res
    # }}}
# }}}


class ParticleTerm:# {{{
    """
    This class provide unified description of any operator 
    composed of creation and/or annihilation operators.

    Methods
    -------
    Public methods:
        isPairing(tag=None)
        isHopping(tag=None)
        isHubbard()
        dagger()
        sameAs(other)
        conjugateOf(other)
        setCoeff(coeff)
        matrixRepr(statemap, rbase, *, lbase=None, coeff=None)
    Special methods:
        __init__(aocs, coeff=1.0)
        __str__()
        __mul__(other)
        __rmul__(other)
    Static methods:
        normalize(aoc_seq)
        matrixFunc(operators, rbase, *, lbase=None, coeff=1.0)
    """

    def __init__(self, aocs, coeff=1.0):# {{{
        """
        Initialize the instance of this class.

        Paramters
        --------
        aocs : tuple or list
            The creation and/or annihilation operators that consist this operator.
        coeff : float, int or complex, optional
            The coefficience of the operator.
            default: 1.0
        """
        
        #Check that every entries is of AoC type.
        for aoc in aocs:
            if not isinstance(aoc, AoC):
                raise TypeError("The input parameter is not instance of AoC.")

        #The _normalized attribute indicates whether the instance is normalized.
        #See docstring of normalize method for the definition of normalized term.
        try:
            normal_aocs, swap_num = self.normalize(aocs)
            self._normalized = True
        except SwapFermionError:
            normal_aocs = list(aocs)
            swap_num = 0
            self._normalized = False

        self._aocs = normal_aocs
        self._coeff = (SWAP_FACTOR_F ** swap_num) * coeff
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "coeff: {0}\n".format(self._coeff)
        for aoc in self._aocs:
            info += str(aoc) + '\n'
        return info
    # }}}

    def __mul__(self, other):# {{{
        """
        Implement the binary arithmetic operation *, the other parameter is the
        right operand. Return a new instance of this class.
        """

        if isinstance(other, self.__class__):
            aocs = self._aocs + other._aocs
            coeff = self._coeff * other._coeff
        elif isinstance(other, (int, float, complex)):
            aocs = self._aocs
            coeff = self._coeff * other
        else:
            raise TypeError("Multiply with the given parameter is not supported.")

        return ParticleTerm(aocs=aocs, coeff=coeff)
    # }}}
    
    def __rmul__(self, other):# {{{
        """
        Implement the binary arithmetic operation *, the other parameter is the
        left operand.

        This method return a new instance of this class. If you just want to 
        update the coeff attribute but not create a new instance, use the 
        setCoeff(coeff) method.
        """

        if isinstance(other, (int, float, complex)):
            aocs = self._aocs
            coeff = other * self._coeff
        else:
            raise TypeError("Multiply with the given parameter is not supported.")

        return ParticleTerm(aocs=aocs, coeff=coeff)
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

        Parameters
        ----------
        aoc_seq : list or tuple
            A collection of creation and/or annihilation operators.

        Returns
        -------
        seq : list
            The norm form of the operator.
        swap_num : int
            The number swap to get the normal form.
        
        Raises
        ------
        SwapFermionError : Exceptions raised when swap creation and annihilation
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

        Parameters
        ----------
        tag : string, optional
            Determine the judgment criterion.
            If tag is none, both particle and hole pairing is ok, if tag is 'p'
            then only particle is ok and if tag is 'h' only hole pairing is ok.
            default: None
        """

        if self._normalized:
            if len(self._aocs) != 2:
                return False
            else:
                otype0 = self._aocs[0].getOtype()
                otype1 = self._aocs[1].getOtype()
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
                "determine it is a pairing term or not using simple logic.")
    # }}}

    def isHopping(self, tag=None):# {{{
        """
        Determining whether the operator is a hopping term.
        
        This method is only valid for these instance that the "normalized"
        attribute is set to be True. For these instance that normalized
        attribute is False, this method will raise NotImplementedError.

        Parameters
        ----------
        tag : string, optional
            Determine the judgment criterion.
            If tag is none, arbitrary hopping term is ok, and if tag is 'n' only
            the number operator is ok.
            default: None
        """

        if self._normalized:
            if len(self._aocs) != 2:
                return False
            else:
                c0 = self._aocs[0].getOtype() == CREATION
                c1 = self._aocs[1].getOtype() == ANNIHILATION
                if tag in ('n', 'N'):
                    c2 = self._aocs[0].sameState(self._aocs[1])
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
                "determine it is a hopping term or not using simple logic.")
    # }}}
    
    def isHubbard(self):# {{{
        """
        Determining whether the operator is a hubbard term.
        
        This method is only valid for these instance that the "normalized"
        attribute is set to be True. For these instance that normalized
        attribute is False, this method will raise NotImplementedError.
        """

        if self._normalized:
            if len(self._aocs) == 4:
                if (self._aocs[0].getOtype() == CREATION and
                    self._aocs[1].getOtype() == CREATION and
                    self._aocs[2].getOtype() == ANNIHILATION and
                    self._aocs[3].getOtype() == ANNIHILATION and
                    self._aocs[0].sameState(self._aocs[3]) and
                    self._aocs[1].sameState(self._aocs[2])
                    ):
                    return True
                else:
                    return False
            else:
                return False
        else:
            raise NotImplementedError("The term is not normalied, We can't "
                "determine it is a hubbard term or not using simple logic.")
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

        return self._aocs == other._aocs
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermit conjugate of this operator.
        """

        aocs = [aoc.dagger() for aoc in self._aocs[::-1]]
        return ParticleTerm(aocs=aocs, coeff=self._coeff.conjugate())
    # }}}

    def conjugateOf(self, other):# {{{
        """
        Determine whether self is the Hermit conjugate of other.

        Warnning:
            The logic of this method maybe incorrect, use this method with
            caution.
        """
        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.sameAs(other.dagger())
    # }}}

    def setCoeff(self, coeff):# {{{
        """
        Set the coeff attribute of instance of this class.
        """
        
        if isinstance(coeff, (int, float, complex)):
            self._coeff = coeff
        else:
            raise TypeError("The input coeff parameter is of invalid type!")
    # }}}

    @staticmethod
    def matrixFunc(operators, rbase, *, lbase=None, coeff=1.0):# {{{
        """
        Return the matrix representation of the term in the Hilbert space 
        specified by the rbase and the optional lbase.

        Parameters
        ----------
        operators : list or tuple
            It is a sequence of length-2 tuples or lists. The first entry is the
            index of the state and the second is the operator type(CREATION or
            ANNIHILATION).
        rbase : tuple or list
            The bases of the Hilbert space before the operation.
        lbase : tuple or list, optional, keyword only
            The bases of the Hilbert space after the operation.
            It not given or None, lbase is the same as rbase.
            default: None
        coeff : int, float or complex, optional, keyword only
            The coefficience of the term.
            default: 1.0

        Returns
        -------
        res : csr_martrix
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

    def matrixRepr(self, statemap, rbase, *, lbase=None, coeff=None):# {{{
        """
        Return the matrix representation of the operator specified by this
        instance in the Hilbert space of the manybody system.
        
        Parameters
        ----------
        statemap : IndexMap
            A map system that associate instance of StateID with an integer 
            index.
        rbase : tuple or list
            The bases of the Hilbert space before the operation.
        lbase : tuple or list, optional, keyword only
            The bases of the Hilbert space after the operation.
            It not given or None, lbase is the same as rbase.
            default: None
        coeff : int, float or complex, optional, keyword only
            The coefficience of the term.
            default: None

        Returns
        -------
        res : csr_martrix
            The matrix representation of the term.
        """

        if coeff is not None:
            self.setCoeff(coeff)

        operators = []
        for aoc in self._aocs:
            operators.append((aoc.getStateIndex(statemap), aoc.getOtype()))
        return self.matrixFunc(operators, rbase, lbase=lbase, coeff=self._coeff)
    # }}}
# }}}
