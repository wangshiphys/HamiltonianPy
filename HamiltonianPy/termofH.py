from scipy.sparse import csr_matrix, identity, kron

import numpy as np

from HamiltonianPy.constant import CREATION, ANNIHILATION, SWAP_FACTOR_F
from HamiltonianPy.constant import SIGMA_MATRIX, SPIN_MATRIX, SPIN_OTYPE
from HamiltonianPy.exception import SwapError
from HamiltonianPy.indexmap import IndexMap
from HamiltonianPy.matrepr import aocmatrix, termmatrix

__all__ = [SiteID, StateID, AoC, SpinOptor, SpinInteraction, ParticleTerm]

#Useful constant in this module!
ALTER = 1000
FLOAT_TYPE = [np.float64, np.float32]
#####################################

def normalize(optors):# {{{
    """
    Reordering an operator specified by optors into norm form.

    For operator consist of creation and annihilation operator, the norm form 
    of an operator means that all the creation operators appear to the left of 
    all the annihilation operators. Also, the creation and annihilation 
    operators are sorted in ascending and descending order respectively 
    according to the single particle state associated with them.

    Paramter:
    ---------
    optors: list
        A collection of optors that consist an operator.
    
    RETURN:
    -------
    seq: list
        The norm form of the operator.
    swap_num: int
        The number of swap need to reordering the operator.
    """

    seq = list(optors[:])
    seq_len = len(seq)
    swap_num = 0

    C = CREATION
    A = ANNIHILATION
    for length in range(seq_len, 1, -1):
        for i in range(0, length-1):
            tmp0 = seq[i]
            tmp1 = seq[i+1]
            otype0 = tmp0.getOtype()
            otype1 = tmp1.getOtype()
            id0 = tmp0.getStateID()
            id1 = tmp1.getStateID()
            case0 = (otype0 == C) and (otype1 == C) and (id0 > id1)
            case1 = (otype0 == A) and (otype1 == A) and (id0 < id1)
            case2 = (otype0 == A) and (otype1 == C)
            case3 = (id0 == id1)

            if case0 or case1 or (case2 and (not case3)):
                seq[i] = tmp1
                seq[i+1] = tmp0
                swap_num += 1
            elif case2 and case3:
                raise SwapError(seq[i], seq[i+1])
    return seq, swap_num
# }}}


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
            self.site = site
        else:
            raise TypeError("The invalid site parameter.")
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describles the content of the instance.
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
            tmp = np.trunc(self.site * ALTER) / ALTER
        else:
            tmp = self.site
        return tuple(tmp)
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
        Return a string that describles the content of the instance.
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
        a 1D array, usually it has length 1, 2 or 3 cooresponding 
        to 1, 2 or 3 space dimension.
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
    General methods:
        getOtype()
        getStateID()
        tupleform()
        dagger()
        sameState(other)
        conjugateOf(other)
        update(otype=None, site=None, spin=None, orbit=None)
        matrixRepr(statemap, rbase, lbase=None)
    Methods Inherit from SiteID:
        __hash__()
        __eq__(other)
        __gt__(other)
        __lt__(other)
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

        #1 ^ 1 = 0 and 0 ^ 1 = 1, xor with 1 flip the bit.
        otype = self.otype ^ CREATION
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

    def update(self, otype=None, site=None, spin=None, orbit=None):# {{{
        """
        Create a new aoc with the same parameter as self except for those 
        given to update method.

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

    def matrixRepr(self, statemap, rbase, lbase=None):# {{{
        """
        Return the matrix representation of the operator specified by this
        instance in the Hilbert space of the manybody system.
        
        Parameter:
        ----------
        statemap: IndexMap
            A map system that associate instance of StateID with an integer 
            index.
        rbase: tuple or list
            The base of the Hilbert space before the operation.
        lbase: tuple or list, optional
            The base of the Hilbert space after the operation.
            If not given or None, lbase is the same as rbase.
            default: None

        Return:
        -------
        res: csr_matrix
            The matrix representation of the creation or annihilation operator.
        """

        index = self.getStateID().getIndex(statemap)
        res = aocmatrix(index, self.otype, rbase, lbase, statistics='F')
        return res
    # }}}
# }}}


class SpinOptor(SiteID):# {{{
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
        default: True

    Method:
    -------
    Special methods:
        __init__(otype, site, Pauli=True)
        __str__()
    General methods:
        tupleform()
        matrix()
        dagger()
        getSiteID()
        getOtype()
        sameSite(other)
        conjugateOf(other)
        update(otype=None, site=None)
        matrixRepr(sitemap)
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

    def __init__(self, otype, site, Pauli=True):# {{{
        SiteID.__init__(self, site=site)
        self.Pauli = Pauli
        if otype in SPIN_OTYPE:
            self.otype = otype
        else:
            raise TypeError("The invalid otype!")
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describles the content of the instance.
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
            matrix = SIGMA_MATRIX
        else:
            matrix = SPIN_MATRIX

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
        Determine whether the self SpinOptor and other SpinOptor is on 
        the same site.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.getSiteID() == other.getSiteID()
    # }}}
    
    def update(self, otype=None, site=None):# {{{
        """
        Create a new SpinOptor with the same parameter as self except for those 
        given to update method.

        Return:
        -------
        res: A new instance of SpinOptor.
        """

        if otype is None:
            otype = self.otype
        if site is None:
            site = self.site

        return SpinOptor(otype=otype, site=site, Pauli=self.Pauli)
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
            A map system that associate instance of SiteID with an integer 
            index.
        Return:
        -------
        res: csr_matrix
            The matrix representation of the creation or annihilation operator.
        """

        tot = len(sitemap)
        index = self.getSiteID().getIndex(sitemap)
        I0 = identity(2**index, dtype=np.int64, format="csr")
        I1 = identity(2**(tot-index-1), dtype=np.int64, format="csr")
        S = csr_matrix(self.matrix())
        res = kron(I1, kron(S, I0, format="csr"), format="csr")
        res.eliminate_zeros()
        return res
    # }}}
# }}}


class SpinInteraction:# {{{
    """
    This class provide a unified description of spin interaction term.

    Attribute:
    ----------
    components: tuple of instances of SpinOptor class.
    coeff: int, float, complex, optional
        The coefficience of this term.
        default: 1.0

    Method:
    -------
    Special methods:
        __init__(components, coeff=1.0)
        __str__()
    General methods:
        dagger()
        sameAs(other)
        conjugateOf(other)
        updateCoeff(coeff)
        matrixRepr(sitemap, coeff)
    """

    def __init__(self, components, coeff=1.0):# {{{
        """
        Initialize instance of this class.

        See also the documentation of this class.
        """
        
        #Sort the spin operators in ascending order according to their SiteID.
        #The relative position of two operators with the same SiteID will not
        #change and the exchange of two spin operators on different site never
        #change the interaction term.
        tmp = sorted(components, key=lambda item: item.getSiteID())
        self.components = tuple(tmp)
        self.coeff = coeff
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describles the content of the instance.
        """

        info = "coeff: {0}\n".format(self.coeff)
        for item in self.components:
            info += str(item)
        return info
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermit conjugate of this operator.
        """

        coeff = self.coeff.conjugate()
        tmp = []
        for item in self.components[::-1]:
            tmp.append(item.dagger())
        return SpinInteraction(components=tmp, coeff=coeff)
    # }}}

    def sameAs(self, other):# {{{
        """
        Judge whether two instance of this class equal to each other.

        Here, the equal defines as follow, if the two instance are consisted of
        the same spin opertors, also these operators are in the same order, 
        then we claim the two instance are equal. We never care about the coeff
        attribute of the two instance.
        """

        if isinstance(other, self.__class__):
            return self.components == other.components
        else:
            raise TypeError("The input parameter is not instance of this class!")
    # }}}

    def conjugateOf(self, other):# {{{
        """
        Determine whether the instance is conjugate of the other instance.

        Here, we also do not care about the coeff attribute.
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

        tot = len(sitemap)
        res = identity(2**tot, dtype=np.int64, format="csr")
        for optor in self.components:
            res = res.dot(optor.matrixRepr(sitemap))
        
        res.eliminate_zeros()
        return self.coeff * res
    # }}}
# }}}


class ParticleTerm:# {{{
    """
    This class provide unified description of any operator 
    composed of creation and annihilation operators.

    Attribute:
    ----------
    aocs: tuple
        The creation and annihilation operators that consist of this operator.
    coeff: float, int or complex, optional
        The coefficient of the operator.
        default: 1.0

    
    Method:
    -------
    Special methods:
        __init__(aocs, coeff=1.0)
        __str__()
    General methods:
        isPairing(tag=None)
        isHopping(tag=None)
        isHubbard()
        sameAs(other)
        dagger()
        conjugateOf(other)
        updateCoeff(coeff)
        matrixRepr(statemap, base, coeff=None)
    """

    def __init__(self, aocs, coeff=1.0):# {{{
        """
        Initilize the instance of this class.

        Paramter:
        --------
        aocs: tuple or list
            The creation and annihilation operators that consist this operator.
        coeff: float, int or complex, optional
            The coefficient of the operator.
            default: 1.0
        """

        normal_aocs, swap_num = normalize(aocs)

        self.aocs = tuple(normal_aocs)
        self.coeff = (SWAP_FACTOR_F ** swap_num) * coeff
    # }}}

    def __str__(self):# {{{
        """
        Return the printing string of instance of this class!
        """

        info = "coeff: {0}".format(self.coeff)
        for aoc in self.aocs:
            info += str(aoc)
        return info
    # }}}

    def isPairing(self, tag=None):# {{{
        """
        Determining whether the operator is a pairing term.

        Parameter:
        ----------
        tag: string, optional
            Determine the judgment criteria.
            If tag is none, both particle and hole pairing is ok, if tag is 'p'
            then only particle is ok and if tag is 'h' only hole pairing is ok.
            default: None
        """

        if len(self.aocs) != 2:
            return False
        else:
            otype0 = self.aocs[0].getOtype()
            otype1 = self.aocs[1].getOtype()
            if tag is None:
                if otype0 == otype1:
                    return True
                else:
                    return False
            elif tag in ('p', 'P'):
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
                raise ValueError("The invalid tag parameter.")
    # }}}

    def isHopping(self, tag=None):# {{{
        """
        Determining whether the operator is a hopping term.

        Parameter:
        ----------
        tag: string, optional
            Determine the judgment criteria.
            If tag is none, arbitrary hopping term is ok, and if tag is 'n' only
            the number operator is ok.
            default: None
        """

        if len(self.aocs) != 2:
            return False
        else:
            c0 = self.aocs[0].getOtype() == CREATION
            c1 = self.aocs[1].getOtype() == ANNIHILATION
            if tag is None:
                if c0 and c1:
                    return True
                else:
                    return False
            elif tag in ('n', 'N'):
                c2 = self.aocs[0].sameState(self.aocs[1]
                if c0 and c1 and c2:
                    return True
                else:
                    return False
            else:
                raise ValueError("The invalid tag parameter.")
    # }}}
    
    def isHubbard(self):# {{{
        """
        Determining whether the operator is a hubbard term.
        """

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
    # }}}

    def sameAs(self, other):# {{{
        """
        Judge whether two instance of this class equal to each other.

        Here, the equal defines as follow, if the two instance are consisted of
        the same creation and/or annihilation opertors, also these operators are
        in the same order, then we claim the two instance are equal. We never
        care about the coeff attribute of the two instance.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.aocs == other.aocs
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermit conjugate of this operator.
        """

        tmp = []
        for aoc in self.aocs[::-1]:
            tmp.append(aoc.dagger())
        res = Optor(tmp, self.coeff.conjugate())
        return res
    # }}}

    def conjugateOf(self, other):# {{{
        """
        Determine whether a operator is the Hermit conjugate of it's self.
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

    def matrixRepr(self, statemap, base, coeff=None):# {{{
        """
        Return the matrix representation of the operator specified by this
        instance in the Hilbert space of the manybody system.
        
        Parameter:
        ----------
        statemap: IndexMap
            A map system that associate instance of StateID with an integer 
            index.
        base: tuple or list
            The base of the Hilbert space.
        coeff: int, float or complex, optional
            The coefficient of the operator.
            default: None

        Return:
        -------
        res: csr_matrix
            The matrix representation of the creation or annihilation operator.
        """

        if coeff is not None:
            self.updateCoeff(coeff)

        if self.isHopping():
            termtype = "hopping"
            cindex = self.aocs[0].getStateID().getIndex(statemap)
            aindex = self.aocs[1].getStateID().getIndex(statemap)
            term = ((cindex, aindex), self.coeff)
        elif self.isPairing():
            termtype = "pairing"
            index0 = self.aocs[0].getStateID().getIndex(statemap)
            index1 = self.aocs[1].getStateID().getIndex(statemap)
            otype = self.aocs[0].getOtype()
            term = ((cindex, aindex), self.coeff, otype)
        elif self.isHubbard():
            termtype = "hubbard"
            index0 = self.aocs[0].getStateID().getIndex(statemap)
            index1 = self.aocs[1].getStateID().getIndex(statemap)
            term = ((index0, index1), self.coeff)
        else:
            raise ValueError("Not supported term.")

        res = termmatrix(term=term, base=base, termtype=termtype, statistics='F')
        return res
    # }}}
# }}}
