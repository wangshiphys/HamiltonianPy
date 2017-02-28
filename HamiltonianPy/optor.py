from copy import deepcopy

import numpy as np

from HamiltonianPy.arrayformat import arrayformat
from HamiltonianPy.constant import CREATION, ANNIHILATION, SWAP_FACTOR_F
from HamiltonianPy.exception import SwapError
from HamiltonianPy.indexmap import IndexMap

__all__ = ['normalform', 'State', 'AoC', 'Optor', 'OptorWithIndex']

#Useful constant in this module!
ALTER = 1000
INDENT = 6
FLOATTYPE = [np.float_, np.float16, np.float32, np.float64]
###############################################################################

def normalform(aocs):# {{{
    """
    Reordering an operator specified by aocs into norm form.

    The norm form of an operator means that all the creation operators appear 
    to the left of all the annihilation operators. Also, the creation and 
    annihilation operators are sorted in ascending and descending order
    respectively according to the single particle state associated with them.

    Paramter:
    ---------
    aocs: list
        A collection of AoCs that consist an operator.
    
    RETURN:
    -------
    seq: list
        The norm form of the operator.
    swap_num: int
        The number of swap need to reordering the operator.
    """

    seq = list(aocs[:])
    seq_len = len(seq)
    swap_num = 0

    for length in range(seq_len, 1, -1):
        for i in range(0, length-1):
            otype0 = seq[i].otype
            otype1 = seq[i+1].otype

            case0 = ((otype0 == CREATION) and 
                     (otype1 == CREATION) and 
                     (seq[i] > seq[i+1]))

            case1 = ((otype0 == ANNIHILATION) and
                     (otype1 == ANNIHILATION) and
                     (seq[i] < seq[i+1]))

            case2 = (otype0 == ANNIHILATION) and (otype1 == CREATION)

            case3 = seq[i].samestate(seq[i+1])

            if case0 or case1 or (case2 and (not case3)):
                buff = seq[i]
                seq[i] = seq[i+1]
                seq[i+1] = buff
                swap_num += 1
            elif case2 and case3:
                info = str(seq[i]) + "\n" + str(seq[i + 1]) + "\n"
                info += "Swap these two operator would generate "
                info += "extra identity operator, which can not " 
                info += "be processed by this function properly!"
                raise SwapError(info)

    return seq, swap_num
# }}}


class State:# {{{
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
    tupleform: tuple
        Combine the above three attributes to a tuple, (orbit, spin, site)

    Method:
    -------
    __init__(site, spin=0, orbit=0)
    __str__()
    __hash__()
    __eq__(other)
    __lt__(other)
    __gt__(other)
    __ne__(other)
    __le__(other)
    __ge__(other)
    """

    def __init__(self, site, spin=0, orbit=0):# {{{
        """
        Initialize this class!

        Parameter:
        ----------
        See the documentation of this class.
        """

        if isinstance(site, np.ndarray) and site.shape in [(1,), (2,), (3,)]:
            self.site = site
        else:
            raise ValueError("The invalid site parameter!")

        if isinstance(spin, int) and spin >= 0:
            self.spin = spin
        else:
            raise ValueError("The invalid spin parameter!")

        if isinstance(orbit, int) and orbit >= 0:
            self.orbit = orbit
        else:
            raise ValueError("The invalid orbit parameter!")
        
        if site.dtype in FLOATTYPE:
            buff = np.trunc(site * ALTER) / ALTER
        else:
            buff = site

        self.tupleform = (orbit, spin) + tuple(buff)
    # }}}

    def __str__(self):# {{{
        """
        Return the printing string of instance of this class! 
        """

        info = "\n" + " " * INDENT + "site:  " + arrayformat(self.site)
        info += "\n" + " " * INDENT + "spin:  {0}".format(self.spin)
        info += "\n" + " " * INDENT + "orbit: {0}\n".format(self.orbit)
        return info
    # }}}

    def __hash__(self):# {{{
        """
        Return the hash value of instance of the class!
        """
        
        return hash(self.tupleform)
    # }}}

    def __lt__(self, other):# {{{
        """
        Define the '<' operator of instance of this class.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.tupleform < other.tupleform
    # }}}

    def __eq__(self, other):# {{{
        """
        Define the '==' operator of instance of this class.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.tupleform == other.tupleform
    # }}}

    def __gt__(self, other):# {{{
        """
        Define the '>' operator of instance of this class.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.tupleform > other.tupleform
    # }}}

    def __le__(self, other):# {{{
        """
        Define the '<=' operator of instance of this class.
        """

        return (self.__lt__(other)) or (self.__eq__(other))
    # }}}

    def __ne__(self, other):# {{{
        return not self.__eq__(other)
    # }}}

    def __ge__(self, other):# {{{
        """
        Define the '>=' operator of instance of this class.
        """

        return (self.__gt__(other)) or (self.__eq__(other))
    # }}}
# }}}


class AoC(State):# {{{
    """
    This class provide a unified description of 
    the creation and annihilation operator.

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
    tupleform: tuple
        Combine the above four attributes to a tuple, (otype, orbit, spin, site)

    Method:
    -------
    __init__(otype, site, spin=0, orbit=0)
    __str__()
    dagger()
    isConjugateOf(other)
    extract()
    samestate(other)
    update(otype, site, spin, orbit)

    Inherit from Base class:
    __hash__()
    __eq__(other)
    __gt__(other)
    __lt__(other)
    __ne__(other)
    __ge__(other)
    __le__(other)
    """

    def __init__(self, otype, site, spin=0, orbit=0):# {{{
        """
        Initilize instance of this class.

        Paramter:
        ---------
        See the documentation of this class.
        """

        State.__init__(self, site=site, spin=spin, orbit=orbit)

        if otype in (ANNIHILATION, CREATION):
            self.otype = otype
        else:
            raise ValueError("The invalid otype!")
        
        #The call to State.__init__() function would set tuplefrom attribute to
        #instance of this class, but that's only partial of the final form.

        self.tupleform = (otype, ) + self.tupleform
    # }}}

    def __str__(self):# {{{
        """
        Return the print string of instance of this class!
        """

        info = '\n' + ' ' * INDENT + 'otype: {0}'.format(self.otype)
        info += State.__str__(self)
        return info
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermite conjugate of self.

        Return:
        -------
        res: A new instance of this class.
        """

        #1 ^ 1 = 0 and 0 ^ 1 = 1, xor with 1 flip the bit.
        otype = self.otype ^ CREATION
        res = AoC(otype=otype, site=self.site, spin=self.spin, orbit=self.orbit)
        return res
    # }}}

    def isConjugateOf(self, other):# {{{
        """
        Determine whether the instance is conjugate of other instance.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")
        
        return self.__eq__(other.dagger())
    # }}}

    def extract(self):# {{{
        """
        Extract the state information of this creation or annihilation operator.

        Return:
        -------
        res: A new instance of State class.
        """

        return State(site=self.site, spin=self.spin, orbit=self.orbit)
    # }}}

    def samestate(self, other):# {{{
        """
        Determine whether the self AoC and other AoC is of the same state.
        """

        if not isinstance(other, self.__class__):
            raise TypeError("The right operand is not instance of this class!")

        return self.extract() == other.extract()
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

        aoc = AoC(otype=otype, site=site, spin=spin, orbit=orbit)
        return aoc
    # }}}
# }}}


class Optor:# {{{
    """
    This class provide unified description of any operator 
    composed of creation and annihilation operators.

    Attribute:
    ----------
    aocs: list
        The creation and annihilation operators that consist of this operator.
    coeff: float, int or complex, optional
        The coefficient of the operator.
        default: 1.0
    num: int
        The number of creation and/or annihilation operators that compose this
        operator.
    otypes: tuple
        The type of this aocs.

    
    Method:
    -------
    __init__(aocs, coeff=1.0)
    __str__()
    ispairing()
    ishopping()
    ishubbard()
    sameoptor(other)
    dagger()
    isSelfConjugate()
    updatecoeff(coeff)
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

        normal_aocs, swap_num = normalform(aocs)

        otypes = []
        for aoc in normal_aocs:
            otypes.append(aoc.otype)

        self.aocs = tuple(normal_aocs)
        self.coeff = (SWAP_FACTOR_F ** swap_num) * coeff
        self.otypes = tuple(otypes)
        self.num = len(normal_aocs)
        # }}}

    def __str__(self):# {{{
        """
        Return the printing string of instance of this class!
        """

        info = "\n" + " " * INDENT + "coeff: {0}".format(self.coeff)
        for aoc in self.aocs:
            info += str(aoc)
        return info
    # }}}

    def ispairing(self):# {{{
        """
        Determining whether the operator is a pairing term.
        """

        if self.num == 2 and self.otypes[0] == self.otypes[1]:
            return True
        else: 
            return False
    # }}}

    def ishopping(self):# {{{
        """
        Determining whether the operator is a hopping term.
        """

        if self.num == 2 and self.otypes[0] != self.otypes[1]:
            return True
        else: 
            return False
    # }}}

    def ishubbard(self):# {{{
        """
        Determining whether the operator is a hubbard term.
        """

        rule0 = self.otypes==(CREATION, CREATION, ANNIHILATION, ANNIHILATION)
        rule1 = self.aocs[0].samestate(self.aocs[3])
        rule2 = self.aocs[1].samestate(self.aocs[2])
        if (rule0 and rule1 and rule2):
            return True
        else:
            return False
    # }}}

    def sameoptor(self, other):# {{{
        """
        Judge whether two instance of this class equal to each other.

        Here, the equal defines as follow, if the two instance are consisted of
        the same creation and/or annihilation opertors, also these operators are
        in the same order, then we claim the two instance are equal. We never
        care about the coeff attribute of the two instance.
        """

        if not isinstance(other, Optor):
            raise TypeError("The right operand is not instance of this class!")

        return hash(self.aocs) == hash(other.aocs)
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermit conjugate of this operator.
        """

        buff = []
        for aoc in self.aocs[::-1]:
            buff.append(aoc.dagger())
        res = Optor(buff, self.coeff.conjugate())
        return res
    # }}}

    def isSelfConjugate(self):# {{{
        """
        Determine whether a operator is the Hermit conjugate of it's self.
        """

        return self.sameoptor(self.dagger())
    # }}}

    def updatecoeff(self, coeff):# {{{
        """
        Update the coeff attribute of instance of this class.
        """

        self.coeff = coeff
    # }}}
# }}}

class OptorWithIndex(Optor):# {{{
    """
    A subclass of class Optor with the single particle state associated with
    index according to the statemap parameter.
    """

    def __init__(self, aocs, statemap, coeff=1.0):
        Optor.__init__(self, aocs=aocs, coeff=coeff)
        
        indices = []
        for aoc in self.aocs:
            state = aoc.extract()
            index = statemap(state)
            indices.append(index)
        self.indices = tuple(indices)

    def __call__(self):
        return (self.indices, self.coeff)
# }}}


if __name__ == "__main__":
    sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    spins = (0, 1)
    orbits = (0, 1)
    otypes = (0, 1)
    aocs = []
    for otype in otypes:
        for orbit in orbits:
            for spin in spins:
                for site in sites:
                    aoc = AoC(otype=otype, site=site, spin=spin, orbit=orbit)
                    print(aoc.tupleform)
                    print("="*30)
                    aocs.append(aoc)
