from copy import deepcopy

import numpy as np

from HamiltonianPy.arrayformat import arrayformat
from HamiltonianPy.constant import CREATION, ANNIHILATION

__all__ = ['State', 'AoC', 'Optor']

#Useful constant in this module!
ALTER = 1000
INDENT = 6
###############################

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

    Method:
    -------
    __init__(site, spin=0, orbit=0)
    __str__()
    _2tuple()
    __hash__()
    __eq__(other)
    __ne__(other)
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

    def _2tuple(self):# {{{
        """
        Compose the site, spin and orbit index to a hashable tuple!
        """

        site = np.trunc(self.site * ALTER) / ALTER
        res = tuple(site) + (self.spin, self.orbit)
        return res
    # }}}

    def __hash__(self):# {{{
        """
        Return the hash value of instance of the class!
        """
        
        return hash(self._2tuple())
    # }}}

    def __eq__(self, other):# {{{
        """
        Define the '==' operator of instance of this class.
        """

        if not isinstance(other, State):
            raise TypeError("The right operand is not instance of this class!")

        return hash(self) == hash(other)
    # }}}

    def __ne__(self, other):# {{{
        return not self.__eq__(other)
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

    Method:
    -------
    __init__(otype, site, spin=0, orbit=0)
    __str__()
    _2tuple()
    isConjugateOf(other)
    dagger()

    Inherit from Base class:
    __hash__()
    __eq__(other)
    __ne__(other)
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
    # }}}

    def __str__(self):# {{{
        """
        Return the print string of instance of this class!
        """

        info = '\n' + ' ' * INDENT + 'otype: {0}'.format(self.otype)
        info += State.__str__(self)
        return info
    # }}}

    def _2tuple(self):# {{{
        """
        Compose the site, spin and orbit index to a hashable tuple!
        """

        res = (self.otype, ) + State._2tuple(self)
        return res
    # }}}

    def isConjugateOf(self, other):# {{{
        """
        Determine whether the instance is conjugate of other instance.
        """

        if not isinstance(other, AoC):
            raise TypeError("The right operand is not instance of this class!")
        
        if (State._2tuple(self) == State._2tuple(other) and
            self.otype != other.otype):
            return True
        else:
            return False
    # }}}

    def dagger(self):# {{{
        """
        Return the Hermite conjugate of self.
        """

        res = deepcopy(self)
        res.otype = self.otype ^ CREATION
        return res
    # }}}

    def extract(self):# {{{
        """
        Extract the state information of this creation or annihilation operator.
        """

        return State(site=self.site, spin=self.spin, orbit=self.orbit)
    # }}}
# }}}


class Optor:# {{{
    """
    This class provide unified description of any operator 
    composed of creation and annihilation operators.

    Attribute:
    ----------
    aocs: tuple
        The creation and annihilation operators that cconsist of this operator.
    coeff: float, int or complex, optional
        The coefficient of the operator.
        default: 1.0
    num: int
        The number of creation and/or annihilation operators that compose this
        operator.
    otypes: tuple
        The type of this aocs.
    indices: tuple
        The state indices of this aocs

    
    Method:
    -------
    __init__(aocs, stateMap, coeff=1.0, check=True)
    __call__()
    __str__()
    ispairing()
    ishooping()
    ishubbard()
    """

    def __init__(self, aocs, coeff=1.0):# {{{
        """
        Initilize the instance of this class.

        Paramter:
        --------
        aocs: tuple
            The creation and annihilation operators that cconsist of this operator.
        coeff: float, int or complex, optional
            The coefficient of the operator.
            default: 1.0
        """

        otypes = []
        for aoc in aocs:
            if isinstance(aoc, AoC):
                otypes.append(aoc.otype)
            else:
                raise TypeError("The input aocs are not all instance of AoC!")

        self.aocs = tuple(aocs)
        self.coeff = coeff
        self.otypes = tuple(otypes)
        self.num = len(aocs)
        # }}}

    def setIndices(self, stateMap):# {{{
        """
        Map the single particle state of these creation and/or annihilation
        operator to integer number.

        Parameter:
        ----------
        stateMap: class IndexMap
            The Map between the single particle state
            and it's corresponding index.
        """

        indices = []
        for aoc in self.aocs:
            state = aoc.extract()
            index = stateMap(state)
            indices.append(index)
        self.indices = tuple(indices)
    # }}}

    def __call__(self):# {{{
        """
        Return the single partice indices and the coefficient of this operator.
        """

        if not hasattr(self, 'indices'):
            raise ValueError("The optor has not specified with indices!")
        return (self.indices, self.coeff)
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

        judge = self.otypes == (1, 0, 1, 0)
        if (judge and 
            self.aocs[0].isConjugateOf(self.aocs[1]) and
            self.aocs[2].isConjugateOf(self.aocs[3])):
            return True
        else:
            return False
    # }}}

    def __eq__(self, other):# {{{
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

    def __ne__(self, other):# {{{
        """
        Judge whether two instance of this class not equal.
        """

        return not self.__eq__(other)
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

        if self.ishubbard():
            return True
        else:
            return self.dagger() == self
    # }}}
# }}}
