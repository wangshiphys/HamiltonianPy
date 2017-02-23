from collections import OrderedDict
from numpy.linalg import solve
from scipy.sparse.linalg import eigsh
from time import time

import numpy as np

from HamiltonianPy.constant import CREATION, ANNIHILATION
from HamiltonianPy.lanczos import Lanczos
from HamiltonianPy.matrepr.base import base_table
from HamiltonianPy.matrepr.matrepr import aocmatrix, termmatrix

__all__ = ['GFED']

class GFED_ABC:# {{{
    """
    This class provide the method to calculate the cluster Green Function.

    Attribute:
    ----------
    Hterms: list
        A collection of all Hamiltonian terms
    lAoCs: list
        All concerned creation and/or annihilation operators. This is the
        collection all operators in left side of the matrix representation of
        perturbation V.
        For example, if we write V in this form
        (C0, C1, C2, C3,...)(V_matrix)(A0, A1, A2, A3, ...)^T, than lAoCs is
        all these C operators in the left side and rAoCs is all 
        these A operators in the right side.
    rAoCs: list
        All concerned creation and/or annihilation operators. This is the
        collections all operators in right side of the matrix representation of
        Green function. All operators in this collection is the Hermit conjugate
        of the corresponding operator in lAoCs.
    StateMap: IndexMap
        A map from all the single particle state to integer number.
    lAoCMap: IndexMap
        A map from all operators in lAoCs to integer number.
    rAoCMap: IndexMap
        A map from all operators in rAoCs to integer number.
    dof: int
        Total degree of freedom of the system.
    statistics: string, optional
        The statistics rule the system obey. This attribute can be only "F" or 
        "B", which represents Fermi and Bose statistics respectively.

    Special Method:
    --------------
    __init__(Hterms, lAoCs, rAoCs, StateMap, lAoCMap, rAoCMap, dof)
    setHamiltonian()
    setgs()
    setprojection()
    setall()
    __call__(omega)
    
    Method:
    -------
    _excitation(aocs, lbase, rbase)
    _gf_term(A, B, omega)
    gf(omega)

    Static Method:
    --------------
    _inverse(omega, H, E0, tag='normal')
    """

    def __init__(self, Hterms, lAoCs, rAoCs, StateMap, 
                 lAoCMap, rAoCMap, dof, statistics="F"):
    # {{{
        """
        Initialize instance of this class.
        
        See also the document of this class.
        """

        self.Hterms = Hterms
        self.lAoCs = lAoCs
        self.rAoCs = rAoCs
        self.StateMap = StateMap
        self.lAoCMap = lAoCMap
        self.rAoCMap = rAoCMap
        self.dof = dof
        self.statistics = statistics
    # }}}

    def setHamiltonian(self):# {{{
        """
        Calculate the Hamiltonian matrix of the cluster.

        Must be overloaded by subclass method.
        """

        raise NotImplementedError
    # }}}

    def setgs(self):# {{{
        """
        Calculate the ground state and ground state energy of the Hamiltonian!
        """

        t0 = time()
        val, vec = eigsh(self.H, k=1, which='SA')
        self.GE = val[0]
        self.GS = vec.copy()
        t1 = time()
        info = "The time spend on ground state {0:.2f}.".format(t1 - t0)
        print(info)
        print('=' * len(info))
    # }}}

    def setprojection(self):# {{{
        """
        Project the Hamiltonian matrix and ground state to the Krylov space.
        """

        raise NotImplementedError
    # }}}

    def setall(self):# {{{
        """
        Set all attribute needed!
        """

        self.setHamiltonian()
        self.setgs()
        self.setprojection()
    # }}}

    def _excitation(self, aocs, lbase, rbase):# {{{
        """
        Calculate the excitation state from the ground state.
        """

        t0 = time()
        res = OrderedDict()
        for aoc in aocs:
            otype = aoc.otype
            index = self.StateMap(aoc.extract())
            M = aocmatrix(index=index, otype=otype, lbase=lbase, 
                          rbase=rbase, statistics=self.statistics)
            res[aoc] = M.dot(self.GS)
        t1 = time()
        info = "The time spend of excitation {0:.2f}.".format(t1 - t0)
        print(info)
        print("=" * len(info))
        return res
    # }}}

    @staticmethod
    def _inverse(omega, H, E0, tag='normal'):# {{{
        """
        Return the represention of vector (omega ∓ (H-E))^-1|ket>.

        The |ket> is a ndarray with shape (n, 1) where n is the dimension of the
        input H matrix. The first element of |ket> is 1.0 and other is all 0.0.

        Parameter:
        ----------
        omega: complex
            The frequence of the green function.
            Omega should consist of a real part with a minor positive 
            imaginary part.
        H: ndarray
            The projected Hamiltionian matrix.
        E0: float
            The ground state energy of the cluster.
        tag: str, optional
            Specify which term of the retarded green function to be calculated.
            'normal', the first term, i.e. <A(t)B>
            'reverse', the second term, i.e. <BA(t)>
            default: 'normal'

        Return:
        ------
        res: ndarray
            The representation of the resulting vector.
        """
        
        if tag == 'normal':
            coeff = -1.0
        elif tag == 'reverse':
            coeff = 1.0
        else:
            raise ValueError("The invalid tag!")

        dim = H.shape[0]
        I = np.identity(dim, dtype=np.float64)
        matrix = omega * I + coeff * (H - E0 * I)
        vec = np.zeros((dim, 1), dtype=np.complex128)
        vec[0] = 1.0
        res = solve(matrix, vec)
        return res
    # }}}
    
    def _gf_term(self, A, B, omega):# {{{
        """
        Calculate a specific green function G_AB = <{A(omega), B}>.
        """

        term = [(A, B, 'normal'), (B, A, 'reverse')]
        res = 0.0
        for a, b, tag in term:
            coeff = self.kets_krylov[b][b][0]
            bra = self.kets_krylov[b][a.dagger()]
            H = self.Hs_krylov[b]
            ket = coeff * self._inverse(omega=omega, H=H, E0=self.GE, tag=tag)
            res += np.vdot(bra, ket)
        return res
    # }}}

    def gf(self, omega):# {{{
        """
        Calculate the Green function matrix of the cluster.
        """

        ldim = len(self.lAoCs)
        rdim = len(self.rAoCs)
        shape = (ldim, rdim)
        res = np.zeros(shape, dtype=np.complex128)
        for laoc in self.lAoCs:
            row = self.lAoCMap(laoc)
            for raoc in self.rAoCs:
                col = self.rAoCMap(raoc)
                res[row, col] = self._gf_term(A=raoc, B=laoc, omega=omega)
        return res
    # }}}

    def __call__(self, omega):# {{{
        """
        Make instance of this clas callable.
        """

        return self.gf(omega)
    # }}}
# }}}


class GFED_Numbu(GFED_ABC):# {{{
    """
    See also document of GFED_ABC class.
    """

    def setHamiltonian(self):# {{{
        """
        See also document of GFED_ABC class.
        """

        t0 = time()
        H = 0.0
        base = base_table(self.dof)
        for term in self.Hterms:
            H += termmatrix(term, base)
        H += H.conjugate().transpose()
        self.H = H
        t1 = time()
        info = "The time spend on generating H matrix {0:.2f}.".format(t1 - t0)
        print(info)
        print('=' * len(info))
    # }}}

    def setprojection(self):# {{{
        base = base_table(self.dof)
        excitation = self._excitation(aocs=self.rAoCs, lbase=base, rbase=base)
        lanczosH = Lanczos(self.H)

        t0 = time()
        self.Hs_krylov, self.kets_krylov = lanczosH(excitation)
        t1 = time()
        info = "The time spend on projection {0:.2f}.".format(t1 - t0)
        print(info)
        print("=" * len(info))
    # }}}
# }}}


class GFED_NotNumbu(GFED_ABC):# {{{
    """
    See also the document of the GFED_ABC class.
    """

    def __init__(self, Hterms, lAoCs, rAoCs, StateMap, lAoCMap, rAoCMap, 
                 dof, occupy=None):# {{{
        if not(isinstance(occupy, int) or (occupy is None)):
            raise TypeError("The invalid occupy parameter!")

        self.occupy = occupy
        GFED_ABC.__init__(self, Hterms, lAoCs, rAoCs,
                          StateMap, lAoCMap, rAoCMap, dof)
    # }}}

    def _base(self):# {{{
        """
        Generate the base of the Hilbert space.
        """

        if self.occupy is None:
            base = base_table(self.dof)
            basep = baseh = base
        else:
            base = base_table(self.dof, self.occupy)
            basep = base_table(self.dof, self.occupy + 1)
            baseh = base_table(self.dof, self.occupy - 1)
        return basep, base, baseh
    # }}}

    def setHamiltonian(self):# {{{
        H = 0.0
        Hp = 0.0
        Hh = 0.0
        basep, base, baseh = self._base()
        
        if self.occupy is None:
            for term in self.Hterms:
                H += termmatrix(term, base)
            H += H.conjugate().transpose()
            Hp = Hh = H
        else:
            for term in self.Hterms:
                H += termmatrix(term, base)
                Hp += termmatrix(term, basep)
                Hh += termmatrix(term, baseh)
            H += H.conjugate().transpose()
            Hp += Hp.conjugate().transpose()
            Hh += Hh.conjugate().transpose()

        self.H = H
        self.Hp = Hp
        self.Hh = Hh
    # }}}

    def setprojection(self):# {{{
        As = self.rAoCs
        Cs = self.lAoCs
        basep, base, baseh = self._base()
        h_excitation = self._excitation(aocs=As, lbase=baseh, rbase=base)
        p_excitation = self._excitation(aocs=Cs, lbase=basep, rbase=base)
        lanczosHp = Lanczos(self.Hp)
        lanczosHh = Lanczos(self.Hh)

        t0 = time()
        Hps_krylov, ketps_krylov = lanczosHp(p_excitation)
        Hhs_krylov, keths_krylov = lanczosHh(h_excitation)
        Hps_krylov.update(Hhs_krylov)
        ketps_krylov.update(keths_krylov)
        self.Hs_krylov = Hps_krylov
        self.kets_krylov = ketps_krylov
        t1 = time()
        info = "The time spend on projection {0:.2f}.".format(t1 - t0)
        print(info)
        print("=" * len(info))
    # }}}
# }}}


def GFED(Hterms, lAoCs, rAoCs, StateMap, lAoCMap, rAoCMap, 
         dof, occupy=None,numbu=False):# {{{
    """
    Return corresponding Green function instance according the input paramter.

    Paramter:
    ---------
    Hterms: list
        A collection of all Hamiltonian terms
    lAoCs: list
        All concerned creation and/or annihilation operators. This is the
        collection all operators in left side of the matrix representation of
        perturbation V.
        For example, if we write V in this form
        (C0, C1, C2, C3,...)(V_matrix)(A0, A1, A2, A3, ...)^T, than lAoCs is
        all these C operators in the left side and rAoCs is all 
        these A operators in the right side.
    rAoCs: list
        All concerned creation and/or annihilation operators. This is the
        collections all operators in right side of the matrix representation of
        Green function. All operators in this collection is the Hermit conjugate
        of the corresponding operator in lAoCs.
    StateMap: IndexMap
        A map from all the single particle state to integer number.
    lAoCMap: IndexMap
        A map from all operators in lAoCs to integer number.
    rAoCMap: IndexMap
        A map from all operators in rAoCs to integer number.
    dof: int
        Total degree of freedom of the system.
    occupy: int, optional
        The number of particle of the system.
        default: None
    numbu: bool, optional
        Whether to use numbu representation.
        default: False

    Return:
    -------
    res: GFED_Numbut or GFED_NotNumbu
    """

    if numbu:
        gf = GFED_Numbu(Hterms, lAoCs, rAoCs, StateMap, lAoCMap, rAoCMap, dof)
    else:
        gf = GFED_NotNumbu(Hterms, lAoCs, rAoCs, StateMap, lAoCMap, rAoCMap,
                           dof, occupy)

    gf.setall()
    return gf
# }}}
