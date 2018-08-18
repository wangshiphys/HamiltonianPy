from collections import OrderedDict
from numpy.linalg import solve
from scipy.sparse.linalg import eigsh
from time import time

import numpy as np

from HamiltonianPy.constant import CREATION, ANNIHILATION
from HamiltonianPy.lanczos import Lanczos
from HamiltonianPy.base import base_table


class GFEDABC_P:# {{{
    """
    This class provide the method to calculate the cluster Green Function with
    particle form Hamiltonian.

    Attribute:
    ----------
    HTerms: list
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
    objMaps: dict
        The keys of this dict should be string such as "sitemap", "statemap",
        "lAoCMap", "rAoCMap", etc,. The value corresponding to these keys are
        instance of IndexMap class.
    dof: int
        Total degree of freedom of the system.
    statistics: string, optional
        The statistics rule the system obey. This attribute can be only "F" or 
        "B", which represents Fermi and Bose statistics respectively.

    Method:
    -------
    Special Method:
        __init__(HTerms, lAoCs, rAoCs, objMaps, dof, statistics='F')
        setHamiltonian()
        setGS()
        setProjection()
        setAll()
        __call__(omega)
    General Method:
        excitation(aocs, rbase, lbase)
        gf(omega)
    Pseudo-private Method:
        _GFTerm(A, B, omega)
    Static Method:
        _inverse(omega, H, E0, tag='normal')
    """

    def __init__(self, HTerms, lAoCs, rAoCs, objMaps, dof, statistics="F"):# {{{
        """
        Initialize instance of this class.
        
        See also the document of this class.
        """

        self.HTerms = HTerms
        self.lAoCs = lAoCs
        self.rAoCs = rAoCs
        self.objMaps = objMaps
        self.dof = dof
        self.statistics = statistics
    # }}}

    def setHamiltonian(self):# {{{
        """
        Calculate the Hamiltonian matrix of the cluster.

        Must be overloaded by subclass method.
        If numbu representation is used, this method should set only one
        attribute: H, else this method should set three attributes: H, Hp, Hh
        """

        raise NotImplementedError
    # }}}

    def setGS(self):# {{{
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

    def setProjection(self):# {{{
        """
        Project the Hamiltonian matrix and ground state to the Krylov space.
        """

        raise NotImplementedError
    # }}}

    def setAll(self):# {{{
        """
        Set all attributes needed!
        """

        self.setHamiltonian()
        self.setGS()
        self.setProjection()
    # }}}

    def excitation(self, aocs, rbase, lbase=None):# {{{
        """
        Calculate the excitation state from the ground state.
        """

        t0 = time()
        res = OrderedDict()
        statemap = self.objMaps["statemap"]
        ket = self.GS
        for aoc in aocs:
            M = aoc.matrix_repr(statemap=statemap, right_bases=rbase, left_bases=lbase)
            res[aoc] = M.dot(ket)
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
        vec = np.zeros((dim, 1), dtype=np.float64)
        vec[0] = 1.0
        res = solve(matrix, vec)
        return res
    # }}}
    
    def _GFTerm(self, A, B, omega):# {{{
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
        lAoCMap = self.objMaps['lAoCMap']
        rAoCMap = self.objMaps['rAoCMap']
        for laoc in self.lAoCs:
            row = laoc.getIndex(lAoCMap)
            for raoc in self.rAoCs:
                col = raoc.getIndex(rAoCMap)
                res[row, col] = self._GFTerm(A=raoc, B=laoc, omega=omega)
        return res
    # }}}

    def __call__(self, omega):# {{{
        """
        Make instance of this clasallable.
        """

        return self.gf(omega)
    # }}}
# }}}


class GFEDNumbu(GFEDABC_P):# {{{
    """
    See also document of GFEDABC_P class.
    """

    def setHamiltonian(self):# {{{
        """
        See also document of GFEDABC_P class.
        """

        t0 = time()
        H = 0.0
        base = base_table(self.dof)
        statemap = self.objMaps["statemap"]
        for term in self.HTerms:
            H += term.matrix_repr(statemap=statemap, base=base)
        H += H.conjugate().transpose()
        self.H = H
        t1 = time()
        info = "The time spend on generating H matrix {0:.2f}.".format(t1 - t0)
        print(info)
        print('=' * len(info))
    # }}}

    def setProjection(self):# {{{
        base = base_table(self.dof)
        excitation = self.excitation(aocs=self.rAoCs, rbase=base, lbase=base)
        lanczosH = Lanczos(self.H)

        t0 = time()
        self.Hs_krylov, self.kets_krylov = lanczosH(excitation)
        t1 = time()
        info = "The time spend on projection {0:.2f}.".format(t1 - t0)
        print(info)
        print("=" * len(info))
    # }}}
# }}}


class GFEDNotNumbu(GFEDABC_P):# {{{
    """
    See also the document of the GFEDABC_P class.
    """

    def __init__(self, HTerms, lAoCs, rAoCs, objMaps, dof, 
                 statistics='F', occupy=None):# {{{
        if not(isinstance(occupy, int) or (occupy is None)):
            raise TypeError("The invalid occupy parameter!")

        self.occupy = occupy
        GFEDABC_P.__init__(self, HTerms, lAoCs, rAoCs, objMaps, dof, statistics)
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
        statemap = self.objMaps["statemap"]
        
        if self.occupy is None:
            for term in self.HTerms:
                H += term.matrix_repr(statemap=statemap, base=base)
            H += H.conjugate().transpose()
            Hp = Hh = H
        else:
            for term in self.HTerms:
                H += term.matrix_repr(statemap=statemap, base=base)
                Hp += term.matrix_repr(statemap=statemap, base=basep)
                Hh += term.matrix_repr(statemap=statemap, base=baseh)
            H += H.conjugate().transpose()
            Hp += Hp.conjugate().transpose()
            Hh += Hh.conjugate().transpose()

        self.H = H
        self.Hp = Hp
        self.Hh = Hh
    # }}}

    def setProjection(self):# {{{
        As = self.rAoCs
        Cs = self.lAoCs
        basep, base, baseh = self._base()
        h_excitation = self.excitation(aocs=As, lbase=baseh, rbase=base)
        p_excitation = self.excitation(aocs=Cs, lbase=basep, rbase=base)
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


def ParticleGF(HTerms, lAoCs, rAoCs, objMaps, dof, 
               statistics='F', occupy=None, numbu=False):# {{{
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
    stateMap: IndexMap
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
    res: GFEDNumbu or GFEDNotNumbu
    """

    if numbu:
        gf = GFEDNumbu(HTerms, lAoCs, rAoCs, objMaps, dof, statistics)
    else:
        gf = GFEDNotNumbu(HTerms, lAoCs, rAoCs, objMaps, dof, statistics, occupy)

    gf.setAll()
    return gf
# }}}


def SpinGF():
    pass
