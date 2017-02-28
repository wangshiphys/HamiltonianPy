from copy import deepcopy
from itertools import combinations, product
from scipy.sparse import csr_matrix

import numpy as np

from HamiltonianPy.constant  import CREATION, ANNIHILATION
from HamiltonianPy.indexmap import IndexMap
from HamiltonianPy.optor import AoC, Optor, OptorWithIndex, State

__all__ = ['Model', 'ModelBasic', 'Periodization']


class ModelBasic:# {{{
    """
    This class provide a basic description of a model defined on a lattice.

    Attribute:
    ----------
    cluster: class Lattice
        The cluster on which the model is defined on.
    numbu: bool, optional
        Determine whether to use numbu representation.
        default: False
    spins: tuple
        All possible spin index.
    orbits: tuple
        All possible orbit index.
    internals: tuple
        All possible internal degree of freedom.
    lAoCs: list
        All concerned creation and/or annihilation operators. This is the
        collection of all operators in left side of the matrix representation 
        of perturbation V.
        For example, if we write V in this form
        (C0, C1, C2, C3,...)(V_matrix)(A0, A1, A2, A3, ...)^T, than lAoCs is
        all these C operators in the left side and rAoCs is all 
        these A operators in the right side.
    rAoCs: list
        All concerned creation and/or annihilation operators. This is the
        collections of all operators in right side of the matrix representation 
        of Green function. All operators in this collection is the Hermit 
        conjugate of the corresponding operator in lAoCs.
    StateMap: IndexMap
        A map from all the single particle state to integer number.
    lAoCMap: IndexMap
        A map from all operators in lAoCs to integer number.
    rAoCMap: IndexMap
        A map from all operators in rAoCs to integer number.

    Method:
    -------
    __init__(cluster, spin_dof=2, orbit_dof=1, numbu=False)
    """

    def __init__(self, cluster, spin_dof=2, orbit_dof=1, numbu=False):# {{{
        """
        Initilize the instance of this class!

        Parameter:
        ----------
        See the docstring of this class.
        """

        spins = tuple(range(spin_dof))
        orbits = tuple(range(orbit_dof))
        internals = tuple(product(orbits, spins, repeat=1))
        states = []
        lAoCs = []
        rAoCs = []

        otypes = (CREATION, )
        if numbu:
            otypes = (CREATION, ANNIHILATION)

        AoC_cfg = product(otypes, orbits, spins, cluster.points, repeat=1)
        for otype, orbit, spin, site in AoC_cfg:
            aoc = AoC(otype=otype, site=site, spin=spin, orbit=orbit)
            lAoCs.append(aoc)
            rAoCs.append(aoc.dagger())
            if otype == CREATION:
                state = State(site=site, spin=spin, orbit=orbit)
                states.append(state)

        self.cluster = cluster
        self.numbu = numbu
        self.spins = spins
        self.orbits = orbits
        self.internals = internals
        self.lAoCs = lAoCs
        self.rAoCs = rAoCs
        self.dim = len(lAoCs)
        self.StateMap = IndexMap(states)
        self.lAoCMap = IndexMap(lAoCs)
        self.rAoCMap = IndexMap(rAoCs)
    # }}}
# }}}


class Model(ModelBasic):
    """
    This class provide a complete description of a model defined on a lattice.

    Attribute:
    ----------
    max_neighbor: int, optional
        The maximum neighboring this model might involve. 0, 1, 2 represents
        onsite, nearest, next-nearest neighbor, respectively.
    _HOptors: list
        A collection of all the hopping, pairing, hubbard terms that this 
        model might include. All coefficients of these terms are with
        default value 1.0, so they cannot be used directly to calculate
        Hamiltonian matrix. This attribute is generate in the 
        initilization process.
    _VOptors: list
        A collection of all the terms that are to be treat as a perturbation.
        Their coefficients are also set to default value 1.0, and should not be
        used directly to perturbation term. This attribute is generate in the
        initilization process.
    HTerms: list
        The same meaning as the _HOptors attribute, but with all coefficients
        set correctly, and it is used to calculate the Hamiltonian matrix.
    VTerms: list
        The same meaning as the _VOptors attribute, but with all coefficients
        set correctly, and it is used to calculate the perturbation matrix V.
    template: dict
        The template for generating perturbation matrix V.

    See also ModelBasic for more attributes, explainations.

    Method:
    -------
    __init__(cluster, spin_dof=2, orbit_dof=2, max_neighbor=1, numbu=False)
    _quadratic(tag='intra')
    _hubbard()
    _mu()
    _setOptors()
    _setCoeff()
    _setTemplate()
    _update(coeff_dict, coeff_generator=None)
    __call__(coeff_dict, coeff_generator=None)
    perturbation(k)
    
    Staticmethod:
    -------------
    _AC_Generator(**tags)
    """

    def __init__(self, cluster, spin_dof=2, orbit_dof=1, 
                 max_neighbor=1, numbu=False):# {{{
        """
        Initilize the instance of this class.

        See also the docstring of this class.
        """

        ModelBasic.__init__(self, cluster, spin_dof, orbit_dof, numbu)
        self.max_neighbor = max_neighbor
        self._setOptors()
    # }}}

    @staticmethod
    def _AC_Generator(*tags):# {{{
        """
        Given collection of state tag, this function generator corresponding
        creation and annihilation operators.
        """

        res = []
        for orbit, spin, site in tags:
            C = AoC(site=site, spin=spin, orbit=orbit, otype=CREATION)
            A = AoC(site=site, spin=spin, orbit=orbit, otype=ANNIHILATION)
            res.append((C, A))
        return res
    # }}}

    def _quadratic(self, bonds, tag='intra'):# {{{
        """
        Generate all possible hopping and/or pairing term on the given bonds.

        Parameter:
        ----------
        bonds: list
            A collection of bonds that might host hopping or pairing term.

        Return:
        -------
        res: list
            Return all possible quadratic terms on these bonds.
        """

        res = []
        config = list(product(self.internals, repeat=2))
        for bond in bonds:
            site0, site1 = bond 
            for (orbit0, spin0), (orbit1, spin1) in config:
                tag0 = (orbit0, spin0, site0)
                tag1 = (orbit1, spin1, site1)
                (C0, A0), (C1, A1) = self._AC_Generator(tag0, tag1)
                if tag == 'inter':
                    hopping = Optor((C0, A1))
                else:
                    hopping = OptorWithIndex((C0, A1), statemap=self.StateMap)
                res.append(hopping)

                if self.numbu:
                    if tag == 'inter':
                        pair = Optor((A0, A1))
                    else:
                        pair = OptorWithIndex((A0, A1), statemap=self.StateMap)
                    res.append(pair)
        return res
    # }}}

    def _hubbard(self):# {{{
        """
        Generate all possible on site interaction term!
        """

        res = []
        config = list(combinations(self.internals, r=2))
        for site in self.cluster.points:
            for (orbit0, spin0), (orbit1, spin1) in config:
                tag0 = (orbit0, spin0, site)
                tag1 = (orbit1, spin1, site)
                (C0, A0), (C1, A1) = self._AC_Generator(tag0, tag1)
                optor = OptorWithIndex((C0, A0, C1, A1), statemap=self.StateMap)
                res.append(optor)
        return res
    # }}}

    def _mu(self):# {{{
        """
        Generate the chemical potential term!
        """

        res = []
        for site in self.cluster.points:
            for orbit, spin in self.internals:
                tag = (orbit, spin, site)
                (C, A), = self._AC_Generator(tag)
                optor = OptorWithIndex((C, A), statemap=self.StateMap)
                res.append(optor)
        return res
    # }}}

    def _setOptors(self):# {{{
        """
        Generate all possible terms this model might include!
        """

        intra, inter = self.cluster.bonds(self.max_neighbor)
        quadratic_optors = self._quadratic(bonds=intra, tag='intra')
        hubbard_optors = self._hubbard()
        mu_optors = self._mu()
        resH = mu_optors + quadratic_optors + hubbard_optors
        resV = self._quadratic(bonds=inter, tag='inter')
        self._HOptors = resH
        self._VOptors = resV
    # }}}

    def _setCoeff(self, coeff_dict, coeff_generator):# {{{
        """
        Set the coefficients of all terms of the Hamiltonian!
        """

        HTerms = []
        VTerms = []
        HOptors = deepcopy(self._HOptors)
        VOptors = deepcopy(self._VOptors)
        lH = len(HOptors)
        lV = len(VOptors)

        for optor in HOptors:
            coeff = coeff_generator(optor, **coeff_dict)
            if coeff != 0.0:
                if optor.isSelfConjugate():
                    optor.updatecoeff(coeff / 2.0)
                else:
                    optor.updatecoeff(coeff)
                HTerms.append(optor)
        
        for optor in VOptors:
            coeff = coeff_generator(optor, **coeff_dict)
            if coeff != 0.0:
                if optor.isSelfConjugate():
                    optor.updatecoeff(coeff / 2.0)
                else:
                    optor.updatecoeff(coeff)
                VTerms.append(optor)
        
        self.HTerms = HTerms
        self.VTerms = VTerms
    # }}}

    def _setTemplate(self):# {{{
        """
        Set the template for calculating the perturbation V matrix!
        """

        row = []
        col = []
        coeffs = []
        dRs = []
        vterms = deepcopy(self.VTerms)
        lv = len(vterms)
        for term in vterms:
            coeff = term.coeff
            aoc0, aoc1 = term.aocs
            site0, dR0 = self.cluster.decompose(aoc0.site)
            site1, dR1 = self.cluster.decompose(aoc1.site)
            new_aoc0 = aoc0.update(site=site0)
            new_aoc1 = aoc1.update(site=site1)
            lindex0 = self.lAoCMap(new_aoc0)
            rindex1 = self.rAoCMap(new_aoc1)
            if self.numbu:
                if term.ishopping():
                    rindex0 = self.rAoCMap(new_aoc0)
                    lindex1 = self.lAoCMap(new_aoc1)
                    row.extend([lindex0, lindex1])
                    col.extend([rindex1, rindex0])
                    coeffs.extend([coeff, - coeff])
                    dRs.extend([1j * (dR1 - dR0), 1j * (dR0 - dR1)])
                elif terms.ispairing():
                    row.append(lindex0)
                    col.append(rindex1)
                    coeffs.append(coeff)
                    dRs.append(1j * (dR1 - dR0))
                else:
                    raise ValueError("The unsupported term!")
            else:
                row.append(lindex0)
                col.append(rindex1)
                coeffs.append(coeff)
                dRs.append(1j * (dR1 - dR0))

        self.template = (tuple(row), tuple(col), tuple(coeffs), tuple(dRs))
    # }}}

    def perturbation(self, k):# {{{
        """
        Calculate the perturbation V matrix with specific k point.

        This can not be called before the model paramter has be set. That it to
        say you have to at least call the update or __call__ method once before
        you can use this function!
        """

        shape = (self.dim, self.dim)
        row, col, coeffs, dRs = self.template
        drs = np.array(dRs).T
        data = np.array(coeffs) * np.exp(np.dot(k, drs))
        res = csr_matrix((data, (row, col)), shape=shape)
        return res.toarray()
    # }}}
    
    def update(self, coeff_dict, coeff_generator):# {{{
        """
        Update or set the model parameter!
        """

        if callable(coeff_generator):
            self._setCoeff(coeff_dict, coeff_generator)
            self._setTemplate()
        else:
            raise TypeError("The input generator is not callable!")
    # }}}

    def __call__(self, coeff_dict, coeff_generator):# {{{
        self.update(coeff_dict, coeff_generator)
        return deepcopy(self.HTerms)
    # }}}


class Periodization:# {{{
    """
    This class implement the periodization procedure of CPT.

    Attribute:
    ----------
    cluster: instance of class Lattice.
        The cluster of CPT.
    cell: instance of class Lattice.
        The unit cell of the model.
    lAoCMap_cluster: IndexMap
    rAoCMap_cluster: IndexMap
    lAoCMap_cell: IndexMap
    rAoCMap_cell: IndexMap
        See also the documentation of ModelBasic class.
    tamplate: tuple
        The template of fourier transformation.

    Method:
    -------
    __init__(cluster, cell, spin_dof=2, orbit_dof=1, numbu=False)
    decompose(index, tag='L')
    setTemplate()
    fourier(M, k)
    __call__(M, )
    """

    def __init__(self, cluster, cell, spin_dof=2, orbit_dof=1, numbu=False):# {{{
        """
        Initilize the instance of this class.
        """

        clusterModel = ModelBasic(cluster=cluster, spin_dof=spin_dof, 
                                  orbit_dof=orbit_dof, numbu=numbu)
        cellModel = ModelBasic(cluster=cell, spin_dof=spin_dof, 
                               orbit_dof=orbit_dof, numbu=numbu)

        self.cluster = cluster
        self.cell = cell
        self.lAoCMap_cluster = clusterModel.lAoCMap
        self.rAoCMap_cluster = clusterModel.rAoCMap
        self.lAoCMap_cell = cellModel.lAoCMap
        self.rAoCMap_cell = cellModel.rAoCMap
        self.dim = len(cellModel.lAoCMap)
        self.setTemplate()
    # }}}

    def _decompose(self, index, tag='L'):# {{{
        """
        Giving a creation or annihilation operator which belong to the cluster,
        this method find the equivalent operator belong to the cell.

        What we mean equivalent here: The site index of the two operators can be
        linked through a translation along the translation vector of the unit
        cell, and all other index of the two operators are identitical.
        The operator belong to the cluster and cell are all indicated by the
        position in the AoCMap which is integer index.
        """

        if tag == 'L':
            clusterMap = self.lAoCMap_cluster
            cellMap = self.lAoCMap_cell
        elif tag == 'R':
            clusterMap = self.rAoCMap_cluster
            cellMap = self.rAoCMap_cell
        else:
            raise ValueError("The invalid tag!")

        aoc = clusterMap(index)
        site = aoc.site
        eqv_site, dR = self.cell.decompose(site)
        new_aoc = aoc.update(site=eqv_site)
        res = cellMap(new_aoc)
        #if aoc.otype == ANNIHILATION:
        #    dR = -dR
        #if tag == 'R':
        #    dR = -dR
        return res, dR
    # }}}

    def setTemplate(self):# {{{
        """
        Set the Template for fourier transformation.
        """

        row_dest = []
        col_dest = []
        row_src = []
        col_src = []
        dRs = []
        dim = len(self.lAoCMap_cluster)
        for row in range(dim):
            index0, dR0 = self._decompose(index=row, tag='L')
            for col in range(dim):
                index1, dR1 = self._decompose(index=col, tag='R')
                row_dest.append(index0)
                col_dest.append(index1)
                row_src.append(row)
                col_src.append(col)
                dRs.append(1j * (dR1 - dR0))
        self.template = (tuple(row_dest), tuple(col_dest), 
                         tuple(row_src), tuple(col_src), tuple(dRs))
        # }}}

    def _fourier(self, M, k):# {{{
        """
        Fourier transformation of the input M matrix to corresponding matrix
        belong to the cell.
        """

        shape = (self.dim, self.dim)
        row_dest, col_dest, row_src, col_src, dRs = self.template
        drs = np.array(dRs).T
        data = M[row_src, col_src] * np.exp(np.dot(k, drs))
        res = csr_matrix((data, (row_dest, col_dest)), shape=shape)
        return res.toarray()
    # }}}

    def __call__(self, M, k):# {{{
        """
        Return the fourier transformed matrix when call instance of this class.
        """

        return self._fourier(M, k)
    # }}}
# }}}
