"""
A test script for the particlesystem module in the quantumoperator subpackage.
"""


import numpy as np
import pytest

from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.quantumoperator.constant import ANNIHILATION, CREATION
from HamiltonianPy.quantumoperator.particlesystem import *
from HamiltonianPy.quantumoperator.quantumstate import StateID


@pytest.fixture(scope="module")
def creator():
    return AoC(CREATION, site=(0, 0))

@pytest.fixture(scope="module")
def annihilator():
    return AoC(ANNIHILATION, site=(0, 0))


class TestAoC:
    def test_init(self):
        site = (0, 0)
        with pytest.raises(AssertionError):
            AoC(otype=2, site=site)

        creator = AoC(CREATION, site=site)
        assert creator.otype == CREATION
        assert creator.state == StateID(site=site)
        assert creator.coordinate == site
        assert np.all(creator.site == site)
        assert creator.spin == 0
        assert creator.orbit == 0
        tmp = "AoC(otype=CREATION, site=(0, 0), spin=0, orbit=0)"
        assert repr(creator) == tmp

    def test_getIndex(self):
        spins = (0, 1)
        orbits = (2, 3)
        sites = np.random.random((3, 3))
        otypes = (CREATION, ANNIHILATION)
        aoc_indices_table = IndexTable(
            AoC(otype, site, spin, orbit)
            for otype in otypes for site in sites
            for spin in spins for orbit in orbits
        )
        for index, aoc in aoc_indices_table:
            assert index == aoc.getIndex(aoc_indices_table)

    def test_hash(self):
        site0 = (0, 0)
        site1 = (1E-4, 1E-4)
        site2 = (1E-8, 1E-8)
        aoc0 = AoC(CREATION, site=site0)
        aoc1 = AoC(CREATION, site=site1)
        aoc2 = AoC(CREATION, site=site2)

        assert hash(aoc0) != hash(aoc1)
        assert hash(aoc0) == hash(aoc2)
        assert hash(aoc1) != hash(aoc2)

    def test_comparison(self):
        c0 = AoC(CREATION, site = (0, 0))
        c1 = AoC(CREATION, site = (0, 1))
        a0 = AoC(ANNIHILATION, site=(0, 0))
        a1 = AoC(ANNIHILATION, site=(0, 1))

        assert c0 < c1 < a1 < a0
        assert  a0 > a1 > c1 > c0
        assert not (c0 == c1)
        assert a0 != a1
        assert c0 <= c1 <= a1 <= a0
        assert a0 >= a1 >= c1 >= c0

        with pytest.raises(TypeError, match="'<' not supported"):
            assert c0 < 0
        with pytest.raises(TypeError, match="'<=' not supported"):
            assert c1 <= 0
        with pytest.raises(TypeError, match="'>' not supported"):
            assert a0 > 0
        with pytest.raises(TypeError, match="'>=' not supported"):
            assert a1 >= 0

    def test_multiply(self, annihilator, creator):
        res = creator * 0.5
        assert res.coeff == 0.5
        assert res.components == (creator, )

        res = annihilator * creator
        assert res.coeff == 1.0
        assert res.components == (annihilator, creator)

        res = 0.5 * annihilator
        assert res.coeff == 0.5
        assert res.components == (annihilator, )

    def test_dagger(self, annihilator, creator):
        assert creator.dagger() == annihilator
        assert annihilator.dagger() == creator

    def test_conjugate_of(self, annihilator, creator):
        assert creator.conjugate_of(annihilator)
        assert annihilator.conjugate_of(creator)

        assert not creator.conjugate_of(creator)
        assert not annihilator.conjugate_of(annihilator)

        with pytest.raises(TypeError, match="not instance of this class"):
            creator.conjugate_of(0)

    def test_same_state(self, annihilator, creator):
        assert annihilator.same_state(annihilator)
        assert annihilator.same_state(creator)
        assert creator.same_state(annihilator)
        assert creator.same_state(creator)

        assert not creator.same_state(AoC(CREATION, site=(0, 1)))

        with pytest.raises(TypeError, match="not instance of this class"):
            assert creator.same_state(0)

    def test_derive(self, annihilator, creator):
        assert creator.derive(otype=ANNIHILATION) == annihilator
        assert annihilator.derive(site=(0, 1)) == AoC(
            ANNIHILATION, site=(0, 1)
        )

    def test_matrix_repr(self):
        # Currently the reliability of the `matrix_repr` method is guaranteed
        # by the `matrix_function` in `matrixrepr` module.
        pass


class TestNumberOperator:
    def test_init(self):
        site = (0, 0)
        N = NumberOperator(site=site)
        assert N.state == StateID(site=site)
        assert N.coordinate == site
        assert np.all(N.site == site)
        assert N.spin == 0
        assert N.orbit == 0
        assert repr(N) == "NumberOperator(site=(0, 0), spin=0, orbit=0)"

    def test_getIndex(self):
        spins = (0, 1)
        orbits = (2, 3)
        sites = np.random.random((3, 3))
        indices_table = IndexTable(
            NumberOperator(site, spin, orbit)
            for site in sites for spin in spins for orbit in orbits
        )
        for index, N in indices_table:
            assert index == N.getIndex(indices_table)

    def test_hash(self):
        sites = [(0, 0), (1E-4, 1E-4), (1E-8, 1E-8)]
        N0 = NumberOperator(site=sites[0])
        N1 = NumberOperator(site=sites[1])
        N2 = NumberOperator(site=sites[2])
        assert hash(N0) != hash(N1)
        assert hash(N0) == hash(N2)
        assert hash(N1) != hash(N2)

    def test_comparison(self):
        N0 = NumberOperator(site=(0, 0))
        N1 = NumberOperator(site=(0, 1))
        N2 = NumberOperator(site=(1, 0))
        N3 = NumberOperator(site=(1, 1))
        assert N0 < N1 < N2 < N3
        assert N3 > N2 > N1 > N0
        assert not (N0 == N1)
        assert N2 != N3
        assert N0 <= N1 <= N2 <= N3
        assert N3 >= N2 >= N1 >= N0

        with pytest.raises(TypeError, match="'<' not supported"):
            assert N0 < 0
        with pytest.raises(TypeError, match="'<=' not supported"):
            assert N1 <= 0
        with pytest.raises(TypeError, match="'>' not supported"):
            assert N2 > 0
        with pytest.raises(TypeError, match="'>=' not supported"):
            assert N3 >= 0

    def test_multiply(self):
        C0 = AoC(CREATION, site=(0, 0))
        A0 = AoC(ANNIHILATION, site=(0, 0))
        C1 = AoC(CREATION, site=(1, 1))
        A1 = AoC(ANNIHILATION, site=(1, 1))
        N0 = NumberOperator(site=(0, 0))
        N1 = NumberOperator(site=(1, 1))

        res = N0 * N1
        assert res.coeff == 1.0
        assert res.components == (C0, A0, C1, A1)

        res = N0 * 2
        assert res.coeff == 2.0
        assert res.components == (C0, A0)

        res = 3j * N1
        assert res.coeff == 3.0j
        assert res.components == (C1, A1)

        res = C0 * N1
        assert res.coeff == 1.0
        assert res.components == (C0, C1, A1)

        res = N0 * A1
        assert res.coeff == 1.0
        assert res.components == (C0, A0, A1)

        res = C0 * N1 * A0
        assert res.coeff == 1.0
        assert res.components == (C0, C1, A1, A0)

    def test_dagger(self):
        N0 = NumberOperator(site=(0, 0))
        assert N0.dagger() is N0

    def test_derive(self):
        N0 = NumberOperator(site=(0, 0))
        N1 = NumberOperator(site=(0, 1))
        assert N0.derive(site=(0, 1)) == N1

    def test_matrix_repr(self):
        # Currently the reliability of the `matrix_repr` method is guaranteed
        # by the `matrix_function` in `matrixrepr` module.
        pass


class TestParticleTerm:
    def test_coeff(self):
        sites = np.random.random((2, 3))
        C = AoC(CREATION, site=sites[0], spin=3, orbit=5)
        A = AoC(ANNIHILATION, site=sites[1], spin=9, orbit=13)
        term = ParticleTerm([C, A])

        assert term.coeff == 1.0
        with pytest.raises(AssertionError, match="Invalid coefficient"):
            term.coeff = "test"

        term.coeff = 0.5
        assert term.coeff == 0.5

    def test_check_compatibility(self):
        site = (0, 0)
        C_UP = AoC(CREATION, site=site, spin=1)
        C_DOWN = AoC(CREATION, site=site, spin=0)
        A_UP = AoC(ANNIHILATION, site=site, spin=1)
        A_DOWN = AoC(ANNIHILATION, site=site, spin=0)

        term = ParticleTerm([C_UP, A_UP], classification="hopping")
        assert not term.check_compatibility()
        term = ParticleTerm([C_UP, C_DOWN], classification="hopping")
        assert not term.check_compatibility()
        term = ParticleTerm([C_UP, A_DOWN], classification="hopping")
        assert term.check_compatibility()

        term = ParticleTerm([C_UP, A_UP], classification="number")
        assert term.check_compatibility()
        term = ParticleTerm([C_UP, C_DOWN], classification="number")
        assert not term.check_compatibility()
        term = ParticleTerm([C_UP, A_DOWN], classification="number")
        assert not term.check_compatibility()

        term = ParticleTerm(
            [C_UP, A_UP, C_DOWN, A_DOWN], classification="Coulomb"
        )
        assert term.check_compatibility()
        term = ParticleTerm(
            [C_UP, C_DOWN, A_DOWN, A_UP], classification="Coulomb"
        )
        assert not term.check_compatibility()

    def test_multiply(self):
        site = (0, 0)
        C_UP = AoC(CREATION, site=site, spin=1)
        C_DOWN = AoC(CREATION, site=site, spin=0)
        A_UP = AoC(ANNIHILATION, site=site, spin=1)
        A_DOWN = AoC(ANNIHILATION, site=site, spin=0)

        term0 = ParticleTerm([C_UP, A_UP], coeff=1.5)
        term1 = ParticleTerm([C_DOWN, A_DOWN], coeff=0.5)
        res = term0 * term1
        assert res.coeff == 0.75
        assert res.components == (C_UP, A_UP, C_DOWN, A_DOWN)

        res = term1 * term0
        assert res.coeff == 0.75
        assert res.components == (C_DOWN, A_DOWN, C_UP, A_UP)

        res = term0 * C_DOWN * A_DOWN
        assert res.coeff == 1.5
        assert res.components == (C_UP, A_UP, C_DOWN, A_DOWN)

        res = term0 * 2j
        assert res.coeff == 3.0j
        assert res.components == (C_UP, A_UP)

        res = C_UP * term1 * A_UP
        assert res.coeff == 0.5
        assert res.components == (C_UP, C_DOWN, A_DOWN, A_UP)

        res = 0.2j * term1
        assert res.coeff == 0.1j
        assert res.components== (C_DOWN, A_DOWN)

    def normalize(self):
        sites = [(0, 0), (0, 1), (1, 0), (1, 1)]
        C0 = AoC(CREATION, site=sites[0])
        A1 = AoC(ANNIHILATION, site=sites[1])
        C2 = AoC(CREATION, sites=sites[2])
        A3 = AoC(ANNIHILATION, site=sites[3])
        aocs = [C0, A1, C2, A3]
        aocs_normalized, swap = ParticleTerm.normalize(aocs)
        assert aocs_normalized == [C0, C2, A3, A1]
        assert swap == 2

    def test_dagger(self):
        sites = [(0, 1), (0, 1)]
        C0 = AoC(CREATION, site=sites[0])
        A0 = AoC(ANNIHILATION, site=sites[0])
        C1 = AoC(CREATION, site=sites[1])
        A1 = AoC(ANNIHILATION, site=sites[1])

        term = ParticleTerm([C0, A1], coeff=1j)
        term_dagger = term.dagger()
        assert term_dagger.coeff == -1j
        assert term_dagger.components == (C1, A0)

        term = ParticleTerm([C0, A0])
        term_dagger = term.dagger()
        assert term_dagger.coeff == 1.0
        assert term_dagger.components == (C0, A0)

        term = ParticleTerm([C0, C1], coeff=1j)
        term_dagger = term.dagger()
        assert term_dagger.coeff == -1j
        assert term_dagger.components == (A1, A0)

        term = ParticleTerm([C0, C1, A1, A0])
        term_dagger = term.dagger()
        assert term_dagger.coeff == 1.0
        assert term_dagger.components == (C0, C1, A1, A0)

    def test_matrix_repr(self):
        # Currently the reliability of the `matrix_repr` method is guaranteed
        # by the `matrix_function` in `matrixrepr` module.
        pass
