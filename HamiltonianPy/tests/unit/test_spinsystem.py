"""
A test script for the spinsystem module in the quantumoperator subpackage.
"""


import numpy as np
import pytest

from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.quantumoperator.constant import ANNIHILATION, CREATION, \
    SPIN_DOWN, SPIN_UP
from HamiltonianPy.quantumoperator.particlesystem import AoC
from HamiltonianPy.quantumoperator.quantumstate import SiteID
from HamiltonianPy.quantumoperator.spinsystem import *


class TestSpinOperator:
    def test_init(self):
        site = (1.2, 2.3)
        with pytest.raises(AssertionError, match="Invalid operator type"):
            SpinOperator(otype="X", site=site)

        sx = SpinOperator(otype="x", site=site)
        assert sx.otype == "x"
        assert sx.site_id == SiteID(site=site)
        assert sx.coordinate == site
        assert np.all(sx.site == site)
        assert repr(sx) == 'SpinOperator(otype="x", site=(1.2, 2.3))'

    def test_getIndex(self):
        otypes = ("x", "y", "z", "m", "p")
        sites = np.random.random((3, 3))
        indices_table = IndexTable(
            SpinOperator(otype=otype, site=site)
            for otype in otypes for site in sites
        )

        for index, operator in indices_table:
            assert index == operator.getIndex(indices_table)

    def test_hash(self):
        sites = [(0, 0), (1E-4, 1E-4), (1E-8, 1E-8)]
        sx0 = SpinOperator("x", site=sites[0])
        sx1 = SpinOperator("x", site=sites[1])
        sx2 = SpinOperator("x", site=sites[2])

        assert hash(sx0) != hash(sx1)
        assert hash(sx0) == hash(sx2)
        assert hash(sx1) != hash(sx2)

    def test_comparison(self):
        sx0 = SpinOperator("x", site=(0, 0))
        sy0 = SpinOperator("y", site=(0, 0))
        sz0 = SpinOperator("z", site=(0, 0))
        sp0 = SpinOperator("p", site=(0, 0))
        sm0 = SpinOperator("m", site=(0, 0))

        sx1 = SpinOperator("x", site=(0, 1))
        sy1 = SpinOperator("y", site=(0, 1))
        sz1 = SpinOperator("z", site=(0, 1))
        sp1 = SpinOperator("p", site=(0, 1))
        sm1 = SpinOperator("m", site=(0, 1))

        assert sm0 < sm1 < sp0 < sp1 < sx0 < sx1 < sy0 < sy1 < sz0 < sz1
        assert sz1 > sz0 > sy1 > sy0 > sx1 > sx0 > sp1 > sp0 > sm1 > sm0
        assert sm0 <= sm1
        assert sz1 >= sz0
        assert not (sx0 == sy1)
        assert sy0 != sm1

        with pytest.raises(TypeError, match="'<' not supported"):
            assert sx0 < 0
        with pytest.raises(TypeError, match="'<=' not supported"):
            assert sy0 <= 0
        with pytest.raises(TypeError, match="'>' not supported"):
            assert sz1 > 0
        with pytest.raises(TypeError, match="'>=' not supported"):
            assert sp1 >= 0

    def test_multiply(self):
        sx0 = SpinOperator("x", site=(0, 1))
        sy1 = SpinOperator("y", site=(0, 0))

        res = sx0 * sy1
        assert res.coeff == 1.0
        assert res.components == (sy1, sx0)

        res = sx0 * 0.5
        assert res.coeff == 0.5
        assert res.components == (sx0, )

        res = sy1 * sx0
        assert res.coeff == 1.0
        assert res.components == (sy1, sx0)

        res = 0.5j * sy1
        assert res.coeff == 0.5j
        assert res.components == (sy1, )

    def test_matrix(self):
        sx = SpinOperator("x", site=np.random.random(3))
        assert np.all(sx.matrix() == np.array([[0, 0.5], [0.5, 0.0]]))

        sy = SpinOperator("y", site=np.random.random(3))
        assert np.all(sy.matrix() == np.array([[0, -0.5j], [0.5j, 0.0]]))

        sz = SpinOperator("z", site=np.random.random(3))
        assert np.all(sz.matrix() == np.array([[0.5, 0], [0, -0.5]]))

        sp = SpinOperator("p", site=np.random.random(3))
        assert np.all(sp.matrix() == np.array([[0, 1], [0, 0]]))

        sm = SpinOperator("m", site=np.random.random(3))
        assert np.all(sm.matrix() == np.array([[0, 0], [1, 0]]))

    def test_dagger(self):
        site = np.random.random(3)
        sx = SpinOperator("x", site=site)
        sy = SpinOperator("y", site=site)
        sz = SpinOperator("z", site=site)
        sp = SpinOperator("p", site=site)
        sm = SpinOperator("m", site=site)

        assert sx.dagger() is sx
        assert sy.dagger() is sy
        assert sz.dagger() is sz
        assert sp.dagger() == sm
        assert sm.dagger() == sp

    def test_conjugate_of(self):
        site = np.random.random(3)
        sx = SpinOperator("x", site=site)
        sy = SpinOperator("y", site=site)
        sz = SpinOperator("z", site=site)
        sp = SpinOperator("p", site=site)
        sm = SpinOperator("m", site=site)

        assert sx.conjugate_of(sx)
        assert sy.conjugate_of(sy)
        assert sz.conjugate_of(sz)
        assert sp.conjugate_of(sm)
        assert sm.conjugate_of(sp)

        assert not sp.conjugate_of(sp)
        assert not sx.conjugate_of(sy)

        with pytest.raises(TypeError, match="not instance of this class"):
            sp.conjugate_of(0)

    def test_Schwinger(self):
        site = np.random.random(3)
        sx = SpinOperator("x", site=site)
        sy = SpinOperator("y", site=site)
        sz = SpinOperator("z", site=site)
        sp = SpinOperator("p", site=site)
        sm = SpinOperator("m", site=site)
        C_UP = AoC(CREATION, site=site, spin=SPIN_UP)
        C_DOWN = AoC(CREATION, site=site, spin=SPIN_DOWN)
        A_UP = AoC(ANNIHILATION, site=site, spin=SPIN_UP)
        A_DOWN = AoC(ANNIHILATION, site=site, spin=SPIN_DOWN)

        terms = sx.Schwinger()
        assert len(terms) == 2
        assert terms[0].coeff == 0.5
        assert terms[1].coeff == 0.5
        assert terms[0].components == (C_UP, A_DOWN)
        assert terms[1].components == (C_DOWN, A_UP)

        terms = sy.Schwinger()
        assert len(terms) == 2
        assert terms[0].coeff == -0.5j
        assert terms[1].coeff == 0.5j
        assert terms[0].components == (C_UP, A_DOWN)
        assert terms[1].components == (C_DOWN, A_UP)

        terms = sz.Schwinger()
        assert len(terms) == 2
        assert terms[0].coeff == 0.5
        assert terms[1].coeff == -0.5
        assert terms[0].components == (C_UP, A_UP)
        assert terms[1].components == (C_DOWN, A_DOWN)

        terms = sp.Schwinger()
        assert len(terms) == 1
        assert terms[0].coeff == 1
        assert terms[0].components == (C_UP, A_DOWN)

        terms = sm.Schwinger()
        assert len(terms) == 1
        assert terms[0].coeff == 1
        assert terms[0].components == (C_DOWN, A_UP)

    def test_matrix_repr(self):
        sites = np.array([[0, 0], [1, 1]])
        site_indices_table = IndexTable(SiteID(site=site) for site in sites)

        sx = SpinOperator("x", site=sites[0])
        M = sx.matrix_repr(site_indices_table).toarray()
        M_ref = np.array(
            [[0, 0.5, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 0.5], [0, 0, 0.5, 0]]
        )
        assert np.all(M == M_ref)

        sy = SpinOperator("y", site=sites[1])
        M = sy.matrix_repr(site_indices_table).toarray()
        M_ref = np.array(
            [
                [0, 0, -0.5j, 0], [0, 0, 0, -0.5j],
                [0.5j, 0, 0, 0], [0, 0.5j, 0, 0]
            ]
        )
        assert np.all(M == M_ref)


class TestSpinInteraction:
    def test_coeff(self):
        sites = np.random.random((2, 3))
        sx = SpinOperator("x", site=sites[0])
        sy = SpinOperator("y", site=sites[1])
        term = SpinInteraction([sx, sy])

        assert term.coeff == 1.0

        with pytest.raises(AssertionError, match="Invalid coefficient"):
            term.coeff = "test"

        term.coeff = 0.5
        assert term.coeff == 0.5

    def test_multiply(self):
        sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        sx = SpinOperator("x", site=sites[0])
        sy = SpinOperator("y", site=sites[1])
        sz = SpinOperator("z", site=sites[2])

        term0 = SpinInteraction([sx, sy], coeff=2.0)
        term1 = SpinInteraction([sy, sz], coeff=-1.0)

        res = term0 * term1
        assert res.coeff == -2.0
        assert res.components == (sx, sy, sy, sz)

        res = term1 * term0
        assert res.coeff == -2.0
        assert res.components == (sx, sy, sy, sz)

        res = term0 * sz
        assert res.coeff == 2.0
        assert res.components == (sx, sy, sz)

        res = term0 * 0.5
        assert res.coeff == 1.0
        assert res.components == (sx, sy)

        res = sz * term0
        assert res.coeff == 2
        assert res.components == (sx, sy, sz)

        res = 0.5j * term0
        assert res.coeff == 1.0j
        assert res.components == (sx, sy)

    def test_dagger(self):
        sites = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        for otype in ("x", "y", "z"):
            s0 = SpinOperator(otype, site=sites[0])
            s1 = SpinOperator(otype, site=sites[1])
            term = SpinInteraction([s0, s1], coeff=1j)
            term_dagger = term.dagger()
            assert term_dagger.coeff == -1j
            assert term_dagger.components == (s0, s1)
        sp0 = SpinOperator("p", site=sites[0])
        sp1 = SpinOperator("p", site=sites[1])
        sm0 = SpinOperator("m", site=sites[0])
        sm1 = SpinOperator("m", site=sites[1])
        term = SpinInteraction((sp0, sm1), coeff=1j)
        term_dagger = term.dagger()
        assert term_dagger.coeff == -1j
        assert term_dagger.components == (sm0, sp1)

        sx = SpinOperator("x", site=sites[0])
        sp = SpinOperator("p", site=sites[0])
        sm = SpinOperator("m", site=sites[0])
        term = SpinInteraction((sx, sp, sm), coeff=1j)
        term_dagger = term.dagger()
        assert term_dagger.coeff == -1j
        assert term_dagger.components == (sp, sm, sx)

    def test_Schwinger(self):
        sites = np.array([[0, 0], [1, 1]])
        sx0 = SpinOperator("x", site=sites[0])
        sx1 = SpinOperator("x", site=sites[1])
        sy0 = SpinOperator("y", site=sites[0])
        sy1 = SpinOperator("y", site=sites[1])
        sz0 = SpinOperator("z", site=sites[0])
        sz1 = SpinOperator("z", site=sites[1])

        C0_UP = AoC(CREATION, site=sites[0], spin=SPIN_UP)
        C1_UP = AoC(CREATION, site=sites[1], spin=SPIN_UP)
        C0_DOWN = AoC(CREATION, site=sites[0], spin=SPIN_DOWN)
        C1_DOWN = AoC(CREATION, site=sites[1], spin=SPIN_DOWN)
        A0_UP = AoC(ANNIHILATION, site=sites[0], spin=SPIN_UP)
        A1_UP = AoC(ANNIHILATION, site=sites[1], spin=SPIN_UP)
        A0_DOWN = AoC(ANNIHILATION, site=sites[0], spin=SPIN_DOWN)
        A1_DOWN = AoC(ANNIHILATION, site=sites[1], spin=SPIN_DOWN)

        sx0_times_sx1 = SpinInteraction((sx0, sx1))
        sy0_times_sy1 = SpinInteraction((sy0, sy1))
        sz0_times_sz1 = SpinInteraction((sz0, sz1))

        sx0_times_sx1_schwinger = sx0_times_sx1.Schwinger()
        sy0_times_sy1_schwinger = sy0_times_sy1.Schwinger()
        sz0_times_sz1_schwinger = sz0_times_sz1.Schwinger()

        assert sx0_times_sx1_schwinger[0].coeff == 0.25
        assert sx0_times_sx1_schwinger[0].components == (
            C0_UP, A0_DOWN, C1_UP, A1_DOWN
        )
        assert sx0_times_sx1_schwinger[1].coeff == 0.25
        assert sx0_times_sx1_schwinger[1].components == (
            C0_UP, A0_DOWN, C1_DOWN, A1_UP
        )
        assert sx0_times_sx1_schwinger[2].coeff == 0.25
        assert sx0_times_sx1_schwinger[2].components == (
            C0_DOWN, A0_UP, C1_UP, A1_DOWN
        )
        assert sx0_times_sx1_schwinger[3].coeff == 0.25
        assert sx0_times_sx1_schwinger[3].components == (
            C0_DOWN, A0_UP, C1_DOWN, A1_UP
        )

        assert sy0_times_sy1_schwinger[0].coeff == -0.25
        assert sy0_times_sy1_schwinger[0].components == (
            C0_UP, A0_DOWN, C1_UP, A1_DOWN
        )
        assert sy0_times_sy1_schwinger[1].coeff == 0.25
        assert sy0_times_sy1_schwinger[1].components == (
            C0_UP, A0_DOWN, C1_DOWN, A1_UP
        )
        assert sy0_times_sy1_schwinger[2].coeff == 0.25
        assert sy0_times_sy1_schwinger[2].components == (
            C0_DOWN, A0_UP, C1_UP, A1_DOWN
        )
        assert sy0_times_sy1_schwinger[3].coeff == -0.25
        assert sy0_times_sy1_schwinger[3].components == (
            C0_DOWN, A0_UP, C1_DOWN, A1_UP
        )

        assert sz0_times_sz1_schwinger[0].coeff == 0.25
        assert sz0_times_sz1_schwinger[0].components == (
            C0_UP, A0_UP, C1_UP, A1_UP
        )
        assert sz0_times_sz1_schwinger[1].coeff == -0.25
        assert sz0_times_sz1_schwinger[1].components == (
            C0_UP, A0_UP, C1_DOWN, A1_DOWN
        )
        assert sz0_times_sz1_schwinger[2].coeff == -0.25
        assert sz0_times_sz1_schwinger[2].components == (
            C0_DOWN, A0_DOWN, C1_UP, A1_UP
        )
        assert sz0_times_sz1_schwinger[3].coeff == 0.25
        assert sz0_times_sz1_schwinger[3].components == (
            C0_DOWN, A0_DOWN, C1_DOWN, A1_DOWN
        )

    def test_matrix_repr(self):
        sites = np.array([[0, 0], [1, 1]])
        site_indices_table = IndexTable(SiteID(site=site) for site in sites)
        SX = np.array([[0, 0.5], [0.5, 0]])
        SY = np.array([[0, -0.5j], [0.5j, 0]])
        SZ = np.array([[0.5, 0], [0, -0.5]])
        I = np.array([[1, 0], [0, 1]])

        for otype, SMatrix in zip(["x", "y", "z"], [SX, SY, SZ]):
            s0 = SpinOperator(otype, site=sites[0])
            s1 = SpinOperator(otype, site=sites[1])
            M = SpinInteraction((s0, s1)).matrix_repr(
                site_indices_table
            ).toarray()
            assert np.all(M == np.kron(SMatrix, SMatrix))

        sx0 = SpinOperator("x", site=sites[0])
        sy0 = SpinOperator("y", site=sites[0])
        sz0 = SpinOperator("z", site=sites[0])

        M = SpinInteraction((sx0, sy0)).matrix_repr(
            site_indices_table
        ).toarray()
        assert np.all(M == np.kron(I, np.dot(SX, SY)))

        M = SpinInteraction((sy0, sz0)).matrix_repr(
            site_indices_table
        ).toarray()
        assert np.all(M == np.kron(I, np.dot(SY, SZ)))

        M = SpinInteraction((sx0, sz0)).matrix_repr(
            site_indices_table
        ).toarray()
        assert np.all(M == np.kron(I, np.dot(SX, SZ)))
