"""
A test script for the lanczos module.
"""


import numpy as np
import pytest
from scipy.sparse import random
from scipy.sparse.linalg import eigsh

from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.lanczos import Lanczos
from HamiltonianPy.quantumoperator.constant import ANNIHILATION, CREATION
from HamiltonianPy.quantumoperator.particlesystem import AoC, ParticleTerm
from HamiltonianPy.quantumoperator.quantumstate import StateID


class TestLanczos:
    def test_init(self):
        dim = 100
        HM = random(dim, dim, format="csr") * 1j
        HM += random(dim, dim, format="csr")

        with pytest.raises(
            AssertionError, match="`HM` must be instance of csr_matrix"
        ):
            Lanczos(HM.tocsc())

        with pytest.raises(AssertionError, match="`HM` must be Hermitian"):
            Lanczos(HM)

        HM += HM.getH()
        lanczos_solver = Lanczos(HM)
        assert lanczos_solver._HMDim == dim

    def test_ground_state(self):
        dim = 100
        HM = random(dim, dim , format="csr") * 1j
        HM += random(dim, dim , format="csr")
        HM += HM.getH()

        lanczos_solver = Lanczos(HM)
        GE = lanczos_solver.ground_state()
        GE_ref = np.linalg.eigvalsh(HM.toarray())[0]
        assert np.allclose(GE, GE_ref)

    @pytest.mark.slow
    def test_call(self):
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        site_num = 12
        sites = np.arange(site_num).reshape((-1, 1))
        state_indices_table = IndexTable(StateID(site=site) for site in sites)
        bases = np.arange(1<<site_num, dtype=np.uint64)

        HM = 0.0
        for i in range(site_num):
            C = AoC(CREATION, site=sites[i])
            A = AoC(ANNIHILATION, site=sites[(i+1) % site_num])
            HM += ParticleTerm([C, A]).matrix_repr(state_indices_table, bases)
        HM += HM.getH()
        (GE, ), GS = eigsh(HM, k=1, which="SA")

        excited_states = {}
        i, j = np.random.randint(0, site_num, 2)
        keys = ["Ci", "Cj", "Ai", "Aj"]
        otypes = [CREATION, CREATION, ANNIHILATION, ANNIHILATION]
        indices = [i, j, i, j]
        for key, otype, index in zip(keys, otypes, indices):
            excited_states[key] = AoC(otype, site=sites[i]).matrix_repr(
                state_indices_table, bases
            ).dot(GS)

        lanczos_solver = Lanczos(HM)
        krylov_reprs_matrix, krylov_reprs_vectors = lanczos_solver(
            excited_states
        )
        omega = np.random.random() + 0.01j

        pairs = [("Ci", "Cj"), ("Ai", "Aj")]
        for coeff, (key0, key1) in zip([-1, 1], pairs):
            HMProjected = krylov_reprs_matrix[key1]
            krylov_subspace_dim = HMProjected.shape[0]
            I = np.identity(krylov_subspace_dim)
            bra = krylov_reprs_vectors[key1][key0]
            ket = np.zeros((krylov_subspace_dim, 1))
            ket[0, 0] = krylov_reprs_vectors[key1][key1][0]
            gf = np.vdot(
                bra, np.linalg.solve(
                    (omega + coeff * GE) * I + coeff * HMProjected, ket
                )
            )
            I = np.identity(len(bases))
            gf_ref = np.vdot(
                excited_states[key0],
                np.linalg.solve(
                    (omega + coeff * GE) * I + coeff * HM.toarray(),
                    excited_states[key1]
                )
            )
            assert np.allclose(gf, gf_ref)
