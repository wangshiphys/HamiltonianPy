"""
Calculate quantum-mechanical averages of quantum operators.
"""


import logging
from time import time

import numpy as np
from scipy.sparse import identity, kron

__all__ = [
    "QuantumAverages",
    "TotalSpinMatrices",
    "TotalSpinAverages",
]


# Matrix representation for single-spin operator
SX_MATRIX = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
# The real SY matrix is np.array([[0.0, -0.5j], [0.5j, 0.0]])
# The following  matrix is the imaginary part
SY_MATRIX_IMAG = np.array([[0.0, -0.5], [0.5, 0.0]], dtype=np.float64)
SZ_MATRIX = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64)


logging.getLogger(__name__).addHandler(logging.NullHandler())


def QuantumAverages(ket, operator):
    """
    Calculate the quantum-mechanical average of the `operator` over the `ket`.

    Parameters
    ----------
    ket : np.ndarray
        The matrix form of the quantum state over which the average is to be
        calculated.
    operator : csr_matrix
        The matrix representation of the `operator`.

    Returns
    -------
    avg : complex
        The corresponding quantum-mechanical average.
    """

    return np.vdot(ket, operator.dot(ket)) / np.vdot(ket, ket)


def TotalSpinMatrices(spin_num):
    """
    Calculate the matrix representation of total spin operators of a N-spin
    system.

    Parameters
    ----------
    spin_num : int
        The number of spin in the system.

    Returns
    -------
    total_sx : csr_matrix
        The matrix representation of the total-sx operator.
    total_sy_imag : csr_matrix
        The imaginary part of the matrix representation of the total-sy
        operator. Since the total_sy matrix is pure imaginary, so
        `total_sy = 1j * total_sy_imag`.
    total_sz : csr_matrix
        The matrix representation of the total-sz operator.
    """

    assert isinstance(spin_num, int) and spin_num > 0, "`spin_num` must be " \
                                                       "positive integer!"

    t0 = time()
    total_sx = total_sy_imag = total_sz = 0.0
    for index in range(spin_num):
        I0 = identity(1 << index, np.float64, "csr")
        I1 = identity(1 << (spin_num - index - 1), np.float64, "csr")
        total_sx += kron(I1, kron(SX_MATRIX, I0, "csr"), "csr")
        total_sz += kron(I1, kron(SZ_MATRIX, I0, "csr"), "csr")
        total_sy_imag += kron(I1, kron(SY_MATRIX_IMAG, I0, "csr"), "csr")
        del I0, I1
    t1 = time()

    logger = logging.getLogger(__name__).getChild("TotalSpinMatrices")
    logger.info("The time spend on calculating spin matrices: %.6fs", t1 - t0)
    return total_sx, total_sy_imag, total_sz


def TotalSpinAverages(spin_num):
    """
    Return a function for calculating the averages of total spin operators.

    The returned function take a ket and calculate the averages of total_sx,
    total_sy, total_sz and total_s2(square of the total spin operator)
    operators.

    Parameters
    ----------
    spin_num : int
        The number of spin in the system.
    """

    total_sx, total_sy_imag, total_sz = TotalSpinMatrices(spin_num=spin_num)
    logger = logging.getLogger(__name__).getChild("TotalSpinAverages")

    def _AveragesCalculator(ket):
        """
        Calculate the averages of total_sx, total_sy, total_sz and total_s2
        over the given `ket`.

        Parameters
        ----------
        ket : np.ndarray
            The state vector.

        Returns
        -------
        total_s2_avg : complex
            The average of the total_s2(square of the total spin operator).
        total_sx_avg : complex
            The average of the total_sx operator.
        total_sy_avg : complex
            The average of the total_sy operator.
        total_sz_avg : complex
            The average of the total_sz operator.
        """

        t0 = time()
        # Normalize the given `ket`
        ket = ket / np.linalg.norm(ket)
        total_sx_dot_ket = total_sx.dot(ket)
        total_sy_imag_dot_ket = total_sy_imag.dot(ket)
        total_sz_dot_ket = total_sz.dot(ket)

        total_sx_avg = np.vdot(ket, total_sx_dot_ket)
        total_sz_avg = np.vdot(ket, total_sz_dot_ket)
        total_sy_avg = 1j * np.vdot(ket, total_sy_imag_dot_ket)
        del ket

        total_s2_avg = np.vdot(total_sy_imag_dot_ket, total_sy_imag_dot_ket)
        del total_sy_imag_dot_ket
        total_s2_avg += np.vdot(total_sx_dot_ket, total_sx_dot_ket)
        del total_sx_dot_ket
        total_s2_avg += np.vdot(total_sz_dot_ket, total_sz_dot_ket)
        del total_sz_dot_ket
        t1 = time()
        logger.info("The time spend on calculating averages: %.6fs", t1 - t0)
        return total_s2_avg, total_sx_avg, total_sy_avg, total_sz_avg

    return _AveragesCalculator


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    spin_num = 16
    shape = (2 ** spin_num, 1)
    calculator = TotalSpinAverages(spin_num)
    for i in range(3):
        t0 = time()
        ket = np.random.random(shape) + np.random.random(shape) * 1j
        t1 = time()
        logging.info("The time spend on preparing ket: %.6fs", t1 - t0)
        s2_avg, sx_avg, sy_avg, sz_avg = calculator(ket)
        logging.info("total_s2 = % .10f %+.10fj", s2_avg.real, s2_avg.imag)
        logging.info("total_sx = % .10f %+.10fj", sx_avg.real, sx_avg.imag)
        logging.info("total_sy = % .10f %+.10fj", sy_avg.real, sy_avg.imag)
        logging.info("total_sz = % .10f %+.10fj", sz_avg.real, sz_avg.imag)
