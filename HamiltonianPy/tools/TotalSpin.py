"""
Calculate the total spin matrices of a system
"""


from scipy.sparse import identity, kron

import numpy as np


__all__ = [
    "TotalSpin",
]


SX_MATRIX = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
# The real SY matrix is np.array([[0.0, -0.5j], [0.5j, 0.0]])
# The following  matrix is the imaginary part
SY_MATRIX_IMAG = np.array([[0.0, -0.5], [0.5, 0.0]], dtype=np.float64)
SZ_MATRIX = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64)


def TotalSpin(spin_num):
    """
    Calculate the total spin matrices of a system with N spins

    Parameters
    ----------
    spin_num : int
        The number of spin in the system

    Returns
    -------
    total_sx : csr_matrix
        The total S^x matrix
    total_sy_imag : csr_matrix
        The imaginary part of the total S^y matrix
        Since the S^y matrix is pure imaginary, so total_sy = 1j * total_sy_imag
    total_sz : csr_matrix
        The total S^z matrix
    """

    assert isinstance(spin_num, int) and spin_num > 0

    total_sx = total_sy_imag = total_sz = 0.0
    for index in range(spin_num):
        I0 = identity(1 << index, np.float64, "csr")
        I1 = identity(1 << (spin_num - index - 1), np.float64, "csr")
        total_sx += kron(I1, kron(SX_MATRIX, I0, "csr"), "csr")
        total_sz += kron(I1, kron(SZ_MATRIX, I0, "csr"), "csr")
        total_sy_imag += kron(I1, kron(SY_MATRIX_IMAG, I0, "csr"), "csr")
        del I0, I1
    return total_sx, total_sy_imag, total_sz


if __name__ == "__main__":
    from time import time

    for total_spin in range(1, 25):
        t0 = time()
        TotalSpin(total_spin)
        t1 = time()
        print(
            "The time spend for system with {0} spins: {1}s".format(
                total_spin, t1 - t0
            )
        )