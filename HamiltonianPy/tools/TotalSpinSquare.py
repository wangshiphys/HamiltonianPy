"""
Calculate the square of the total_spin matrix(S^2)
"""


from pathlib import Path
from time import time

from scipy.sparse import identity, kron, save_npz

import argparse

import numpy as np


__all__ = [
    "TotalSpinSquare",
]


SPIN_MATRICES = {
    "x": np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
    # The real S^y matrix is np.array([[0.0, -0.5j], [0.5j, 0.0]])
    # The following matrix is (S^y / 1j), called it SY
    "y": np.array([[0.0, -0.5], [0.5, 0.0]], dtype=np.float64),
    "z": np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64),
    "p": np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64),
    "m": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
}


# Two methods to calculate the square of the total_spin
def _total_spin_square_core0(spin_num):
    SX_tot = SY_tot = SZ_tot = 0.0
    for index in range(spin_num):
        I0 = identity(1 << index, np.float64, "csr")
        I1 = identity(1 << (spin_num-index-1), np.float64, "csr")
        SX_tot += kron(I1, kron(SPIN_MATRICES["x"], I0, "csr"), "csr")
        SY_tot += kron(I1, kron(SPIN_MATRICES["y"], I0, "csr"), "csr")
        SZ_tot += kron(I1, kron(SPIN_MATRICES["z"], I0, "csr"), "csr")
    S_tot_square = SX_tot.dot(SX_tot)
    del SX_tot
    # Since the matrix used for calculation SY = (S^y / 1j) so
    # (S^y) * (S^y) = - SY * SY
    S_tot_square -= SY_tot.dot(SY_tot)
    del SY_tot
    S_tot_square += SZ_tot.dot(SZ_tot)
    del SZ_tot
    return S_tot_square


def _total_spin_square_core1(spin_num):
    SP, SM, SZ = SPIN_MATRICES["p"], SPIN_MATRICES["m"], SPIN_MATRICES["z"]
    S_tot_square = (3 * spin_num / 4) * identity(
        1 << spin_num, dtype=np.float64, format="csr"
    )

    for i in range(spin_num):
        dim0 = 1 << i
        for j in range(i+1, spin_num):
            Sij = 0.0
            dim1, dim2 = 1 << (j - i - 1), 1 << (spin_num - j - 1)
            for S0, S1 in [(SP, SM), (SZ, SZ)]:
                if dim1 == 1:
                    tmp = kron(S1, S0, format="csr")
                else:
                    I = identity(dim1, dtype=np.float64, format="csr")
                    tmp = kron(S1, kron(I, S0, format="csr"), format="csr")
                if dim0 != 1:
                    tmp = kron(tmp, identity(dim0, np.float64, "csr"), "csr")
                if dim2 != 1:
                    tmp = kron(identity(dim2, np.float64, "csr"), tmp, "csr")
                Sij += tmp
            Sij += Sij.transpose()
            S_tot_square += Sij
    return S_tot_square


def TotalSpinSquare(spin_num, fast=False, save=True, path="."):
    """
    Calculate the total spin square matrix(S^2) of a system

    Parameters
    ----------
    spin_num : int
        The number of spin in the system
    fast : bool, optional
        There are two methods to calculate the S^2
        The first method is faster but requires much more memory. This
        method can be selected by setting `fast=True`
        The second method is slower but requires less memory.
        default: False
    save : bool, optional
        Whether to save the result to a file
        default: True
    path : str, optional
        Where to save the result
        default: The current directory

    Returns
    -------
    res : csr_matrix
        The square of the total_spin matrix(S^2)
    """

    assert isinstance(spin_num, int) and spin_num > 0

    t0 = time()
    if fast:
        S_tot_square = _total_spin_square_core0(spin_num)
    else:
        S_tot_square = _total_spin_square_core1(spin_num)
    t1 = time()
    msg = "The time spend on S_tot^2 with spin_num={0:02} is: {1}s"
    print(msg.format(spin_num, t1 - t0), flush=True)

    if save:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        file_name = "S_tot^2 with spin_num={0:02}.npz".format(spin_num)
        full_name = directory / file_name
        save_npz(full_name, S_tot_square, compressed=False)
        print("Saving result to: {0}".format(full_name.resolve()), flush=True)
    print("=" * 80, flush=True)
    return S_tot_square


def ArgParser():
    """
    Parse the command line arguments
    """

    parser = argparse.ArgumentParser(
        description="Parse command line arguments"
    )

    parser.add_argument(
        "-n", "--num",
        type = int,
        help = "The number of spin in the system"
    )

    parser.add_argument(
        "-f", "--fast",
        action = "store_true",
        help = "Whether to use the faster but more memory consuming method"
    )

    parser.add_argument(
        "-s", "--save",
        action = "store_true",
        help = "Whether to save the result to file system"
    )

    parser.add_argument(
        "-p", "--path",
        type = str,
        default = ".",
        help = "Where to save the result"
    )
    args = parser.parse_args()
    return args.num, args.fast, args.save, args.path


if __name__ == "__main__":
    total_spin, fast, save, path = ArgParser()
    S_tot_square1 = TotalSpinSquare(total_spin, save=save, path=path)
    S_tot_square0 = TotalSpinSquare(total_spin, fast=True, save=save, path=path)
    assert (S_tot_square0 - S_tot_square1).nnz == 0
