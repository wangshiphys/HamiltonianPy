"""
Some useful constant
"""


from enum import IntEnum

import numpy as np


class ParticleOperatorType(IntEnum):
    """
    Symbolic names for creation and annihilation operators
    """

    ANNIHILATION = 0
    CREATION = 1


class SpinState(IntEnum):
    """
    Symbolic names for spin states
    """

    DOWN = 0
    UP = 1


# Constant that identify the creation and annihilation operator respectively
CREATION = 1
ANNIHILATION = 0

# Absolute value less than this quantity is viewed as 0
VIEW_AS_ZERO = 1E-8

# The factor that result from exchanging two fermions or bosons
SWAP_FACTOR_F = -1
SWAP_FACTOR_B = 1

# Constant that identify the spin up and spin down state
SPIN_UP = 1
SPIN_DOWN = 0

# All possible type of the spin operator
SPIN_OTYPE = ("x", "y", "z", "p", "m")

# The Pauli matrix
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
SIGMA_X.setflags(write=False)

SIGMA_Y = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64) * 1j
SIGMA_Y.setflags(write=False)

SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
SIGMA_Z.setflags(write=False)

SIGMA_P = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
SIGMA_P.setflags(write=False)

SIGMA_M = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
SIGMA_M.setflags(write=False)

SIGMA_MATRICES = {
    "x": SIGMA_X,
    "y": SIGMA_Y,
    "z": SIGMA_Z,
    "p": SIGMA_P,
    "m": SIGMA_M,
}

# The spin matrices
S_X = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
S_X.setflags(write=False)

S_Y = np.array([[0.0, -0.5], [0.5, 0.0]], dtype=np.float64) * 1j
S_Y.setflags(write=False)

S_Z = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64)
S_Z.setflags(write=False)

S_P = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
S_P.setflags(write=False)

S_M = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
S_M.setflags(write=False)

SPIN_MATRICES = {
    "x": S_X,
    "y": S_Y,
    "z": S_Z,
    "p": S_P,
    "m": S_M,
}

