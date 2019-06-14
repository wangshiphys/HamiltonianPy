"""
Some commonly used constant in this project.
"""


__all__ = [
    "ANNIHILATION", "CREATION",
    "SPIN_DOWN", "SPIN_UP", "SPIN_OTYPES",
    "SPIN_X", "SPIN_Y", "SPIN_Z", "SPIN_P", "SPIN_M", "SPIN_MATRICES",
    "SIGMA_X", "SIGMA_Y", "SIGMA_Z", "SIGMA_P", "SIGMA_M", "SIGMA_MATRICES",
    "NUMERIC_TYPES_INT", "NUMERIC_TYPES_FLOAT", "NUMERIC_TYPES_REAL",
    "NUMERIC_TYPES_COMPLEX", "NUMERIC_TYPES_GENERAL",
]


import numpy as np

# Constant that identify the creation and annihilation operator respectively
CREATION = 1
ANNIHILATION = 0

# Constant that identify the spin-up and spin-down state respectively
SPIN_UP = 1
SPIN_DOWN = 0

# All possible types of the spin operator
SPIN_OTYPES = ("x", "y", "z", "p", "m")

# Classification of numeric types
NUMERIC_TYPES_INT = (int, np.integer)
NUMERIC_TYPES_FLOAT = (float, np.floating)
NUMERIC_TYPES_COMPLEX = (complex, np.complexfloating)
NUMERIC_TYPES_REAL = (int, float, np.integer, np.floating)
NUMERIC_TYPES_GENERAL = (int, float, complex, np.number)

# Pauli matrices
SIGMA_X = np.array([[0.0,  1.0], [1.0,  0.0]], dtype=np.float64)
SIGMA_Y = np.array([[0.0, -1.0], [1.0,  0.0]], dtype=np.float64) * 1j
SIGMA_Z = np.array([[1.0,  0.0], [0.0, -1.0]], dtype=np.float64)
SIGMA_P = np.array([[0.0,  2.0], [0.0,  0.0]], dtype=np.float64)
SIGMA_M = np.array([[0.0,  0.0], [2.0,  0.0]], dtype=np.float64)
SIGMA_MATRICES = {
    "x": SIGMA_X,
    "y": SIGMA_Y,
    "z": SIGMA_Z,
    "p": SIGMA_P,
    "m": SIGMA_M,
}

# Spin matrices
SPIN_X = np.array([[0.0,  0.5], [0.5,  0.0]], dtype=np.float64)
SPIN_Y = np.array([[0.0, -0.5], [0.5,  0.0]], dtype=np.float64) * 1j
SPIN_Z = np.array([[0.5,  0.0], [0.0, -0.5]], dtype=np.float64)
SPIN_P = np.array([[0.0,  1.0], [0.0,  0.0]], dtype=np.float64)
SPIN_M = np.array([[0.0,  0.0], [1.0,  0.0]], dtype=np.float64)
SPIN_MATRICES = {
    "x": SPIN_X,
    "y": SPIN_Y,
    "z": SPIN_Z,
    "p": SPIN_P,
    "m": SPIN_M,
}
