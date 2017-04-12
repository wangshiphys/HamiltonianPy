"""
This module define some useful constant!
"""

import numpy as np

#Constant that identify the creation and annihilation operator respectively.
CREATION = 1
ANNIHILATION = 0

#Absolute value less than this quantity is viewed as 0.
VIEW_AS_ZERO = 1E-8

#The factor that result from exchanging two fermions or bosons.
SWAP_FACTOR_F = -1
SWAP_FACTOR_B = 1

#Data type supportted by numpy.
FLOAT_TYPE = (np.float_, np.float16, np.float32, np.float64)
INT_TYPE = (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64)
COMPLEX_TYPE = (np.complex_, np.complex64, np.complex128)

#Constant that identify the spin up and spin down state.
SPIN_UP = 1
SPIN_DOWN = 0

#The number of digits after the decimal point should be consider when 
#processing float point numbers.
NDIGITS = 4

#All possible type of the spin operator.
SPIN_OTYPE = ('x', 'y', 'z', 'p', 'm')

#The 2 * 2 identity matrix
IDENTITY = np.array([[1, 0], [0, 1]], dtype=np.int64)
IDENTITY.setflags(write=False)

#The Pauli matrix.
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.int64)
SIGMA_X.setflags(write=False)
SIGMA_Y = np.array([[0, -1], [1, 0]], dtype=np.int64) * 1j
SIGMA_Y.setflags(write=False)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.int64)
SIGMA_Z.setflags(write=False)
SIGMA_P = np.array([[0, 2], [0, 0]], dtype=np.int64)
SIGMA_P.setflags(write=False)
SIGMA_M = np.array([[0, 0], [2, 0]], dtype=np.int64)
SIGMA_M.setflags(write=False)
SIGMA_MATRIX = {'x': SIGMA_X, 'y': SIGMA_Y, 'z': SIGMA_Z, 
                'p': SIGMA_P, 'm':SIGMA_M}

#The S matrix.
S_X = 0.5 * SIGMA_X 
S_X.setflags(write=False)
S_Y = 0.5 * SIGMA_Y
S_Y.setflags(write=False)
S_Z = 0.5 * SIGMA_Z
S_Z.setflags(write=False)
S_P = 0.5 * SIGMA_P
S_P.setflags(write=False)
S_M = 0.5 * SIGMA_M
S_M.setflags(write=False)
S_MATRIX = {'x': S_X, 'y': S_Y, 'z': S_Z, 'p': S_P, 'm':S_M}

SPIN_MATRIX = {"sigma": SIGMA_MATRIX, "s": S_MATRIX}
