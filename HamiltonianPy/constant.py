"""Some useful constant
"""

import numpy as np

#Constant that identify the creation and annihilation operator respectively
CREATION = 1
ANNIHILATION = 0

#Absolute value less than this quantity is viewed as 0
VIEW_AS_ZERO = 1E-8

#The factor that result from exchanging two fermions or bosons
SWAP_FACTOR_F = -1
SWAP_FACTOR_B = 1

#Constant that identify the spin up and spin down state
SPIN_UP = 1
SPIN_DOWN = 0

#All possible type of the spin operator.
SPIN_OTYPE = ('x', 'y', 'z', 'p', 'm')

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
SIGMA_MATRIX = {
        'x': SIGMA_X, 'y': SIGMA_Y, 'z': SIGMA_Z, 'p': SIGMA_P, 'm':SIGMA_M}
