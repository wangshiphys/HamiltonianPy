"""
QuantumOperator
===============

Components for constructing model Hamiltonian

Available Classes
-----------------
SiteID
    Description of lattice site
StateID
    Description of single-particle-state
AoC
    Description of annihilation/creation operator
NumberOperator
    Description of particle-number operator
SpinOperator
    Description of quantum spin operator
SpinInteraction
    Description of spin interaction term
ParticleTerm
    Description of any term composed of creation and/or annihilation operators

Available Functions
-------------------
CPFactory
    Generate chemical potential term: '$\\mu c_i^{\\dagger} c_i$'
HoppingFactory
    Generate hopping term: '$t c_i^{\\dagger} c_j$'
PairingFactory
    Generate pairing term: '$p c_i^{\\dagger} c_j^{\\dagger}$' or '$p c_i c_j$'
HubbardFactory
    Generate Hubbard term: '$U n_{i\\uparrow} n_{i\\downarrow}$'
CoulombFactory
    Generate Coulomb interaction term: '$U n_i n_j$'
HeisenbergFactory
    Generate Heisenberg interaction term: '$J S_i S_j$'
IsingFactory
    Generate Ising type spin interaction term: '$J S_i^{\\alpha} S_j^{\\alpha}$'
TwoSpinTermFactory
    Generate general two spin interaction term: '$J S_i^{\\alpha} S_j^{\\beta}$'

set_float_point_precision
    Set the float-point precision for processing coordinate of lattice site

Available Constants
-------------------
ANNIHILATION, CREATION
    Constant that identify the creation and annihilation operator respectively
SPIN_DOWN, SPIN_UP
    Constant that identify the spin-up and spin-down state respectively
SPIN_OTYPES
    All possible types of the spin operator
SPIN_X, SPIN_Y, SPIN_Z, SPIN_P, SPIN_M, SPIN_MATRICES
    Spin matrices
SIGMA_X, SIGMA_Y, SIGMA_Z, SIGMA_P, SIGMA_M, SIGMA_MATRICES
    Pauli/Sigma matrices
NUMERIC_TYPES_INT
    All recognizable integer type
NUMERIC_TYPES_FLOAT
    All recognizable float-point type
NUMERIC_TYPES_COMPLEX
    All recognizable complex type
NUMERIC_TYPES_REAL
    All recognizable real type
NUMERIC_TYPES_GENERAL
    All recognizable numeric type
"""


from .constant import *
from .quantumstate import *
from .particlesystem import *
from .spinsystem import *
from .factory import *


__all__ = []
__all__ += constant.__all__
__all__ += quantumstate.__all__
__all__ += particlesystem.__all__
__all__ += spinsystem.__all__
__all__ += factory.__all__
