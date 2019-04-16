"""
HamiltonianPy
=============

Provides
  1. Unified description of common lattice with translation symmetry
  2. Bases of the Hilbert space in occupation number representation
  3. Building block for constructing a model Hamiltonian
  4. Lanczos algorithm for calculating the ground state energy and single
  particle Green's function

Available modules
-----------------
bond
    Bond class that describes the bond connecting two points
constant
    Some useful constant
hilbertspace
    Description of Hilbert space in the occupation-number representation
indextable
    Mapping hashable objects to integers in a continuous range
lanczos
    Implementation of the Lanczos Algorithm
lattice
    Description of common lattice with translation symmetry
termofH
    Components for constructing a model Hamiltonian
"""


from .constant import ANNIHILATION, CREATION, SPIN_DOWN, SPIN_UP
from .greenfunction import ClusterGFSolver
from .hilbertspace import base_vectors
from .indextable import IndexTable
from .lattice import KPath, Lattice, lattice_generator, special_cluster
from .termofH import SiteID, StateID
from .termofH import AoC, ParticleTerm
from .termofH import SpinOperator, SpinInteraction
from .version import version as __version__


__all__ = [
    "__version__",
    "ANNIHILATION", "CREATION",
    "SPIN_DOWN", "SPIN_UP",
    "ClusterGFSolver",
    "base_vectors",
    "IndexTable",
    "KPath",
    "Lattice", "lattice_generator", "special_cluster",
    "SiteID", "StateID",
    "AoC", "ParticleTerm",
    "SpinOperator", "SpinInteraction",
]