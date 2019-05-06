"""
QuantumOperator
===============

Components for constructing second-quantized model Hamiltonian

Available Classes
-----------------
SiteID
    Description of lattice site
StateID
    Description of single-particle-state
AoC
    Description of annihilation and creation operator
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
    Generate chemical potential term
HoppingFactory
    Generate hopping term
PairingFactory
    Generate pairing term
HubbardFactory
    Generate Hubbard term
CoulombFactory
    Generate Coulomb interaction term
HeisenbergFactory
    Generate Heisenberg interaction term
IsingFactory
    Generate Ising type spin interaction term
TwoSpinTermFactory
    Generate general two spin interaction term
"""
