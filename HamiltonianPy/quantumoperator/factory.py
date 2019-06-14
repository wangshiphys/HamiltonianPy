"""
This module provides functions that generate commonly used Hamiltonian terms.
"""


__all__ = [
    "CPFactory",
    "HoppingFactory",
    "PairingFactory",
    "HubbardFactory",
    "CoulombFactory",
    "HeisenbergFactory",
    "IsingFactory",
    "TwoSpinTermFactory",
]


from HamiltonianPy.quantumoperator.constant import ANNIHILATION, CREATION, \
    SPIN_DOWN, SPIN_UP
from HamiltonianPy.quantumoperator.particlesystem import AoC, ParticleTerm
from HamiltonianPy.quantumoperator.spinsystem import *


def CPFactory(site, *, spin=0, orbit=0, coeff=1.0):
    """
    Generate chemical potential term: '$\\mu c_i^{\\dagger} c_i$'.

    Parameters
    ----------
    site : list, tuple or 1D np.ndarray
        The coordinates of the localized single-particle state.
        The `site` parameter should be 1D array with length 1,2 or 3.
    spin : int, optional, keyword-only
        The spin index of the single-particle state.
        Default: 0.
    orbit : int, optional, keyword-only
        The orbit index of the single-particle state.
        Default: 0.
    coeff : int or float, optional, keyword-only
        The coefficient of this term.
        Default: 1.0.

    Returns
    -------
    term : ParticleTerm
        The corresponding chemical potential term.
    """

    c = AoC(CREATION, site=site, spin=spin, orbit=orbit)
    a = AoC(ANNIHILATION, site=site, spin=spin, orbit=orbit)
    return ParticleTerm((c, a), coeff=coeff, classification="number")


def HoppingFactory(
        site0, site1, *, spin0=0, spin1=None, orbit0=0, orbit1=None, coeff=1.0
):
    """
    Generate hopping term: '$t c_i^{\\dagger} c_j$'.

    These parameters suffixed with '0' are for the creation operator and '1'
    for annihilation operator.

    Parameters
    ----------
    site0, site1 : list, tuple or 1D np.ndarray
        The coordinates of the localized single-particle state.
        `site0` and `site1` should be 1D array with length 1, 2 or 3.
    spin0, spin1 : int, optional, keyword-only
        The spin index of the single-particle state.
        The default value for `spin0` is 0;
        The default value for `spin1` is None, which implies that `spin1`
        takes the same value as `spin0`.
    orbit0, orbit1 : int, optional, keyword-only
        The orbit index of the single-particle state.
        The default value for `orbit0` is 0;
        The default value for `orbit1` is None, which implies that `orbit1`
        takes the same value as `orbit0`.
    coeff : int, float or complex, optional, keyword-only
        The coefficient of this term.
        Default: 1.0.

    Returns
    -------
    term : ParticleTerm
        The corresponding hopping term.
    """

    if spin1 is None:
        spin1 = spin0
    if orbit1 is None:
        orbit1 = orbit0

    c = AoC(CREATION, site=site0, spin=spin0, orbit=orbit0)
    a = AoC(ANNIHILATION, site=site1, spin=spin1, orbit=orbit1)
    classification = "hopping" if c.state != a.state else "number"
    return ParticleTerm((c, a), coeff=coeff, classification=classification)


def PairingFactory(
        site0, site1, *, spin0=0, spin1=0, orbit0=0, orbit1=0,
        coeff=1.0, which="h"
):
    """
    Generate pairing term: '$p c_i^{\\dagger} c_j^{\\dagger}$' or '$p c_i c_j$'.

    These parameters suffixed with '0' are for the 1st operator and '1' for
    2nd operator.

    Parameters
    ----------
    site0, site1 : list, tuple or 1D np.ndarray
        The coordinates of the localized single-particle state.
        `site0` and `site1` should be 1D array with length 1, 2 or 3.
    spin0, spin1 : int, optional, keyword-only
        The spin index of the single-particle state.
        Default: 0.
    orbit0, orbit1 : int, optional, keyword-only
        The orbit index of the single-particle state.
        Default: 0.
    coeff : int, float or complex, optional, keyword-only
        The coefficient of this term.
        Default: 1.0.
    which : str, optional, keyword-only
        Determine whether to generate a particle- or hole-pairing term.
        Valid values:
            ["h" | "hole"] for hole-pairing;
            ["p" | "particle"] for particle-pairing.
        Default: "h".

    Returns
    -------
    term : ParticleTerm
        The corresponding pairing term.
    """

    assert which in ("h", "hole", "p", "particle")

    otype = ANNIHILATION if which in ("h", "hole") else CREATION
    aoc0 = AoC(otype, site=site0, spin=spin0, orbit=orbit0)
    aoc1 = AoC(otype, site=site1, spin=spin1, orbit=orbit1)
    return ParticleTerm((aoc0, aoc1), coeff=coeff)


def HubbardFactory(site, *, orbit=0, coeff=1.0):
    """
    Generate Hubbard term: '$U n_{i\\uparrow} n_{i\\downarrow}$'.

    This function is valid only for spin-1/2 system.

    Parameters
    ----------
    site : list, tuple or 1D np.ndarray
        The coordinates of the localized single-particle state.
        `site` should be 1D array with length 1,2 or 3.
    orbit : int, optional, keyword-only
        The orbit index of the single-particle state.
        Default: 0.
    coeff : int or float, optional, keyword-only
        The coefficient of this term.
        Default: 1.0.

    Returns
    -------
    term : ParticleTerm
        The corresponding Hubbard term.
    """

    c_up = AoC(CREATION, site=site, spin=SPIN_UP, orbit=orbit)
    c_down = AoC(CREATION, site=site, spin=SPIN_DOWN, orbit=orbit)
    a_up = AoC(ANNIHILATION, site=site, spin=SPIN_UP, orbit=orbit)
    a_down = AoC(ANNIHILATION, site=site, spin=SPIN_DOWN, orbit=orbit)
    return ParticleTerm(
        (c_up, a_up, c_down, a_down), coeff=coeff, classification="Coulomb"
    )


def CoulombFactory(
        site0, site1, *, spin0=0, spin1=0, orbit0=0, orbit1=0, coeff=1.0
):
    """
    Generate Coulomb interaction term: '$U n_i n_j$'.

    These parameters suffixed with '0' are for the 1st operator and '1' for
    2nd operator.

    Parameters
    ----------
    site0, site1 : list, tuple or 1D np.ndarray
        The coordinates of the localized single-particle state.
        `site0` and `site1` should be 1D array with length 1, 2 or 3.
    spin0, spin1 : int, optional, keyword-only
        The spin index of the single-particle state.
        Default: 0.
    orbit0, orbit1 : int, optional, keyword-only
        The orbit index of the single-particle state.
        Default: 0.
    coeff : int or float, optional, keyword-only
        The coefficient of this term.
        Default: 1.0.

    Returns
    -------
    term : ParticleTerm
        The corresponding Coulomb interaction term.
    """

    c0 = AoC(CREATION, site=site0, spin=spin0, orbit=orbit0)
    a0 = AoC(ANNIHILATION, site=site0, spin=spin0, orbit=orbit0)
    c1 = AoC(CREATION, site=site1, spin=spin1, orbit=orbit1)
    a1 = AoC(ANNIHILATION, site=site1, spin=spin1, orbit=orbit1)
    return ParticleTerm((c0, a0, c1, a1), coeff=coeff, classification="Coulomb")


def HeisenbergFactory(site0, site1, *, coeff=1.0):
    """
    Generate Heisenberg interaction term: '$J S_i S_j$'.

    Parameters
    ----------
    site0, site1 : list, tuple or 1D np.ndarray
        The coordinates of the lattice site on which the spin operator is
        defined. `site0` and `site1` should be 1D array with length 1,
        2 or 3. `site0` for the first spin operator and `site1` for the
        second spin operator.
    coeff : int or float, optional, keyword-only
        The coefficient of this term.
        Default: 1.0.

    Returns
    -------
    terms : 3-tuple
        terms[0] is the '$J   S_i^z S_j^z$' term;
        terms[1] is the '$J/2 S_i^+ S_j^-$' term;
        terms[2] is the '$J/2 S_i^- S_j^+$' term.
    """

    sz0 = SpinOperator(otype="z", site=site0)
    sp0 = SpinOperator(otype="p", site=site0)
    sm0 = SpinOperator(otype="m", site=site0)
    sz1 = SpinOperator(otype="z", site=site1)
    sp1 = SpinOperator(otype="p", site=site1)
    sm1 = SpinOperator(otype="m", site=site1)

    return (
        SpinInteraction((sz0, sz1), coeff=coeff),
        SpinInteraction((sp0, sm1), coeff=coeff/2),
        SpinInteraction((sm0, sp1), coeff=coeff/2),
    )


def IsingFactory(site0, site1, alpha, *, coeff=1.0):
    """
    Generate Ising type spin interaction term:
        '$J S_i^{\\alpha} S_j^{\\alpha}$'

    Parameters
    ----------
    site0, site1 : list, tuple or 1D np.ndarray
        The coordinates of the lattice site on which the spin operator is
        defined. `site0` and `site1` should be 1D array with length 1,
        2 or 3. `site0` for the first spin operator and `site1` for the
        second spin operator.
    alpha : {"x", "y" or "z"}
        Which type of spin operator is involved.
    coeff : int or float, optional, keyword-only
        The coefficient of this term.
        Default: 1.0.

    Returns
    -------
    term: SpinInteraction
        The corresponding spin interaction term.
    """

    assert alpha in ("x", "y", "z")

    s0_alpha = SpinOperator(otype=alpha, site=site0)
    s1_alpha = SpinOperator(otype=alpha, site=site1)
    return SpinInteraction((s0_alpha, s1_alpha), coeff=coeff)


def TwoSpinTermFactory(site0, site1, alpha0, alpha1, *, coeff=1.0):
    """
    Generate general two spin interaction term:
        '$J S_i^{\\alpha} S_j^{\\beta}$'

    Parameters
    ----------
    site0, site1 : list, tuple or 1D np.ndarray
        The coordinates of the lattice site on which the spin operator is
        defined. `site0` and `site1` should be 1D array with length 1,
        2 or 3. `site0` for the first spin operator and `site1` for the
        second spin operator.
    alpha0, alpha1 : {"x", "y" or "z"}
        Which type of spin operator is involved.
        `alpha0` for the first and `alpha1` for the second spin operator.
    coeff : int or float, optional, keyword-only
        The coefficient of this term.
        Default: 1.0.

    Returns
    -------
    term: SpinInteraction
        The corresponding spin interaction term.
    """

    assert alpha0 in ("x", "y", "z")
    assert alpha1 in ("x", "y", "z")

    s0_alpha = SpinOperator(otype=alpha0, site=site0)
    s1_alpha = SpinOperator(otype=alpha1, site=site1)
    return SpinInteraction((s0_alpha, s1_alpha), coeff=coeff)
