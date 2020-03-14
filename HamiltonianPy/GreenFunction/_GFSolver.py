"""
This module provides functions to calculate the following Green-Function term:
    $G_{AB}(z) = <GS| A M^{-1} B |GS>$
where $M = z + (H - GE)$ or $M = z - (H - GE)$.
$H$ is the model Hamiltonian, $|GS>$ the ground state and $GE$ the ground
state energy.

Apart from functions for calculating the Green-Function term, this module also
contains functions for calculating the retarded Green-Function:
    $G_{AB}^{r}(z) = <GS| A M_0^{-1} B |GS> +/- <GS| B M_1^{-1} A |GS>$
where $M_0 = z - (H - GE)$ and $M_1 = z + (H - GE)$.
"""


__all__ = [
    "GFSolverExactSingle",
    "GFSolverExactMultiple",
    "GFSolverLanczosSingle",
    "GFSolverLanczosMultiple",

    "RGFSolverExactSingle",
    "RGFSolverExactMultiple",
    "RGFSolverLanczosSingle",
    "RGFSolverLanczosMultiple",
]


import numpy as np
from numba import jit, prange


def GFSolverExactSingle(
        omega, A, B, GE, HM, excited_states, eta=0.01, sign="+"
):
    """
    Calculate the Green-Function term $G_{AB}(z) = <GS| A M^{-1} B |GS>$.

    Parameters
    ----------
    omega : float
    A, B : Instance of SpinOperator or AoC
    GE : float
        Ground state energy of HM.
    HM : np.ndarray with shape (N, N)
        The model Hamiltonian matrix.
    excited_states : dict
        A collection of related excited states.
        For example, to calculate $G_{AB}(z)$, `excited_states` must contain
        excited_states[B] = $B |GS>$ and
        excited_states[A.dagger()] = $A^{+} |GS>$.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z' parameter in $G_{AB}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether $M = z - (HM - GE)$ or $M = z + (HM - GE)$.
        Default: "+".

    Returns
    -------
    GF_AB : complex
        The corresponding Green-Function.
    """

    z = omega + 1j * eta
    I = np.identity(HM.shape[0])
    M = (z - GE) * I + HM if sign == "+" else (z + GE) * I - HM

    B_dot_GS = excited_states[B]
    A_dagger_dot_GS = excited_states[A.dagger()]
    return np.vdot(A_dagger_dot_GS, np.linalg.solve(M, B_dot_GS))


@jit(nopython=True, cache=True, parallel=True)
def _IterateOverZsExact(zs, GE, HM, bras, kets, sign):
    z_num = zs.shape[0]
    I = np.identity(HM.shape[0])
    res = np.empty((z_num, bras.shape[0], kets.shape[1]), dtype=np.complex128)
    for i in prange(z_num):
        M = (zs[i] - GE) * I + HM if sign == "+" else (zs[i] + GE) * I - HM
        # The `@` operator performs matrix multiplication
        res[i] = bras @ np.linalg.solve(M, kets)
    return res


def GFSolverExactMultiple(
        omegas, As, Bs, GE, HM, excited_states,
        eta=0.01, sign="+", structure="array",
):
    """
    Calculate Green-Function terms $G_{AB}(z)$ for all `As`, `Bs` and `omegas`.

    Parameters
    ----------
    omegas : 1D np.ndarray with shape (M, )
    As, Bs : A collection of SpinOperators or AoCs.
    GE : float
        Ground state energy of HM.
    HM : np.ndarray with shape (N, N)
        The model Hamiltonian matrix.
    excited_states : dict
        A collection of related excited states.
        See also document of `GFSolverExactSingle`.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z' parameter in $G_{AB}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether $M = z - (HM - GE)$ or $M = z + (HM - GE)$.
        Default: "+".
    structure : ["dict" | "array"], str, optional
        The data structure of the returned Green-Functions.
        Default: "array".

    Returns
    GFs : dict or np.ndarray
        If `structure="dict"`, the returned `GFs` is a dict,
        GFs[(A, B)] = GF_AB_vs_omegas, where GF_AB_vs_omegas is 1D np.ndarray
        with the same length as `omegas`, correspond to $G_{AB}(z)$ for all
        given `omegas`;
        If `structure="array"`, the returned `GFs` is a 3D np.ndarray with
        shape (M, len(As), len(Bs)). The first dimension correspond to
        different `omegas`, the second and third dimension correspond to
        different `As` and `Bs`.
    """

    Bs_dot_GS = np.concatenate(
        [excited_states[B] for B in Bs], axis=1
    ).astype(np.complex128)
    As_dagger_dot_GS_dagger = np.concatenate(
        [excited_states[A.dagger()] for A in As], axis=1
    ).T.conj().astype(np.complex128)

    GFs_Array = _IterateOverZsExact(
        omegas + 1j * eta, GE, HM, As_dagger_dot_GS_dagger, Bs_dot_GS, sign
    )
    if structure == "array":
        return GFs_Array
    else:
        GFs_Dict = dict()
        for row_index, A in enumerate(As):
            for col_index, B in enumerate(Bs):
                GFs_Dict[(A, B)] = GFs_Array[:, row_index, col_index]
        return GFs_Dict


def GFSolverLanczosSingle(
        omega, A, B, GE, projected_matrices, projected_vectors,
        eta=0.01, sign="+"
):
    """
    Calculate the Green-Function term $G_{AB}(z)=<GS| A M^{1} B |GS>$.

    Parameters
    ----------
    omega : float
    A, B : Instance of SpinOperator or AoC
    GE : float
        Ground state energy of the model Hamiltonian.
    projected_matrices, projected_vectors : dict
        A collection of Lanczos projected model Hamiltonian and excited states.
        For example, to calculate $G_{AB}(z)$, the Lanczos projection must
        start with $B |GS>$. `projected_matrices` must contain the Lanczos
        projection of the model Hamiltonian: `projected_matrices[B]`;
        `projected_vectors` must contain the Lanczos projection of $B |GS>$
        and $A^{+} |GS>$: `projected_vectors[B][B]` and
        `projected_vectors[B][A.dagger()]`.
        See also `HamiltonianPy.lanczos`.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z' parameter in $G_{AB}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether $M = z - (HM - GE)$ or $M = z + (HM - GE)$.
        Default: "+".

    Returns
    -------
    GF_AB : complex
        The corresponding Green-Function.
    """

    z = omega + 1j * eta
    HM = projected_matrices[B]
    I = np.identity(HM.shape[0])
    temp = projected_vectors[B][B]
    B_dot_GS = np.zeros_like(temp)
    B_dot_GS[0, 0] = temp[0, 0]
    A_dagger_dot_GS = projected_vectors[B][A.dagger()]
    M = (z - GE) * I + HM if sign == "+" else (z + GE) * I - HM
    return np.vdot(A_dagger_dot_GS, np.linalg.solve(M, B_dot_GS))


@jit(nopython=True, cache=True, parallel=True)
def _IterateOverZsLanczos(zs, GE, HM, bra_dagger, ket, sign):
    z_num = zs.shape[0]
    I = np.identity(HM.shape[0])
    res = np.empty(z_num, dtype=np.complex128)
    for i in prange(z_num):
        M = (zs[i] - GE) * I + HM if sign == "+" else (zs[i] + GE) * I - HM
        res[i] = np.vdot(bra_dagger, np.linalg.solve(M, ket))
    return res


def GFSolverLanczosMultiple(
        omegas, As, Bs, GE, projected_matrices, projected_vectors,
        eta=0.01, sign="+", structure="array",
):
    """
    Calculate Green-Function terms $G_{AB}(z)$ for all `As`, `Bs` and `omegas`.

    Parameters
    ----------
    omegas : 1D np.ndarray with shape (M, )
    As, Bs : A collection of SpinOperators or AoCs.
    GE : float
        Ground state energy of the model Hamiltonian.
    projected_matrices, projected_vectors : dict
        A collection of Lanczos projected model Hamiltonian and excited states.
        See also document of `GFSolverLanczosSingle`.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z' parameter in $G_{AB}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether $M = z - (HM - GE)$ or $M = z + (HM - GE)$.
        Default: "+".
    structure : ["dict" | "array"], str, optional
        The data structure of the returned Green-Functions.
        Default: "array".

    Returns
    GFs : dict or np.ndarray.
        If `structure="dict"`, the returned `GFs` is a dict,
        GFs[(A, B)] = GF_AB_vs_omegas, where GF_AB_vs_omegas is 1D np.ndarray
        with the same length as `omegas`, correspond to $G_{AB}(z)$ for all
        given `omegas`;
        If `structure="array"`, the returned `GFs` is a 3D np.ndarray with
        shape (M, len(As), len(Bs)). The first dimension correspond to
        different `omegas`, the second and third dimension correspond to
        different `As` and `Bs`.
    """

    GFs_Dict = {}
    for B in Bs:
        # HM is real matrix
        HM = projected_matrices[B]
        temp = projected_vectors[B][B]
        B_dot_GS = np.zeros(len(temp), dtype=np.complex128)
        B_dot_GS[0] = temp[0, 0]
        for A in As:
            # A_dagger_dot_GS is complex128
            A_dagger_dot_GS = np.ravel(projected_vectors[B][A.dagger()])
            GFs_Dict[(A, B)] = _IterateOverZsLanczos(
                omegas + 1j * eta, GE, HM, A_dagger_dot_GS, B_dot_GS, sign
            )

    if structure == "array":
        num = omegas.shape[0]
        GFs_Array = np.empty((num, len(As), len(Bs)), dtype=np.complex128)
        for row_index, A in enumerate(As):
            for col_index, B in enumerate(Bs):
                GFs_Array[:, row_index, col_index] = GFs_Dict[(A, B)]
        return GFs_Array
    else:
        return GFs_Dict


def RGFSolverExactSingle(
        omega, A, B, GE, HM, excited_states, eta=0.01, sign="+"
):
    """
    Calculate the retarded Green-Function: $G_{AB}^{r}(z)$.

    Parameters
    ----------
    omega : float
    A, B : Instance of SpinOperator or AoC
    GE : float
        Ground state energy of HM.
    HM : np.ndarray with shape (N, N)
        The model Hamiltonian matrix.
    excited_states : dict
        A collection of related excited states.
        See also document of `GFSolverExactSingle`.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z' parameter in $G_{AB}^{r}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether to add or subtract the two parts of the retarded
        Green-Function.
        Note: The meaning of the `sign` parameter in `RGFSolver*` functions is
        different from that in those `GFSolver*` functions.
        Default: "+.

    Returns
    -------
    RGF_AB : complex
        The corresponding retarded Green-Function.
    """

    z = omega + 1j * eta
    I = np.identity(HM.shape[0])
    M1 = (z + GE) * I - HM
    M2 = (z - GE) * I + HM

    A_dot_GS = excited_states[A]
    B_dot_GS = excited_states[B]
    A_dagger_dot_GS = excited_states[A.dagger()]
    B_dagger_dot_GS = excited_states[B.dagger()]
    GF_AB = np.vdot(A_dagger_dot_GS, np.linalg.solve(M1, B_dot_GS))
    GF_BA = np.vdot(B_dagger_dot_GS, np.linalg.solve(M2, A_dot_GS))
    return GF_AB + GF_BA if sign == "+" else GF_AB - GF_BA


def RGFSolverExactMultiple(
        omegas, As, Bs, GE, HM, excited_states,
        eta=0.01, sign="+", structure="array",
):
    """
    Calculate retarded Green-Functions $G_{AB}^{r}(z)$ for
    all `As`, `Bs` and `omegas`.

    Parameters
    ----------
    omegas : 1D np.ndarray with shape (M, )
    As, Bs : A collection of SpinOperators or AoCs
    GE : float
        Ground state energy of HM.
    HM : np.ndarray with shape (N, N)
        The model Hamiltonian matrix.
    excited_states : dict
        A collection of related excited states.
        See also document of `GFSolverExactSingle`.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z' parameter in $G_{AB}^{r}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether to add or subtract the two parts of the retarded
        Green-Function.
        Note: The meaning of the `sign` parameter in `RGFSolver*` functions is
        different from that in those `GFSolver*` functions.
        Default: "+.
    structure : ["dict" | "array"], str, optional
        The data structure of the returned Green-Functions.
        Default: "array".

    Returns
    -------
    RGFs : dict or np.ndarray.
        If `structure="dict"`, the returned `RGFs` is a dict,
        RGFs[(A, B)] = RGF_AB_vs_omegas, where RGF_AB_vs_omegas is
        1D np.ndarray with the same length as `omegas`, correspond to
        $G_{AB}^{r}(z)$ for all given `omegas`;
        If `structure="array"`, the returned `RGFs` is a 3D np.ndarray with
        shape (M, len(As), len(Bs)). The first dimension correspond to
        different `omegas`, the second and third dimension correspond to
        different `As` and `Bs`.
    """

    GFs_AB = GFSolverExactMultiple(
        omegas, As, Bs, GE, HM, excited_states,
        eta=eta, sign="-", structure="array"
    )
    GFs_BA = GFSolverExactMultiple(
        omegas, Bs, As, GE, HM, excited_states,
        eta=eta, sign="+", structure="array"
    )

    if structure == "array":
        RGFs = np.empty_like(GFs_AB)
        for i, A in enumerate(As):
            for j, B in enumerate(Bs):
                if sign == "+":
                    RGFs[:, i, j] = GFs_AB[:, i, j] + GFs_BA[:, j, i]
                else:
                    RGFs[:, i, j] = GFs_AB[:, i, j] - GFs_BA[:, j, i]
    else:
        RGFs = dict()
        for i, A in enumerate(As):
            for j, B in enumerate(Bs):
                if sign == "+":
                    RGFs[(A, B)] = GFs_AB[:, i, j] + GFs_BA[:, j, i]
                else:
                    RGFs[(A, B)] = GFs_AB[:, i, j] - GFs_BA[:, j, i]
    return RGFs


def RGFSolverLanczosSingle(
        omega, A, B, GE, projected_matrices, projected_vectors,
        eta=0.01, sign="+"
):
    """
    Calculate the retarded Green-Function $G_{AB}^{r}(z)$.

    Parameters
    ----------
    omega : float
    A, B : Instance of SpinOperator or AoC
    GE : float
        Ground state energy of the model Hamiltonian.
    projected_matrices, projected_vectors : dict
        A collection of Lanczos projected model Hamiltonian and excited states.
        See also the document of `GFSolverLanczosSingle`.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z' parameter in $G_{AB}^{r}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether to add or subtract the two parts of the retarded
        Green-Function.
        Note: The meaning of the `sign` parameter in `RGFSolver*` functions is
        different from that in those `GFSolver*` functions.
        Default: "+.

    Returns
    -------
    RGF_AB : complex
        The corresponding retarded Green-Function.
    """

    GF_AB = GFSolverLanczosSingle(
        omega, A, B, GE, projected_matrices, projected_vectors,
        eta=eta, sign="-"
    )
    GF_BA = GFSolverLanczosSingle(
        omega, B, A, GE, projected_matrices, projected_vectors,
        eta=eta, sign="+"
    )
    return GF_AB + GF_BA if sign == "+" else GF_AB - GF_BA


def RGFSolverLanczosMultiple(
        omegas, As, Bs, GE, projected_matrices, projected_vectors,
        eta=0.01, sign="+", structure="array",
):
    """
    Calculate retarded Green-Functions $G_{AB}^{r}{z}$ for
    all `As`, `Bs` and `omegas`.

    Parameters
    ----------
    omegas : 1D np.ndarray with shape (M, )
    As, Bs : A collection of SpinOperators or AoCs.
    GE : float
        Ground state energy of model Hamiltonian.
    projected_matrices, projected_vectors : dict
        A collection of Lanczos projected model Hamiltonian and excited states.
        See also the document of `GFSolverLanczosSingle`.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z' parameter in $G_{AB}^{r}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether to add or subtract the two parts of the retarded
        Green-Function.
        Note: The meaning of the `sign` parameter in `RGFSolver*` functions is
        different from that in those `GFSolver*` functions.
        Default: "+.
    structure : ["dict" | "array"], str, optional
        The data structure of the returned Green-Functions.
        Default: "array".

    Returns
    RGFs : dict or np.ndarray.
        If `structure="dict"`, the returned `RGFs` is a dict,
        RGFs[(A, B)] = RGF_AB_vs_omegas, where RGF_AB_vs_omegas is
        1D np.ndarray with the same length as `omegas`, correspond to
        $G_{AB}^{r}(z)$ for all given `omegas`;
        If `structure="array"`, the returned `RGFs` is a 3D np.ndarray with
        shape (M, len(As), len(Bs)). The first dimension correspond to
        different `omegas`, the second and third dimension correspond to
        different `As` and `Bs`.
    """

    GFs_AB = GFSolverLanczosMultiple(
        omegas, As, Bs, GE, projected_matrices, projected_vectors,
        eta=eta, sign="-", structure="dict",
    )
    GFs_BA = GFSolverLanczosMultiple(
        omegas, Bs, As, GE, projected_matrices, projected_vectors,
        eta=eta, sign="+", structure="dict",
    )

    if structure == "array":
        RGFs = np.empty(
            (omegas.shape[0], len(As), len(Bs)), dtype=np.complex128
        )
        for i, A in enumerate(As):
            for j, B in enumerate(Bs):
                if sign == "+":
                    RGFs[:, i, j] = GFs_AB[(A, B)] + GFs_BA[(B, A)]
                else:
                    RGFs[:, i, j] = GFs_AB[(A, B)] - GFs_BA[(B, A)]
    else:
        RGFs = dict()
        for A in As:
            for B in Bs:
                if sign == "+":
                    RGFs[(A, B)] = GFs_AB[(A, B)] + GFs_BA[(B, A)]
                else:
                    RGFs[(A, B)] = GFs_AB[(A, B)] - GFs_BA[(B, A)]
    return RGFs
