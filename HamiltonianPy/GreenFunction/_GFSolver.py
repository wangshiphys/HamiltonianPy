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
        A collection of related excited states. The values of
        `excited_states` are np.ndarray with shape (N, ).
        For example, to calculate $G_{AB}(z)$, `excited_states` must contain
        excited_states[B] = $B |GS>$ and
        excited_states[A.dagger()] = $A^{+} |GS>$.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z` parameter in $G_{AB}(z)$.
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
        $omega + 1j * eta$ compose the `z` parameter in $G_{AB}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether $M = z - (HM - GE)$ or $M = z + (HM - GE)$.
        Default: "+".
    structure : ["dict" | "array"], str, optional
        The data structure of the returned Green-Functions.
        Default: "array".

    Returns
    -------
    GFs : dict or np.ndarray
        If `structure="dict"`, the returned `GFs` is a dict,
        GFs[(A, B)] = GF_AB_vs_omegas, where GF_AB_vs_omegas is 1D np.ndarray
        with the same length as `omegas`, correspond to $G_{AB}(z)$ for all
        given `omegas`;
        If `structure="array"`, the returned `GFs` is a 3D np.ndarray with
        shape (len(As), len(Bs), M). The 1st and 2nd dimension correspond to
        different `As` and `Bs`, the 3rd dimension correspond to different
        `omegas`.
    """

    zs = omegas + 1j * eta
    I = np.identity(HM.shape[0])
    GFs_Array = np.empty((len(As), len(Bs), len(zs)), dtype=np.complex128)
    for ZIndex, z in enumerate(zs):
        M = (z - GE) * I + HM if sign == "+" else (z + GE) * I - HM
        for BIndex, B in enumerate(Bs):
            ket = np.linalg.solve(M, excited_states[B])
            for AIndex, A in enumerate(As):
                GFs_Array[AIndex, BIndex, ZIndex] = np.vdot(
                    excited_states[A.dagger()], ket
                )

    if structure == "dict":
        GFs_Dict = dict()
        for AIndex, A in enumerate(As):
            for BIndex, B in enumerate(Bs):
                GFs_Dict[(A, B)] = GFs_Array[AIndex, BIndex]
        return GFs_Dict
    else:
        return GFs_Array


def GFSolverLanczosSingle(
        omega, A, B, GE, projected_matrices, projected_vectors,
        eta=0.01, sign="+"
):
    """
    Calculate the Green-Function term $G_{AB}(z) = <GS| A M^{-1} B |GS>$.

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
        $omega + 1j * eta$ compose the `z` parameter in $G_{AB}(z)$.
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
    B_dot_GS[0] = temp[0]
    A_dagger_dot_GS = projected_vectors[B][A.dagger()]
    M = (z - GE) * I + HM if sign == "+" else (z + GE) * I - HM
    return np.vdot(A_dagger_dot_GS, np.linalg.solve(M, B_dot_GS))


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
        $omega + 1j * eta$ compose the `z` parameter in $G_{AB}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether $M = z - (HM - GE)$ or $M = z + (HM - GE)$.
        Default: "+".
    structure : ["dict" | "array"], str, optional
        The data structure of the returned Green-Functions.
        Default: "array".

    Returns
    -------
    GFs : dict or np.ndarray
        If `structure="dict"`, the returned `GFs` is a dict,
        GFs[(A, B)] = GF_AB_vs_omegas, where GF_AB_vs_omegas is 1D np.ndarray
        with the same length as `omegas`, correspond to $G_{AB}(z)$ for all
        given `omegas`;
        If `structure="array"`, the returned `GFs` is a 3D np.ndarray with
        shape (len(As), len(Bs), M). The 1st and 2nd dimension correspond to
        different `As` and `Bs`, the 3rd dimension correspond to different
        `omegas`.
    """

    zs = omegas + 1j * eta
    GFs_Array = np.empty((len(As), len(Bs), len(zs)), dtype=np.complex128)
    for BIndex, B in enumerate(Bs):
        HM = projected_matrices[B]
        I = np.identity(HM.shape[0])
        temp = projected_vectors[B][B]
        B_dot_GS = np.zeros_like(temp)
        B_dot_GS[0] = temp[0]
        for ZIndex, z in enumerate(zs):
            M = (z - GE) * I + HM if sign == "+" else (z + GE) * I - HM
            ket = np.linalg.solve(M, B_dot_GS)
            for AIndex, A in enumerate(As):
                GFs_Array[AIndex, BIndex, ZIndex] = np.vdot(
                    projected_vectors[B][A.dagger()], ket
                )

    if structure == "dict":
        GFs_Dict = dict()
        for AIndex, A in enumerate(As):
            for BIndex, B in enumerate(Bs):
                GFs_Dict[(A, B)] = GFs_Array[AIndex, BIndex]
        return GFs_Dict
    else:
        return GFs_Array


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
        $omega + 1j * eta$ compose the `z` parameter in $G_{AB}^{r}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether to add or subtract the two parts of the retarded
        Green-Function.
        Note: The meaning of the `sign` parameter in `RGFSolver*` functions is
        different from that in those `GFSolver*` functions.
        Default: "+".

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
        $omega + 1j * eta$ compose the `z` parameter in $G_{AB}^{r}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether to add or subtract the two parts of the retarded
        Green-Function.
        Note: The meaning of the `sign` parameter in `RGFSolver*` functions is
        different from that in those `GFSolver*` functions.
        Default: "+".
    structure : ["dict" | "array"], str, optional
        The data structure of the returned Green-Functions.
        Default: "array".

    Returns
    -------
    RGFs : dict or np.ndarray
        If `structure="dict"`, the returned `RGFs` is a dict,
        RGFs[(A, B)] = RGF_AB_vs_omegas, where RGF_AB_vs_omegas is
        1D np.ndarray with the same length as `omegas`, correspond to
        $G_{AB}^{r}(z)$ for all given `omegas`;
        If `structure="array"`, the returned `RGFs` is a 3D np.ndarray with
        shape (len(As), len(Bs), M). The 1st and 2nd dimension correspond to
        different `As` and `Bs`, the 3rd dimension correspond to different
        `omegas`.
    """

    GFs_AB = GFSolverExactMultiple(
        omegas, As, Bs, GE, HM, excited_states,
        eta=eta, sign="-", structure="array"
    )
    GFs_BA = GFSolverExactMultiple(
        omegas, Bs, As, GE, HM, excited_states,
        eta=eta, sign="+", structure="array"
    )

    if structure == "dict":
        RGFs = dict()
        for i, A in enumerate(As):
            for j, B in enumerate(Bs):
                if sign == "+":
                    RGFs[(A, B)] = GFs_AB[i, j] + GFs_BA[j, i]
                else:
                    RGFs[(A, B)] = GFs_AB[i, j] - GFs_BA[j, i]
    else:
        RGFs = np.empty_like(GFs_AB)
        for i, A in enumerate(As):
            for j, B in enumerate(Bs):
                if sign == "+":
                    RGFs[i, j] = GFs_AB[i, j] + GFs_BA[j, i]
                else:
                    RGFs[i, j] = GFs_AB[i, j] - GFs_BA[j, i]
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
        $omega + 1j * eta$ compose the `z` parameter in $G_{AB}^{r}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether to add or subtract the two parts of the retarded
        Green-Function.
        Note: The meaning of the `sign` parameter in `RGFSolver*` functions is
        different from that in those `GFSolver*` functions.
        Default: "+".

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
    As, Bs : A collection of SpinOperators or AoCs
    GE : float
        Ground state energy of model Hamiltonian.
    projected_matrices, projected_vectors : dict
        A collection of Lanczos projected model Hamiltonian and excited states.
        See also the document of `GFSolverLanczosSingle`.
    eta : float, optional
        The broadening coefficient.
        $omega + 1j * eta$ compose the `z` parameter in $G_{AB}^{r}(z)$.
        Default: 0.01.
    sign : ["+" | "-"], str, optional
        Determining whether to add or subtract the two parts of the retarded
        Green-Function.
        Note: The meaning of the `sign` parameter in `RGFSolver*` functions is
        different from that in those `GFSolver*` functions.
        Default: "+".
    structure : ["dict" | "array"], str, optional
        The data structure of the returned Green-Functions.
        Default: "array".

    Returns
    -------
    RGFs : dict or np.ndarray
        If `structure="dict"`, the returned `RGFs` is a dict,
        RGFs[(A, B)] = RGF_AB_vs_omegas, where RGF_AB_vs_omegas is
        1D np.ndarray with the same length as `omegas`, correspond to
        $G_{AB}^{r}(z)$ for all given `omegas`;
        If `structure="array"`, the returned `RGFs` is a 3D np.ndarray with
        shape (len(As), len(Bs), M). The 1st and 2nd dimension correspond to
        different `As` and `Bs`, the 3rd dimension correspond to different
        `omegas`.
    """

    GFs_AB = GFSolverLanczosMultiple(
        omegas, As, Bs, GE, projected_matrices, projected_vectors,
        eta=eta, sign="-", structure="array",
    )
    GFs_BA = GFSolverLanczosMultiple(
        omegas, Bs, As, GE, projected_matrices, projected_vectors,
        eta=eta, sign="+", structure="array",
    )

    if structure == "dict":
        RGFs = dict()
        for i, A in enumerate(As):
            for j, B in enumerate(Bs):
                if sign == "+":
                    RGFs[(A, B)] = GFs_AB[i, j] + GFs_BA[j, i]
                else:
                    RGFs[(A, B)] = GFs_AB[i, j] - GFs_BA[j, i]
    else:
        RGFs = np.empty_like(GFs_AB)
        for i, A in enumerate(As):
            for j, B in enumerate(Bs):
                if sign == "+":
                    RGFs[i, j] = GFs_AB[i, j] + GFs_BA[j, i]
                else:
                    RGFs[i, j] = GFs_AB[i, j] - GFs_BA[j, i]
    return RGFs
