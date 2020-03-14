"""
GreenFunction
=============

Provide functions for calculating the Green-Function.

Available Functions
-------------------
GFSolverExactSingle
    Calculate the Green-Function term $G_{AB}(z) = <GS| A M^{-1} B |GS>$.
GFSolverExactMultiple
    Calculate Green-Function terms $G_{AB}(z)$ for all `As`, `Bs` and `omegas`.
GFSolverLanczosSingle
    Calculate the Green-Function term $G_{AB}(z)=<GS| A M^{1} B |GS>$.
GFSolverLanczosMultiple
    Calculate Green-Function terms $G_{AB}(z)$ for all `As`, `Bs` and `omegas`.
RGFSolverExactSingle
    Calculate the retarded Green-Function: $G_{AB}^{r}(z)$.
RGFSolverExactMultiple
    Calculate retarded Green-Functions $G_{AB}^{r}(z)$ for all `As`, `Bs` and `omegas`.
RGFSolverLanczosSingle
    Calculate the retarded Green-Function $G_{AB}^{r}(z)$.
RGFSolverLanczosMultiple
    Calculate retarded Green-Functions $G_{AB}^{r}{z}$ for all `As`, `Bs` and `omegas`.
"""


from ._GFSolver import *

__all__ = []
__all__ += _GFSolver.__all__
