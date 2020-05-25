"""
Implementation of the Lanczos algorithm.

The basic idea behind the Lanczos algorithm is to build a Krylov space
from an Hermitian matrix M (usually large sparse matrix) and a starting
vector v0. Starting from v0, by iterated application of M (v0 is assumed
to be normalized):
    v1' = M.dot(v0),
    orthogonalize v1' with v0 we get v1'', normalize v1'' we get v1;
    v2' = M.dot(v1),
    orthogonalize v2' with v0 and v1 we get v2'', normalize v2'' we get v2;
    v3' = M.dot(v2),
    orthogonalize v3' with v0, ..., v2 we get v3'', normalize v3'' we get v3;
    ......
    v_n' = M.dot(v_{n-1}),
    orthogonalize v_n' with v0, ..., v_{n-1} we get v_n'',
    normalize v_n'' we get v_n;
    ......
The Krylov space is spanned by the orthonormalized vectors:
    {v0, v1, ..., v_n, ...}
"""


__all__ = ["set_threshold", "KrylovRepresentation", "MultiKrylov"]


import logging
from time import time

import numpy as np

# |Values| <= `_VIEW_AS_ZERO` will be treated as zero in this module
_VIEW_AS_ZERO = 1E-4

logging.getLogger(__name__).addHandler(logging.NullHandler())


def set_threshold(threshold=1E-4):
    """
    Set the threshold for viewing a value as zero.

    If |x| <= `threshold`, then `x` is viewed as zero. If you want to change
    the default threshold, you must call this function before using any other
    functions defined in this module.

    Parameters
    ----------
    threshold : float, optional
        The threshold value.
        Default: 1E-4.
    """

    assert isinstance(threshold, float) and threshold >= 0
    global _VIEW_AS_ZERO
    _VIEW_AS_ZERO = threshold


def KrylovRepresentation(M, vectors, v0, krylov_space_dim=200):
    """
    Calculate the representation of `vectors` and `M` in the Krylov space.

    The representation of `M` in the Krylov space is a real tri-diagonal matrix
    and is returned as a np.ndarray.

    The representation of the given `vectors` in the Krylov space are stored
    as a dict. The keys of the dict are the identifiers of these vectors and
    the values are np.ndarray with shape (actual_krylov_space_dim, ).

    Parameters
    ----------
    M : sparse or dense matrix with shape (N, N)
        The Hermitian matrix for constructing the Krylov space.
    vectors : dict of arrays
        The vectors to be projected onto the Krylov space.
        The keys of the dict are the identifiers of these vectors and the
        values are arrays with shape (N, ).
    v0 : np.ndarray with shape (N, )
        Starting vector for constructing the Krylov space.
    krylov_space_dim : int, optional
        Maximum dimension of the Krylov space. `krylov_space_dim` should be
        in the range[1, N] where N is the dimension of `M`.
        Note: The actually dimension of the constructed Krylov space may be
        less than this value.
        Default: 200.

    Returns
    -------
    projected_matrix : np.ndarray
        The representation of `M` in the Krylov space.
    projected_vectors : dict
        The representation of `vectors` in the Krylov space.
    """

    log_level = logging.INFO - 5
    logger = logging.getLogger(__name__).getChild("KrylovRepresentation")

    v0_norm = np.linalg.norm(v0)
    if v0_norm > _VIEW_AS_ZERO:
        ket = v0 / v0_norm
    else:
        raise ValueError("The given `v0` is a zero vector.")
    ket_old = np.zeros_like(ket)

    As = list()
    Bs = list()
    projected_vectors = dict()
    for key in vectors:
        projected_vectors[key] = list()

    for ith in range(krylov_space_dim):
        t0 = time()
        # Perform projection
        M_dot_ket = M.dot(ket)
        # In principle A and B should be real number
        A = np.vdot(ket, M_dot_ket)
        B = np.vdot(ket_old, M_dot_ket)
        As.append(A.real)
        Bs.append(B.real)
        for key in vectors:
            projected_vectors[key].append(np.vdot(ket, vectors[key]))

        # Update
        M_dot_ket -= A * ket
        M_dot_ket -= B * ket_old
        norm = np.linalg.norm(M_dot_ket)
        if norm > _VIEW_AS_ZERO:
            M_dot_ket /= norm
            ket_old = ket
            ket = M_dot_ket
        else:
            logger.warning("Got zero vector at %3dth iteration", ith)
            break
        t1 = time()
        logger.log(log_level, "Time spend on %3dth ket: %.3fs", ith, t1 - t0)

    Bs = Bs[1:]
    projected_matrix = np.diag(As, k=0) + np.diag(Bs, k=-1) + np.diag(Bs, k=1)
    for key in projected_vectors:
        projected_vectors[key] = np.array(projected_vectors[key])
    return projected_matrix, projected_vectors


def MultiKrylov(M, vectors, krylov_space_dim=200):
    """
    Return the representation of `vectors` and `M` in the Krylov space.

    Choosing every vector in `vectors` as the starting vector,
    this method generate the corresponding Krylov space, calculate the
    representation of `M` and `vectors` in the Krylov space.

    Parameters
    ----------
    M : sparse or dense matrix with shape (N, N)
        The Hermitian matrix for constructing the Krylov space.
    vectors : dict of arrays
        The vectors to be projected onto the Krylov space.
        The keys of the dict are the identifiers of these vectors and the
        values are arrays with shape (N, ).
    krylov_space_dim : int, optional
        Maximum dimension of the Krylov space. `krylov_space_dim` should be
        in the range[1, N] where N is the dimension of `M`.
        Note: The actually dimension of the constructed Krylov space may be
        less than this value.
        Default: 200.

    Returns
    -------
    projected_matrices : dict
        The representations of `M` in these Krylov spaces.
    projected_vectors : dict
        The representations of `vectors` in these Krylov space.
    """

    projected_vectors = dict()
    projected_matrices = dict()
    msg = "Time spend on %2dth starting vector: %.3fs"
    logger = logging.getLogger(__name__).getChild("MultiKrylov")
    for index, key in enumerate(vectors):
        t0 = time()
        projected_matrices[key], projected_vectors[key] = KrylovRepresentation(
            M, vectors, v0=vectors[key], krylov_space_dim=krylov_space_dim
        )
        t1 = time()
        logger.info(msg, index, t1 - t0)
    return projected_matrices, projected_vectors
