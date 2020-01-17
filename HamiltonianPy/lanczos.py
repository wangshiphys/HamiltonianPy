"""
Implementation of the Lanczos algorithm.

The basic idea behind the Lanczos algorithm is to build a Krylov space
from a matrix M (usually large sparse matrix) and a starting vector v0.
Starting from v0, by iterated application of M (v0 is assumed to be
normalized):
    v1' = M.dot(v0),
    orthogonalize v1' with v0 we get v1'', normalize v1'' we get v1;
    v2' = M.dot(v1),
    orthogonalize v2' with v0 and v1 we get v2'', normalize v2'' we get v2;
    v3' = M.dot(v2),
    orthogonalize v3' with v0, ..., v2 we get v3'', normalize v3'' we get v3;
    ......
    v_n' = M.dot(v_{n-1}),
    orthogonalize v_n' with v0, ..., v_{n-1} we get v_n'', normalize v_n'' we get v_n;
    ......
The Krylov space is spanned by the orthonormalized vectors:
    {v0, v1, ..., v_n, ...}
"""


__all__ = [
    "Schmidt",
    "Lanczos",
    "KrylovSpace",
    "MultiKrylov",
    "set_threshold",
    "KrylovRepresentation",
]


import logging
from time import time

import numpy as np
from numba import jit, prange


# |Values| <= `_VIEW_AS_ZERO` will be treated as zero in this module
_VIEW_AS_ZERO = 1E-10
################################################################################

logging.getLogger(__name__).addHandler(logging.NullHandler())


# Both `ket_new` and `ket` with shape: (N, )
@jit(nopython=True, cache=True, parallel=True)
def InplaceSubtract1D(ket_new, ket, coeff):
    for i in prange(ket_new.shape[0]):
        ket_new[i] -= coeff * ket[i]


# Both `ket_new` and `ket` with shape: (N, 1)
@jit(nopython=True, cache=True, parallel=True)
def InplaceSubtract2D(ket_new, ket, coeff):
    for i in prange(ket_new.shape[0]):
        ket_new[i, 0] -= coeff * ket[i, 0]


def set_threshold(threshold=1E-10):
    """
    Set the threshold for viewing a value as zero.

    If |x| <= threshold, then `x` is viewed as zero. If you want to change
    the default threshold, you must call this function before using any other
    functions or classes defined in this module.

    Parameters
    ----------
    threshold : float, optional
        The threshold value.
        Default: 1E-10.
    """

    assert isinstance(threshold, float) and threshold >= 0
    global _VIEW_AS_ZERO
    _VIEW_AS_ZERO = threshold


def Schmidt(vectors, check=False):
    """
    Schmidt orthogonalization.

    Parameters
    ----------
    vectors : list of 1D arrays
        A collection of vectors to be orthonormalized.
        All vectors in `vectors` must be of the same shape.
    check : bool, optional
        Whether to check the orthonormality of the resulting vectors.
        Note: This operation is time consuming.
        Default: False.

    Returns
    -------
    vectors : list of 1D arrays
        The corresponding orthonormalized vectors.
    """

    vector_num = len(vectors)
    orthonormal_vectors = [vectors[0] / np.linalg.norm(vectors[0])]
    for i in range(1, vector_num):
        vector_old = vectors[i]
        vector_new = np.array(vector_old, copy=True)
        for orthonormal_vector in orthonormal_vectors:
            coeff = np.vdot(orthonormal_vector, vector_old)
            vector_new -= coeff * orthonormal_vector
        vector_new /= np.linalg.norm(vector_new)
        orthonormal_vectors.append(vector_new)

    if check:
        for i in range(vector_num):
            vector_i = orthonormal_vectors[i]
            for j in range(vector_num):
                vector_j = orthonormal_vectors[j]
                inner = np.vdot(vector_i, vector_j)
                inner_ref = 1 if i == j else 0
                assert np.allclose(inner, inner_ref)
    return orthonormal_vectors


def _starting_vector(N, v0=None, which="random"):
    # Prepare starting vector for Lanczos iteration.
    # The `which` parameter only takes effect when `v0` is None.
    # The shape of the starting vector should be (N, 1).

    assert isinstance(N, int) and N > 0, "`N` must be positive integer."

    shape = (N, 1)
    if v0 is None:
        if which == "uniform":
            theta = 2 * np.pi * np.random.random()
            v0 = np.ones(shape, dtype=np.float64) * np.exp(1j * theta)
            v0 /= np.linalg.norm(v0)
        else:
            while True:
                v0 = np.random.random(shape) + np.random.random(shape) * 1j
                v0_norm = np.linalg.norm(v0)
                if v0_norm > _VIEW_AS_ZERO:
                    v0 /= v0_norm
                    break
    elif isinstance(v0, np.ndarray):
        if v0.shape == (N, ) or v0.shape == (N, 1):
            v0_norm = np.linalg.norm(v0)
            if v0_norm > _VIEW_AS_ZERO:
                # The passed in `v0` should not be changed
                v0 = v0.reshape((-1, 1)) / v0_norm
            else:
                raise ValueError("The given `v0` is a zero vector.")
        else:
            msg = "The shape of `v0` is expected to be {0}, but got {1}."
            raise ValueError(msg.format(shape, v0.shape))
    else:
        raise ValueError("Invalid `v0`.")
    return v0


def KrylovSpace(M, v0=None, which="random", krylov_space_dim=200):
    """
    Construct Krylov space from the given `M` and `v0`.

    Parameters
    ----------
    M : csr_matrix with shape (N, N)
        The matrix for constructing the Krylov space.
        It must be compressed sparse row matrix (csr_matrix) and Hermitian.
    v0 : np.ndarray with shape (N, 1) or None, optional
        Starting vector for constructing the Krylov space.
        If not given or None, a starting vector will be generated according
        to the `which` parameter.
        Default: None.
    which : ["random" | "uniform"], str, optional
        Whether to generate a random starting vector or an uniform one.
        This parameter only takes effect when `v0` is None.
        Default: "random".
    krylov_space_dim : int, optional
        Maximum dimension of the Krylov space. `krylov_space_dim` should be
        in the range[1, N] where N is the dimension of `M`.
        Note: The actually dimension of the constructed Krylov space may be
        less than this value.
        Default: 200.

    Returns
    -------
    krylov_matrix : np.ndarray
        The representation of `M` in the Krylov space.
    krylov_bases : list
        The bases of the Krylov space.
    """

    assert isinstance(krylov_space_dim, int) and krylov_space_dim > 0
    v0 = _starting_vector(N=M.shape[0], v0=v0, which=which)

    As = []
    Bs = []
    krylov_bases = [v0]
    logger = logging.getLogger(__name__).getChild("KrylovSpace")

    start = time()
    for i in range(1, krylov_space_dim):
        t0 = time()
        ket_old = krylov_bases[-1]
        M_dot_ket_old = M.dot(ket_old)
        # `ket_new` is a copy of `M_dot_ket_old`
        # `ket_new` will be updated inplace
        # while `M_dot_ket_old` keep unchanged
        ket_new = np.array(M_dot_ket_old, copy=True)
        # Orthogonalize `ket_new` with previous kets.
        for ket in krylov_bases:
            coeff = np.vdot(ket, M_dot_ket_old)
            # ket_new -= coeff * ket
            InplaceSubtract2D(ket_new, ket, coeff)
        ket_new_norm = np.linalg.norm(ket_new)
        if ket_new_norm <= _VIEW_AS_ZERO:
            logger.warning("Got zero vector at %3dth iteration", i)
            break
        else:
            ket_new /= ket_new_norm
            A = np.vdot(ket_old, M_dot_ket_old)
            B = np.vdot(ket_new, M_dot_ket_old)
            assert A.imag <= _VIEW_AS_ZERO
            assert B.imag <= _VIEW_AS_ZERO
            As.append(A.real)
            Bs.append(B.real)
            krylov_bases.append(ket_new)
        t1 = time()
        logger.info("The time spend on %3dth ket: %.4fs", i, t1 - t0)
    A = np.vdot(krylov_bases[-1], M.dot(krylov_bases[-1]))
    assert A.imag <= _VIEW_AS_ZERO
    As.append(A.real)
    krylov_matrix = np.diag(As, k=0) + np.diag(Bs, k=-1) + np.diag(Bs, k=1)
    end = time()
    logger.info("The total time spend on Krylov space: %.4fs", end - start)
    return krylov_matrix, krylov_bases


def KrylovRepresentation(
        M, vectors, v0=None, which="random", krylov_space_dim=200
):
    """
    Calculate the representation of `vectors` and `M` in the Krylov space.

    The representation of `M` in the Krylov space is a tri-diagonal matrix
    and is returned as a np.ndarray.

    The representation of the given `vectors` in the Krylov space are stored
    as a dict. The keys of the dict are the identifiers of these vectors and
    the values are np.ndarray with shape (actual_krylov_space_dim, 1).

    Parameters
    ----------
    M : csr_matrix with shape (N, N)
        The matrix for constructing the Krylov space.
        It must be compressed sparse row matrix (csr_matrix) and Hermitian.
    vectors : dict of arrays
        The vectors to be projected onto the Krylov space.
        The keys of the dict are the identifiers of these vectors and the
        values are arrays with shape (N, 1).
    v0 : np.ndarray with shape (N, 1) or None, optional
        Starting vector for constructing the Krylov space.
        If not given or None, a starting vector will be generated according
        to the `which` parameter.
        Default: None.
    which : ["random" | "uniform"], str, optional
        Whether to generate a random starting vector or an uniform one.
        This parameter only takes effect when `v0` is None.
        Default: "random".
    krylov_space_dim : int, optional
        Maximum dimension of the Krylov space. `krylov_space_dim` should be
        in the range[1, N] where N is the dimension of `M`.
        Note: The actually dimension of the constructed Krylov space may be
        less than this value.
        Default: 200.

    Returns
    -------
    krylov_matrix : np.ndarray
        The representation of `M` in the Krylov space.
    krylov_vectors : dict
        The representation of `vectors` in the Krylov space.
    """

    krylov_matrix, krylov_bases = KrylovSpace(M, v0, which, krylov_space_dim)
    actual_dim = len(krylov_bases)

    krylov_vectors = {}
    for key in vectors:
        vector = vectors[key]
        krylov_vector = np.zeros((actual_dim, 1), dtype=np.complex128)
        for i in range(actual_dim):
            krylov_vector[i, 0] = np.vdot(krylov_bases[i], vector)
        krylov_vectors[key] = krylov_vector
    return krylov_matrix, krylov_vectors


def MultiKrylov(M, vectors, krylov_space_dim=200):
    """
    Return the representation of `vectors` and `M` in the Krylov space.

    Choosing every vector in `vectors` as the starting vector,
    this method generate the corresponding Krylov space, calculate the
    representation of `M` and `vectors` in the Krylov space.

    Parameters
    ----------
    M : csr_matrix with shape (N, N)
        The matrix for constructing the Krylov space.
        It must be compressed sparse row matrix (csr_matrix) and Hermitian.
    vectors : dict of arrays
        The vectors to be projected onto the Krylov space.
        The keys of the dict are the identifiers of these vectors and the
        values are arrays with shape (N, 1).
    krylov_space_dim : int, optional
        Maximum dimension of the Krylov space. `krylov_space_dim` should be
        in the range[1, N] where N is the dimension of `M`.
        Note: The actually dimension of the constructed Krylov space may be
        less than this value.
        Default: 200.

    Returns
    -------
    krylovs_matrix : dict
        The representations of `M` in these Krylov spaces.
    krylovs_vectors : dict
        The representations of `vectors` in these Krylov space.
    """

    krylovs_matrix = dict()
    krylovs_vectors = dict()
    msg = "The time spend on the %2dth starting vector: %.4fs"
    logger = logging.getLogger(__name__).getChild("MultiKrylov")
    for index, key in enumerate(vectors):
        t0 = time()
        krylovs_matrix[key], krylovs_vectors[key] = KrylovRepresentation(
            M, vectors, v0=vectors[key], krylov_space_dim=krylov_space_dim
        )
        t1 = time()
        logger.info(msg, index, t1 - t0)
    return krylovs_matrix, krylovs_vectors


class Lanczos:
    """
    Implementation of the Lanczos Algorithm.

    This class is provided for backward compatibility. The equivalent module
    level functions are recommended.
    """

    def __init__(self, M):
        """
        Customize the newly created instance.

        Parameters
        ----------
        M : csr_matrix with shape (N, N)
            The matrix for constructing the Krylov space.
            It must be compressed sparse row matrix (csr_matrix) and Hermitian.
        """

        self._M = M
        self._MDim = M.shape[0]

    def KrylovSpace(self, v0=None, which="random", krylov_space_dim=200):
        """
        Construct Krylov space from the given starting vector `v0`.

        Parameters
        ----------
        v0 : np.ndarray with shape (N, 1) or None, optional
            Starting vector for constructing the Krylov space.
            If not given or None, a starting vector will be generated according
            to the `which` parameter.
            Default: None.
        which : ["random" | "uniform"], str, optional
            Whether to generate a random starting vector or an uniform one.
            This parameter only takes effect when `v0` is None.
            Default: "random".
        krylov_space_dim : int, optional
            Maximum dimension of the Krylov space. `krylov_space_dim` should be
            in the range[1, N] where N is the dimension of `M`.
            Note: The actually dimension of the constructed Krylov space may be
            less than this value.
            Default: 200.

        Returns
        -------
        krylov_matrix : np.ndarray
            The representation of `M` in the Krylov space.
        krylov_bases : list
            The bases of the Krylov space.
        """

        return KrylovSpace(
            self._M, v0=v0, which=which, krylov_space_dim=krylov_space_dim
        )

    def projection(
            self, vectors, v0=None, which="random", krylov_space_dim=200
    ):
        """
        Calculate the representation of `vectors` and `M` in the Krylov space.

        The representation of `M` in the Krylov space is a tri-diagonal
        matrix and is returned as a np.ndarray.

        The representation of the given `vectors` in the Krylov space are
        stored as a dict. The keys of the dict are the identifiers of these
        vectors and the values are np.ndarray with shape
        (actual_krylov_space_dim, 1).

        Parameters
        ----------
        vectors : dict of arrays
            The vectors to be projected onto the Krylov space.
            The keys of the dict are the identifiers of these vectors and the
            values are arrays with shape (N, 1).
        v0 : np.ndarray with shape (N, 1) or None, optional
            Starting vector for constructing the Krylov space.
            If not given or None, a starting vector will be generated according
            to the `which` parameter.
            Default: None.
        which : ["random" | "uniform"], str, optional
            Whether to generate a random starting vector or an uniform one.
            This parameter only takes effect when `v0` is None.
            Default: "random".
        krylov_space_dim : int, optional
            Maximum dimension of the Krylov space. `krylov_space_dim` should be
            in the range[1, N] where N is the dimension of `M`.
            Note: The actually dimension of the constructed Krylov space may be
            less than this value.
            Default: 200.

        Returns
        -------
        krylov_matrix : np.ndarray
            The representation of `M` in the Krylov space.
        krylov_vectors : dict
            The representation of `vectors` in the Krylov space.
        """

        return KrylovRepresentation(
            self._M, vectors, v0=v0,
            which=which, krylov_space_dim=krylov_space_dim
        )

    def __call__(self, vectors, krylov_space_dim=200):
        """
        Return the representation of `vectors` and `M` in the Krylov space.

        Choosing every vector in `vectors` as the starting vector,
        this method generate the corresponding Krylov space, calculate the
        representation of `M` and `vectors` in the Krylov space.

        Parameters
        ----------
        vectors : dict of arrays
            The vectors to be projected onto the Krylov space.
            The keys of the dict are the identifiers of these vectors and the
            values are arrays with shape (N, 1).
        krylov_space_dim : int, optional
            Maximum dimension of the Krylov space. `krylov_space_dim` should be
            in the range[1, N] where N is the dimension of `M`.
            Note: The actually dimension of the constructed Krylov space may be
            less than this value.
            Default: 200.

        Returns
        -------
        krylovs_matrix : dict
            The representations of `M` in these Krylov spaces.
        krylovs_vectors : dict
            The representations of `vectors` in these Krylov space.
        """

        krylovs_matrix = dict()
        krylovs_vectors = dict()
        logger = logging.getLogger(__name__).getChild(
            ".".join([self.__class__.__name__, "Call"])
        )
        msg = "The time spend on the %2dth starting vector: %.4fs"
        for index, key in enumerate(vectors):
            t0 = time()
            krylovs_matrix[key], krylovs_vectors[key] = KrylovRepresentation(
                self._M, vectors, v0=vectors[key],
                krylov_space_dim=krylov_space_dim
            )
            t1 = time()
            logger.info(msg, index, t1 - t0)
        return krylovs_matrix, krylovs_vectors
