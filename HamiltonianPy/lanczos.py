"""
Implementation of the Lanczos Algorithm
"""


__all__ = [
    "Lanczos",
    "set_threshold",
]


from scipy.sparse import isspmatrix_csr
from time import strftime, time

import numpy as np


# Useful global constant
_VIEW_AS_ZERO = 1E-10
################################################################################



def set_threshold(threshold=1E-10):
    """
    Set the threshold for viewing a vector as zero-vector.

    If the norm of a vector is less than the given `threshold`, then the
    vector is viewed as zero-vector.

    Parameters
    ----------
    threshold : float, optional
        The threshold value.
        Default: 1E-10.
    """

    assert isinstance(threshold, float) and threshold >= 0
    global _VIEW_AS_ZERO
    _VIEW_AS_ZERO = threshold


class ConvergenceError(Exception):
    """
    Exception raised when the requested convergence is not obtained
    """

    def __init__(self, msg):
        self.msg = msg


class Lanczos:
    """
    Implementation of the Lanczos Algorithm

    The basic idea behind the Lanczos method is to build a projection
    HM_Krylov of the full Hamiltonian matrix HM onto a Krylov subspace.
    After the projection, the representation of vectors and HM in this Krylov
    subspace can be obtained and used for later calculation.
    """

    def __init__(self, HM):
        """
        Customize the newly created instance

        Parameters
        ----------
        HM : csr_matrix
            The compressed sparse row matrix to be processed
            It must be an Hermitian matrix
        """

        assert isspmatrix_csr(HM), "`HM` must be instance of csr_matrix"
        assert (HM - HM.getH()).count_nonzero() == 0, "`HM` must be Hermitian"

        self._HM = HM
        self._HMDim = HM.shape[0]

    def _starting_vector(self, v0=None):
        # Prepare starting vector for Lanczos iteration
        shape = (self._HMDim, 1)
        if not isinstance(v0, np.ndarray) or v0.shape != shape:
            if (v0 is None) or (v0 == "random"):
                v0 = np.random.random_sample(size=shape)
            elif v0 == "uniform":
                v0 = np.ones(shape=shape)
            else:
                raise ValueError("Invalid `v0` parameter!")

        v0_norm = np.linalg.norm(v0)
        if v0_norm < _VIEW_AS_ZERO:
            raise ValueError("The given `v0` is a zero vector!")
        return v0 / v0_norm

    def ground_state(self, v0=None, tol=1e-12):
        """
        Find the smallest eigenvalue of HM using Lanczos algorithm

        Parameters
        ----------
        v0 : ndarray
            Starting vector for Lanczos iteration
            Valid value for v0:
                None | "random" | "uniform" | ndarray with shape (N, 1)
            default: None(The same as "random")
        tol : float
            Relative accuracy for eigenvalue(stopping criterion)
            default: 1e-12

        Returns
        -------
        val : float
            The smallest eigenvalue of the matrix

        Raises
        ------
        ConvergenceError
            When the requested convergence accuracy is not obtained
        """

        HM = self._HM
        v_old = 0.0
        v = self._starting_vector(v0)
        v_new = HM.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []

        E = c1
        for i in range(self._HMDim):
            v_new -= c1 * v
            v_new -= v_old

            c0 = np.linalg.norm(v_new)
            if c0 < _VIEW_AS_ZERO:
                raise ConvergenceError("Got an invariant subspace!")
            v_new /= c0

            v_old = v
            v_old *= c0
            v = v_new
            v_new = HM.dot(v)
            c1 = np.vdot(v, v_new)
            c1s.append(c1)
            c0s.append(c0)

            val = np.linalg.eigvalsh(
                np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
            )[0]
            if abs(val - E) < tol:
                return val
            else:
                E = val
        raise ConvergenceError(
            "Got Krylov subspace larger than the original space!"
        )

    def krylov_basis(self, v0=None, maxiter=200):
        """
        Generate the bases of the Krylov subspace

        This method calculate the bases of the Krylov subspace as well as the
        representation of the matrix in the Krylov subspace.

        The bases are stored as a np.ndarray with dimension (N, maxiter) and
        every column of the array represents a base vector.

        Parameters
        ----------
        v0 : ndarray, optional
            Starting vector for iteration
            Valid value for v0:
                None | "random" | "uniform" | ndarray with shape (N, 1)
            default : None(The same as "random")
        maxiter : int, optional
            Number of maximum Lanczos iteration
            The actual number of iteration may less than this because of
            achieving an invariant subspace!
            default: 200

        Returns
        -------
        krylov_repr_matrix : np.ndarray
            The representation of `HM` in the Krylov subspace
        krylov_bases : np.ndarray
            The bases of the Krylov subspace
        """

        assert isinstance(maxiter, int) and 0 < maxiter < self._HMDim

        HM = self._HM
        v_old = 0.0
        v = self._starting_vector(v0)
        v_new = HM.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []
        bases = [v]

        for i in range(1, maxiter):
            v_new -= c1 * v
            v_new -= v_old

            c0 = np.linalg.norm(v_new)
            if c0 < _VIEW_AS_ZERO:
                break
            v_new /= c0

            v_old = v
            v_old *= c0
            v = v_new
            v_new = HM.dot(v)
            c1 = np.vdot(v, v_new)
            bases.append(v)
            c1s.append(c1)
            c0s.append(c0)

        krylov_bases = np.concatenate(bases, axis=1)
        krylov_repr_matrix = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
        return krylov_repr_matrix, krylov_bases

    def projection(self, vectors, v0=None, maxiter=200):
        """
        Return the representation of vectors and matrix in the Krylov subspace

        The representation of HM in the Krylov subspace is a tri-diagonal
        matrix and is returned as a np.ndarray.

        The representation of the given vectors in the Krylov subspace are
        stored as a dict. The keys of the dict are the identifiers of these
        vectors and the values should be np.ndarray with shape (maxiter, 1).

        Parameters
        ----------
        vectors : dict
            The vectors to be projected onto the Krylov subspace
            The keys of the dict are the identifiers of these vectors and the
            values of this dict should be np.ndarray with shape (N, 1).
        v0 : ndarray, optional
            Starting vector for iteration
            Valid value for v0:
                None | "random" | "uniform" | ndarray with shape (N, 1)
            default : None(The same as "random")
        maxiter : int, optional
            Number of maximum Lanczos iteration
            The actual number of iteration may less than this because of
            achieving an invariant subspace!
            default: 200

        Returns
        -------
        krylov_repr_matrix : np.ndarray
            The representation of the matrix in the Krylov subspace
        krylov_repr_vectors : dict
            The representation of the vectors in the Krylov subspace
        """

        assert isinstance(maxiter, int) and 0 < maxiter < self._HMDim

        HM = self._HM
        v_old = 0.0
        v = self._starting_vector(v0)
        v_new = HM.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []

        krylov_repr_vectors = dict()
        for key in vectors:
            krylov_repr_vectors[key] = [np.vdot(v, vectors[key])]

        for i in range(1, maxiter):
            v_new -= c1 * v
            v_new -= v_old

            c0 = np.linalg.norm(v_new)
            if c0 < _VIEW_AS_ZERO:
                break
            v_new /= c0

            v_old = v
            v_old *= c0
            v = v_new
            v_new = HM.dot(v)
            c1 = np.vdot(v, v_new)
            c1s.append(c1)
            c0s.append(c0)

            for key in vectors:
                krylov_repr_vectors[key].append(np.vdot(v, vectors[key]))

        for key in vectors:
            krylov_repr_vectors[key] = np.reshape(
                krylov_repr_vectors[key], newshape=(-1, 1)
            )
        krylov_repr_matrix = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
        return krylov_repr_matrix, krylov_repr_vectors

    def __call__(self, vectors, maxiter=200, process_num=1, log=False):
        """
        Return the representation of vectors and matrix in the Krylov subspace

        Choosing every vector in `vectors` as the starting vector, this method
        generate the Krylov subspace, calculate the representation of the
        matrix and all vectors in this space.

        Parameters
        ----------
        vectors : dict
            The vectors to be projected onto the Krylov subspace
            The keys of the dict are the identifiers of these vectors and the
            values of this dict should be np.ndarray with shape (N, 1).
        maxiter : int, optional
            Number of maximum Lanczos iteration
            The actual number of iteration may less than this because of
            achieving an invariant subspace!
            default: 200
        process_num : int, optional
            The number of process to use.
            It is recommended to set this parameter to be integer which can be
            divided exactly by the number of vectors for load balancing of every
            process, also, this parameter should not be too large because of the
            bandwidth limit of the RAM.
            default: 1
        log : boolean, optional
            Whether to print the log information to stdout
            default: False

        Returns
        -------
        krylov_reprs_matrix : dict
            The representations of HM in these Krylov subspaces.
        krylov_reprs_vectors : dict
            The representations of the vectors in these Krylov subspaces.
        """

        assert isinstance(process_num, int) and process_num > 0
        # TODO: add support for multi-process parallel

        krylov_reprs_matrix = dict()
        krylov_reprs_vectors = dict()

        fmt = "%Y-%d-%m %H:%M:%S"
        log_template = "{0}: {1}, {2}th vector, {3}s"
        for index, key in enumerate(vectors):
            t0 = time()
            krylov_repr_matrix, krylov_repr_vectors = self.projection(
                vectors, vectors[key], maxiter
            )
            krylov_reprs_matrix[key] = krylov_repr_matrix
            krylov_reprs_vectors[key] = krylov_repr_vectors
            t1 = time()
            if log:
                msg = log_template.format(strftime(fmt), key, index, t1 - t0)
                print(msg, flush=True)
        return krylov_reprs_matrix, krylov_reprs_vectors
