"""
Implementation of the Lanczos Algorithm.

See also:
    http://www.netlib.org/utk/people/JackDongarra/etemplates/node85.html
    https://www.cond-mat.de/events/correl11/manuscript/Koch.pdf
"""


__all__ = [
    "Lanczos",
    "set_threshold",
]


import logging
from time import time

import numpy as np
from scipy.sparse import isspmatrix_csr

# Useful global constant
_VIEW_AS_ZERO = 1E-10
################################################################################

logging.getLogger("Lanczos").addHandler(logging.NullHandler())


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
    Exception raised when the requested convergence is not obtained.
    """

    def __init__(self, message):
        self.message = message


class Lanczos:
    """
    Implementation of the Lanczos Algorithm.

    The basic idea behind the Lanczos algorithm is to build a Krylov subspace
    from a matrix(usually large sparse matrix) and a starting vector. Then
    the matrix and vectors can be projected to this Krylov subspace.
    """

    def __init__(self, HM):
        """
        Customize the newly created instance.

        Parameters
        ----------
        HM : csr_matrix
            The compressed sparse row matrix to be processed.
            It must be an Hermitian matrix.
        """

        assert isspmatrix_csr(HM), "`HM` must be instance of csr_matrix"
        assert (HM - HM.getH()).count_nonzero() == 0, "`HM` must be Hermitian"

        self._HM = HM
        self._HMDim = HM.shape[0]

    def _starting_vector(self, v0=None, which="random"):
        # Prepare starting vector for Lanczos iteration.
        # The `which` parameter only takes effect when `v0` is None.

        shape = (self._HMDim, 1)
        assert (
            (v0 is None) or (isinstance(v0, np.ndarray) and v0.shape == shape)
        ), "Invalid `v0` parameter!"

        if v0 is None:
            core_func = np.ones if which == "uniform" else np.random.random
            while True:
                v0 = core_func(shape)
                v0_norm = np.linalg.norm(v0)
                if v0_norm > _VIEW_AS_ZERO:
                    v0 /= v0_norm
                    break
        else:
            v0_norm = np.linalg.norm(v0)
            if v0_norm > _VIEW_AS_ZERO:
                v0 = v0 / v0_norm
            else:
                raise ValueError("The given `v0` is a zero vector!")
        return np.zeros(shape), v0

    # Core function for generating basis of Krylov subspace
    def _lanczos_iter_core(self, vector, vector_old):
        # Calculate the matrix elements:
        # < vector | HM | vector > and < vector_old | HM | vector_old >.
        # In theory, the above matrix elements should be real numbers,
        # but because of numerical errors, they will have small imaginary parts.
        # Here we discard the imaginary parts manually.
        vector_new = self._HM.dot(vector)
        alpha = np.vdot(vector, vector_new).real
        beta = np.vdot(vector_old, vector_new).real

        # Orthogonalize `vector_new` with `vector` and `vector_old`
        vector_new -= alpha * vector
        vector_new -= beta * vector_old
        # If `vector_new` is non-zero vector, normalize it;
        # If zero vector, return None.
        vector_new_norm = np.linalg.norm(vector_new)
        if vector_new_norm <= _VIEW_AS_ZERO:
            vector_new = None
        else:
            vector_new /= vector_new_norm
        return alpha, beta, vector_new

    def ground_state(self, v0=None, *, which="random", tol=1E-12, maxdim=None):
        """
        Finding the smallest eigenvalue of `HM` using Lanczos algorithm.

        This method stop the Lanczos iteration when the difference between
        the smallest eigenvalue of the tridiagonal matrix in this iteration
        and previous iteration is less than the given `tol`. This criterion
        is simple but impractical. This method is not suitable for use in
        practice, it is only provided as a demonstration of the Lanczos
        algorithm. The `eigsh` function in the `scipy.sparse.linalg` package
        is recommended for practical usage.

        Parameters
        ----------
        v0 : np.ndarray or None, optional
            Starting vector for Lanczos iteration.
            If not given or None, a starting vector will be generated
            according to the `which` parameter.
            Default: None.
        which : {"random" or "uniform"}, optional, keyword-only
            Whether to generate a random or uniform starting vector.
            This parameter only takes effect when `v0` is None.
            Default: "random".
        tol : float, optional, keyword-only
            Relative accuracy for eigenvalue(stopping criterion).
            Default: 1E-12.
        maxdim : int, optional, keyword-only
            Maximum dimension of the Krylov subspace. `maxdim` should be in the
            range [1, N] where N is the dimension of `HM`. If `maxdim` was
            set to any invalid value, then `maxdim` will be reset to N.
            Default : None.

        Returns
        -------
        res : float
            The smallest eigenvalue of `HM`.
        """

        assert isinstance(tol, float) and tol >= 0

        logger = logging.getLogger("Lanczos.ground_state")
        message_template = "Time spend on {0:03}th iteration: {1:.3f}s"
        if not (isinstance(maxdim, int) and 0 < maxdim <= self._HMDim):
            maxdim = self._HMDim

        alphas = []
        betas = []
        values = []
        vector_old, vector = self._starting_vector(v0=v0, which=which)
        for i in range(maxdim):
            t0 = time()
            alpha, beta, vector_new = self._lanczos_iter_core(
                vector, vector_old
            )
            alphas.append(alpha)
            betas.append(beta)
            tri = np.diag(alphas, 0)
            tri += np.diag(betas[1:], 1)
            tri += np.diag(betas[1:], -1)
            values.append(np.linalg.eigvalsh(tri)[0])
            message = message_template.format(i, time() - t0)
            logger.info(message)

            if len(values) >= 2 and abs(values[-1] - values[-2]) < tol:
                return values[-1]
            else:
                if vector_new is not None:
                    # Update `vector_old` and `vector` for next iteration
                    vector_old = vector
                    vector = vector_new
                else:
                    raise RuntimeError("Got an invariant subspace.")
        raise ConvergenceError("The required accuracy cannot be obtained!")

    def krylov_basis(self, v0=None, *, which="random", maxdim=200):
        """
        Generate the bases of the Krylov subspace.

        This method calculate the bases of the Krylov subspace as well as the
        representation of `HM` in the Krylov subspace.

        The bases are stored as a np.ndarray with dimension (N, maxdim) and
        every column of the array corresponds to a base vector.

        Parameters
        ----------
        v0 : np.ndarray or None, optional
            Starting vector for Lanczos iteration.
            If not given or None, a starting vector will be generated
            according to the `which` parameter.
            Default: None.
        which : {"random" or "uniform"}, optional, keyword-only
            Whether to generate a random or uniform starting vector.
            This parameter only takes effect when `v0` is None.
            Default: "random".
        maxdim : int, optional, keyword-only
            Maximum dimension of the Krylov subspace. `maxdim` should be in the
            range [1, N] where N is the dimension of `HM`. If `maxdim` was
            set to any invalid value, then `maxdim` will be reset to N.
            Default : 200.

        Returns
        -------
        krylov_repr_matrix : np.ndarray
            The representation of `HM` in the Krylov subspace.
        krylov_bases : np.ndarray
            The bases of the Krylov subspace.
        """

        logger = logging.getLogger("Lanczos.krylov_basis")
        message_template = "Time spend on {0:03}th iteration: {1:.3f}s"
        if not (isinstance(maxdim, int) and 0 < maxdim <= self._HMDim):
            maxdim = self._HMDim

        alphas = []
        betas = []
        bases = []
        vector_old, vector = self._starting_vector(v0=v0, which=which)
        for i in range(maxdim):
            t0 = time()
            alpha, beta, vector_new = self._lanczos_iter_core(
                vector, vector_old
            )
            alphas.append(alpha)
            betas.append(beta)
            bases.append(vector)
            message = message_template.format(i, time() - t0)
            logger.info(message)

            if vector_new is not None:
                # Update `vector_old` and `vector` for next iteration
                vector_old = vector
                vector = vector_new
            else:
                break

        krylov_bases = np.concatenate(bases, axis=1)
        krylov_repr_matrix = np.diag(alphas, 0)
        krylov_repr_matrix += np.diag(betas[1:], 1)
        krylov_repr_matrix += np.diag(betas[1:], -1)
        return krylov_repr_matrix, krylov_bases

    def projection(self, vectors, v0=None, *, which="random", maxdim=200):
        """
        Return the representation of `vectors` and `HM` in the Krylov subspace.

        The representation of `HM` in the Krylov subspace is a tri-diagonal
        matrix and is returned as a np.ndarray.

        The representation of the given `vectors` in the Krylov subspace are
        stored as a dict. The keys of the dict are the identifiers of these
        vectors and the values should be np.ndarray with shape (maxdim, 1).

        Parameters
        ----------
        vectors : dict
            The vectors to be projected onto the Krylov subspace.
            The keys of the dict are the identifiers of these vectors and the
            values of the dict should be np.ndarray with shape (N, 1).
        v0 : np.ndarray or None, optional
            Starting vector for Lanczos iteration.
            If not given or None, a starting vector will be generated
            according to the `which` parameter.
            Default: None.
        which : {"random" or "uniform"}, optional, keyword-only
            Whether to generate a random or uniform starting vector.
            This parameter only takes effect when `v0` is None.
            Default: "random".
        maxdim : int, optional, keyword-only
            Maximum dimension of the Krylov subspace. `maxdim` should be in the
            range [1, N] where N is the dimension of `HM`. If `maxdim` was
            set to any invalid value, then `maxdim` will be reset to N.
            Default : 200.

        Returns
        -------
        krylov_repr_matrix : np.ndarray
            The representation of `HM` in the Krylov subspace.
        krylov_repr_vectors : dict
            The representation of `vectors` in the Krylov subspace.
        """

        logger = logging.getLogger("Lanczos.projection")
        message_template = "Time spend on {0:03}th iteration: {1:.3f}s"
        if not (isinstance(maxdim, int) and 0 < maxdim <= self._HMDim):
            maxdim = self._HMDim

        alphas = []
        betas = []
        krylov_repr_vectors = dict()
        for key in vectors:
            krylov_repr_vectors[key] = list()

        vector_old, vector = self._starting_vector(v0=v0, which=which)
        for i in range(maxdim):
            t0 = time()
            alpha, beta, vector_new = self._lanczos_iter_core(
                vector, vector_old
            )
            alphas.append(alpha)
            betas.append(beta)
            for key in vectors:
                krylov_repr_vectors[key].append(np.vdot(vector, vectors[key]))
            message = message_template.format(i, time() - t0)
            logger.info(message)

            if vector_new is not None:
                # Update `vector_old` and `vector` for next iteration
                vector_old = vector
                vector = vector_new
            else:
                break

        for key in vectors:
            krylov_repr_vectors[key] = np.reshape(
                krylov_repr_vectors[key], newshape=(-1, 1)
            )
        krylov_repr_matrix = np.diag(alphas, 0)
        krylov_repr_matrix += np.diag(betas[1:], 1)
        krylov_repr_matrix += np.diag(betas[1:], -1)
        return krylov_repr_matrix, krylov_repr_vectors

    def __call__(self, vectors, *, maxdim=200):
        """
        Return the representation of `vectors` and `HM` in the Krylov subspace.

        Choosing every vector in `vectors` as the starting vector, this method
        generate the corresponding Krylov subspace, calculate the
        representation of `HM` and `vectors` in this space.

        Parameters
        ----------
        vectors : dict
            The vectors to be projected onto the Krylov subspace.
            The keys of the dict are the identifiers of these vectors and the
            values of the dict should be np.ndarray with shape (N, 1).
        maxdim : int, optional, keyword-only
            Maximum dimension of the Krylov subspace. `maxdim` should be in the
            range [1, N] where N is the dimension of `HM`. If `maxdim` was
            set to any invalid value, then `maxdim` will be reset to N.
            Default : 200.

        Returns
        -------
        krylov_reprs_matrix : dict
            The representations of `HM` in these Krylov subspaces.
        krylov_reprs_vectors : dict
            The representations of `vectors` in these Krylov subspaces.
        """

        logger = logging.getLogger("Lanczos.__call__")
        message_template = "Time spend on the {0:03}th " \
                           "starting vector: {1:.3f}s"
        if not (isinstance(maxdim, int) and 0 < maxdim <= self._HMDim):
            maxdim = self._HMDim

        krylov_reprs_matrix = dict()
        krylov_reprs_vectors = dict()

        for index, key in enumerate(vectors):
            t0 = time()
            krylov_repr_matrix, krylov_repr_vectors = self.projection(
                vectors, v0=vectors[key], maxdim=maxdim
            )
            krylov_reprs_matrix[key] = krylov_repr_matrix
            krylov_reprs_vectors[key] = krylov_repr_vectors
            message = message_template.format(index, time() - t0)
            logger.info(message)
        return krylov_reprs_matrix, krylov_reprs_vectors
