"""
Implementation of the Lanczos Algorithm
"""


__all__ = [
    "Lanczos",
]


from scipy.sparse import isspmatrix_csr
from time import time

import multiprocessing as mp
import numpy as np


# Useful constant
_VIEW_AS_ZERO = 1e-8
################################################################################


class ConvergenceError(Exception):
    """
    Exception raised when the requested convergence is not obtained
    """

    def __init__(self, msg):
        self.msg = msg


class Lanczos:
    """
    Implementation of the Lanczos Algorithm

    This class provide method to build a projection of a large matrix onto a
    Krylov subspace. Calculating representation of vectors and matrices in
    this Krylov subspace.

    Attributes
    HM : csr_matrix
        The compressed sparse row matrix to be processed
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

        if not isspmatrix_csr(HM):
            raise TypeError("The input `HM` is not a csr_matrix!")

        if np.any(np.absolute((HM - HM.getH()).data) > _VIEW_AS_ZERO):
            raise ValueError("The input `HM` is not an Hermitian matrix!")

        self.HM = HM

    def _starting_vector(self, v0=None):
        # Prepare starting vector for Lanczos iteration
        shape = (self.HM.shape[0], 1)
        if v0 is None:
            v0 = np.random.random_sample(size=shape)
        elif isinstance(v0, str):
            if v0 == "random":
                v0 = np.random.random_sample(size=shape)
            elif v0 == "uniform":
                v0 = np.ones(shape=shape)
            else:
                raise ValueError("The given v0 string is not supported!")
        else:
            if not (isinstance(v0, np.ndarray) and v0.shape ==shape):
                raise ValueError("The shape of v0 and HM does not match!")

        v0_norm = np.linalg.norm(v0)
        if v0_norm < _VIEW_AS_ZERO:
            raise ValueError("The given v0 is a zero vector!")
        return v0 / v0_norm

    def ground_state(self, v0=None, tol=1e-12):
        """
        Find the smallest eigenvalue of `HM` using Lanczos algorithm

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

        HM = self.HM

        v_old = 0.0
        v = self._starting_vector(v0)
        v_new = HM.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []

        E = c1
        for i in range(HM.shape[0]):
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

    def krylov_basis(self, v0=None, step=200):
        """
        Generate the bases of the Krylov subspace

        This method calculate the bases of the Krylov subspace as well as the
        representation of the matrix in the Krylov subspace.

        The bases are stored as a np.ndarray with dimension (N, step) and every
        column of the array represents a base vector.

        Parameters
        ----------
        v0 : ndarray, optional
            Starting vector for iteration
            Valid value for v0:
                None | "random" | "uniform" | ndarray with shape (N, 1)
            default : None(The same as "random")
        step : int, optional
            Number of Lanczos iteration
            default: 200

        Returns
        -------
        tri : ndarray
            The representation of the matrix in the Krylov subspace
        krylov_bases : ndarray
            The bases of the Krylov subspace
        """

        HM = self.HM
        assert isinstance(step, int) and step > 0 and step < HM.shape[0]

        v_old = 0.0
        v = self._starting_vector(v0)
        v_new = HM.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []
        bases = [v]

        for i in range(1, step):
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
        tri = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
        return tri, krylov_bases

    def projection(self, vectors, v0=None, step=200):
        """
        Return the representation of vectors and matrix in the Krylov subspace

        The representation of HM in the Krylov subspace is a tri-diagonal
        matrix and is returned as a np.ndarray.

        The representation of the given vectors in the Krylov subspace are
        stored as a dict. The keys of the dict are the identifiers of these
        vectors and the values should be np.ndarray with shape (step, 1).

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
        step : int, optional
            Number of Lanczos iteration
            default: 200

        Returns
        -------
        repr_matrix : np.array
            The representation of the matrix in the Krylov subspace
        repr_vectors : dict
            The representation of the vectors in the Krylov subspace
        """

        HM = self.HM
        assert isinstance(step, int) and step > 0 and step < HM.shape[0]

        v_old = 0.0
        v = self._starting_vector(v0)
        v_new = HM.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []

        repr_vectors = dict()
        for key in vectors:
            repr_vectors[key] = [np.vdot(v, vectors[key])]

        for i in range(1, step):
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
                repr_vectors[key].append(np.vdot(v, vectors[key]))
        for key in vectors:
            repr_vectors[key] = np.reshape(repr_vectors[key], newshape=(-1, 1))
        repr_matrix = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
        return repr_matrix, repr_vectors

    def _parallel_projection(self, keys, vectors, step, proxy_matrix,
                             proxy_vectors):
        # Defined for multi-process parallel
        for key in keys:
            t0 = time()
            repr_matrix, repr_vectors = self.projection(
                vectors, vectors[key], step
            )
            proxy_matrix[key] = repr_matrix
            proxy_vectors[key] = repr_vectors
            t1 = time()
            print("The current key: {}".format(key), flush=True)
            print("The time spend on this key: {}s".format(t1 - t0), flush=True)
            print("=" * 80, flush=True)

    def __call__(self, vectors, step=200, process_num=1):
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
        process_num : int, optional
            The number of process to use.
            It is recommended to set this parameter to be integer which can be
            divided exactly by the number of vectors for load balancing of every
            process, also, this parameter should not be too large because of the
            bandwidth limit of the RAM.
            default: 1
        step : int, optional
            Number of Lanczos iteration
            default: 200

        Returns
        -------
        reprs_matrix : dict
            The representations of HM in these Krylov subspaces.
        reprs_vectors : dict
            The representations of the vectors in these Krylov subspaces.
        """

        assert isinstance(process_num, int) and process_num > 0

        reprs_matrix = dict()
        reprs_vectors = dict()
        keys = tuple(vectors.keys())

        if process_num == 1:
            for index, key in enumerate(keys):
                t0 = time()
                repr_matrix, repr_vectors = self.projection(
                    vectors, vectors[key], step
                )
                reprs_matrix[key] = repr_matrix
                reprs_vectors[key] = repr_vectors
                t1 = time()
                info = "The time spend on the {0}th starting vector: {1}s"
                print(info.format(index, t1 - t0), flush=True)
                print("=" * 80, flush=True)
        else:
            with mp.Manager() as manager:
                proxy_matrix = manager.dict()
                proxy_vectors = manager.dict()
                kwargs = {
                    "vectors": vectors,
                    "proxy_matrix": proxy_matrix,
                    "proxy_vectors": proxy_vectors,
                    "step": step,
                }

                processes = []
                for process_index in range(process_num):
                    process = mp.Process(
                        target=self._parallel_projection,
                        args=(keys[process_index::process_num], ),
                        kwargs=kwargs
                    )
                    process.start()
                    processes.append(process)

                for process in processes:
                    process.join()

                for key in keys:
                    reprs_matrix[key] = proxy_matrix[key]
                    reprs_vectors[key] = proxy_vectors[key]
        return reprs_matrix, reprs_vectors
