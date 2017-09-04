"""
The implemetation of the Lanczos Algorithm.
"""

__all__ = ['Lanczos']


from numpy.linalg import eigvalsh, norm
from scipy.sparse import csr_matrix, isspmatrix_csr
from time import time

import multiprocessing as mp
import numpy as np

#Useful constant
VIEW_AS_ZERO = 1e-8
###################


class ConvergenceError(Exception):# {{{
    """
    Exception raised when the requested convergence is not obtained.
    """

    def __init__(self, msg):
        self.msg = msg
# }}}

class Lanczos:# {{{
    """
    Implementation of the Lanczos Algorithm.

    This class provide method to build a projection of the large compressed
    sparse row matrix onto a Krylov subspace. Generate the representation of
    vectors and matrix in this Krylov subspace.

    Methods:
    -------
    Public methods:
        gs()
        krylov_basis()
        projection(vectors, v_init)
    Special methods:
        __init__(HM, *, v0=None, step=400, tol=1e-12, uniform=False)
        __call__(vectors)
    """

    def __init__(self, HM, *, v0=None, step=400, tol=1e-12, uniform=False):# {{{
        """
        Initialize the instance of this class.

        Parameters
        ----------
        HM : csr_matrix
            The compressed sparse row matrix to be processed.
            It must be a Hermitian matrix.
        v0 : ndarray, keyword-only, optional
            Starting vector for iteration, v0 is assumed of shape (dim,1).
            If not given or None, a random or uniform vector of shape(dim, 1)
            will be generated according to the uniform parameter.
            default: None
        step : int, keyword-only, optional
            The maximum number of lanczos iteration (stopping critertion).
            default: 400
        tol : float, keyword-only, optional
            Relative accuracy for eigenvalues (stopping criterion).
            default: 1e-12
        uniform : boolean, keyword-only, optional
            If v0 is not given or None, this parameter determines whether to
            use a uniform starting vector. If False, a random starting vector
            will be used.
            default: False
        """

        if not isspmatrix_csr(HM):
            raise TypeError("The input HM matrix is not a csr_matrix!")
        elif ((HM - HM.getH()) > VIEW_AS_ZERO).count_nonzero() != 0:
            raise TypeError("The input HM matrix is not an Hermitian matrix!")
        else:
            dim = HM.shape[0]
            self._HM = HM
            self._dim = dim

        if v0 is None:
            if uniform:
                v0 = np.ones(shape=(dim, 1))
            else:
                v0 = np.random.random_sample(size=(dim, 1))
            v0 /= norm(v0)
        else:
            if v0.shape != (dim, 1):
                raise ValueError("The dimension of v0 and HM does not match!")
            v0_norm = norm(v0)
            #Ensure the input v0 is not a zero vector!
            if v0_norm < VIEW_AS_ZERO:
                raise ValueError("The given v0 is a zero vector!")
            v0 = v0 / v0_norm
        self._v0 = v0

        if isinstance(step, int) and step > 0:
            self._step = step
        else:
            raise ValueError("The step parameter should be positive integer.")

        if isinstance(tol, float) and tol > 0:
            self._tol = tol
        else:
            raise ValueError("The tol parameter should be positive float.")
    # }}}

    def gs(self):# {{{
        """
        Find samllest eigenvalue of the HM matrix.

        This method calculate the smallest eigenvalue of the input spasre 
        matrix with the given precision specified by the tol parameter.

        Returns
        -------
        val : float
            The smallest eigenvalue of the matrix.

        Raises
        ------
        ConvergenceError
            When the requested convergence is not obtained.
        """

        M = self._HM

        v_old = 0.0
        v = np.array(self._v0, copy=True)
        v_new = M.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []

        E = c1
        for i in range(self._dim):
            v_new -= c1 * v
            v_new -= v_old

            c0 = norm(v_new)
            if c0 < VIEW_AS_ZERO:
                raise ConvergenceError("Got an invariant subspace!")
                #break
            v_new /= c0

            v_old = v
            v_old *= c0
            v = v_new
            v_new = M.dot(v)
            c1 = np.vdot(v, v_new)
            c1s.append(c1)
            c0s.append(c0)

            tri = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
            val = eigvalsh(tri)[0]
            if abs(val - E) < self._tol:
                return val
            else:
                E = val
        errmsg = "Got Krylov subspace larger than the original space."
        raise ConvergenceError(errmsg)
    # }}}

    def krylov_basis(self):# {{{
        """
        Generate the bases of the Krylov subspace.

        This method calculate the bases of the Krylov subspace as well as the
        representation of the matrix in the base.
        The projected matrix has the tridiagonal form and is return as the tri.
        The bases are stored as a ndarray with dimension (self._dim, self._step)
        Every column of the array represents a base vector.

        Returns
        -------
        tri : ndarray
            The projection of the input matrix in the Krylov space.
        krylov_basis : ndarray
            The bases of the krylov space.
        """

        M = self._HM

        v_old = 0.0
        v = np.array(self._v0, copy=True)
        v_new = M.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []
        bases = [v]

        for i in range(1, self._step):
            v_new -= c1 * v
            v_new -= v_old

            c0 = norm(v_new)
            if c0 < VIEW_AS_ZERO:
                break

            v_new /= c0
            v_old = v
            v_old *= c0
            v = v_new
            v_new = M.dot(v)
            c1 = np.vdot(v, v_new)
            bases.append(v)
            c1s.append(c1)
            c0s.append(c0)
        krylov_bases = np.concatenate(bases, axis=1)
        tri = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
        return tri, krylov_bases
    # }}}

    def projection(self, vectors, v_init):# {{{
        """
        Return the representation of vectors and matrix in the Krylov space.

        The representation of self._HM in Krylov space is a tridiagonal matrix
        and is returned as a np.array. The representaion of the given vectors
        in the Krylov space are stored as dict. The keys of the dict are
        the identifiers of these vectors and the values should be np.array with
        shape (self._step, 1).

        Parameters
        ----------
        vectors : dict
            The vectors to be projected in the Krylov space. The keys of the
            dict are the identifiers of these vectors. The values of this dict
            should be np.array and of shape (self.dim, 1).
        v_init : ndarray
            Starting vector of the lanczos iteration.

        Returns
        -------
        HM_proj : np.array
            The tridiagonal representaion of the matrix.
        vectors_proj : dict
            The projection of the vectors in the Krylov space.
        """

        norm_v = norm(v_init)
        if norm_v < VIEW_AS_ZERO:
            raise ValueError("The given v_init is a zero vector.")

        M = self._HM

        v_old = 0.0
        v = v_init / norm_v
        v_new = M.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []

        keys = vectors.keys()
        temp = dict()
        for key in keys:
            temp[key] = [np.vdot(v, vectors[key])]

        for i in range(1, self._step):
            v_new -= c1 * v
            v_new -= v_old

            c0 = norm(v_new)
            if c0 < VIEW_AS_ZERO:
                break
            v_new /= c0

            v_old = v
            v_old *= c0
            v = v_new
            v_new = M.dot(v)
            c1 = np.vdot(v, v_new)

            c1s.append(c1)
            c0s.append(c0)

            for key in keys:
                temp[key].append(np.vdot(v, vectors[key]))

        vectors_proj = dict()
        for key in keys:
            vectors_proj[key] = np.array([temp[key]]).T
        HM_proj = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
        return HM_proj, vectors_proj
    # }}}

    def _pprojection(self, keys, vectors, HM_proxy, vectors_proxy):# {{{
        #Defined for multi-process parallel.

        for key in keys:
            t0 = time()
            HM_proj, vectors_proj = self.projection(vectors, vectors[key])
            HM_proxy[key] = HM_proj
            vectors_proxy[key] = vectors_proj
            t1 = time()
            info = "The time spend on this key: {0}s".format(t1 - t0)
            print("The current key:\n", key, flush=True)
            print(info, flush=True)
            print("=" * len(info), flush=True)
    # }}}

    def __call__(self, vectors, *, procs_num=1):# {{{
        """
        Return the representation of vectors and matrix in the Krylov space.

        Choosing every vector in vectors as the starting vector, this method
        generate the krylov space, calculate the representation of the matrix
        and all vectors in this space.

        Parameters
        ----------
        vectors : dict
            The vectors to be projected in the Krylov space.
            The keys of the dict are the identifiers of these vectors.The
            values of this dict should be np.array and of shape (self._dim, 1).
        procs_num : int, keyword-only, optional
            The number of process to use.
            It is recommended to set this parameter to be integers which can be
            divided exactly by the number of vectors for load balancing of every
            process, also, this parameter should not be too large because of the
            bandwidth limit of the RAM.
            default: 1

        Returns
        -------
        HM_projs : dict
            The representations of HM in these Krylov space.
        vectors_projs : dict
            The projections of the vectors in these Krylov space.
        """

        if not isinstance(vectors, dict):
            raise TypeError("The input vectors parameter is not dict.")
        else:
            for key in vectors.keys():
                if vectors[key].shape != (self._dim, 1):
                    raise ValueError("The wrong vector dimension!")
        if not isinstance(procs_num, int) or procs_num <= 0:
            raise ValueError("The procs_num parameter should be positive integer.")

        tasks = list(vectors.keys())
        HM_projs = dict()
        vectors_projs = dict()
        if procs_num == 1:
            for count, task in enumerate(tasks):
                t0 = time()
                HM_proj, vectors_proj = self.projection(vectors, vectors[task])
                HM_projs[task] = HM_proj
                vectors_projs[task] = vectors_proj
                t1 = time()
                info = "The time spend on the {0}th starting vector: {1:.4f}s"
                info = info.format(count, t1-t0)
                print(info, flush=True)
                print("=" * len(info), flush=True)
        else:
            with mp.Manager() as manager:
                HM_proxy = manager.dict()
                vectors_proxy = manager.dict()
                kwargs = {"vectors": vectors, "HM_proxy": HM_proxy,
                          "vectors_proxy": vectors_proxy}

                procs = []
                for proc_index in range(procs_num):
                    keys = tasks[proc_index::procs_num]
                    proc = mp.Process(target=self._pprojection,
                                        args=(keys, ), kwargs=kwargs)
                    proc.start()
                    procs.append(proc)

                for proc in procs:
                    proc.join()

                for key in tasks:
                    HM_projs[key] = HM_proxy[key]
                    vectors_projs[key] = vectors_proxy[key]
        return HM_projs, vectors_projs
    # }}}
# }}}
