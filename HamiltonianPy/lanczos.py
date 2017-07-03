"""
The implemetation of the Lanczos Algorithm.
"""

from numpy.linalg import eigvalsh, norm
from scipy.sparse import csr_matrix, isspmatrix_csr
from time import time

import multiprocessing as mp
import numpy as np

from HamiltonianPy.constant import VIEW_AS_ZERO
from HamiltonianPy.exception import ConvergenceError

__all__ = ['Lanczos']

class Lanczos:# {{{
    """
    Implementation of the Lanczos Algorithm.

    This class provide method to build a projection of the large compressed
    sparse row matrix onto a Krylov subspace. Generate the representation of
    vectors and matrix in this Krylov subspace.

    Attributes:
    -----------
    HM: csr_matrix
        The large compressed sparse row matrix to be processed. It must be a
        Hermitian matrix.
    dim: int
        The dimension of the csr_matrix.
    step: int
        The maximum number of lanczos iteration (stopping critertion).
    tol: float
        Relative accuracy for eigenvalues (stopping criterion).
    v0: ndarray, optional
        Starting vector for iteration, v0 is assumed of shape (dim,1).
        default: random

    Methods:
    -------
    Special methods:
        __init__(HM, *, v0=None, step=200, tol=1e-12, vtype='rd')
        __call__(vectors)
    General methods:
        gs()
        krylov_basis()
        projection(vectors, v_init)
    """

    def __init__(self, HM, *, v0=None, step=200, tol=1e-12, vtype='rd'):# {{{
        """
        Initialize the instance of this class.

        See also the document of this class!
        """

        if not isspmatrix_csr(HM):
            raise TypeError("The input HM matrix is not a csr_matrix!")
        elif ((HM - HM.getH()) > VIEW_AS_ZERO).count_nonzero() != 0:
            raise TypeError("The input HM matrix is not an Hermitian matrix!")
        else:
            self.HM = HM
            self.dim = HM.shape[0]
        
        v0_shape = (HM.shape[0], 1)
        if v0 is None:
            if vtype.lower() in ("rd", "random"):
                v0 = np.random.random_sample(size=v0_shape)
            else:
                v0 = np.ones(shape=v0_shape)
            v0 /= norm(v0)
        else:
            if v0.shape != v0_shape:
                raise ValueError("The dimension of v0 and HM does not match!")
            v0_norm = norm(v0)
            #Ensure the input v0 is not a zero vector!
            if v0_norm < VIEW_AS_ZERO:
                raise ValueError("The given v0 is a zero vector!")
            v0 = v0 / v0_norm
        self.v0 = v0

        if isinstance(step, int) and step > 0:
            self.step = step
        else:
            raise ValueError("The step parameter should be positive integer.")

        if isinstance(tol, float) and tol > 0:
            self.tol = tol
        else:
            raise ValueError("The tol parameter should be positive float.")
    # }}}

    def gs(self):# {{{
        """
        Find samllest eigenvalue of the HM matrix.

        This method calculate the smallest eigenvalue of the input spasre 
        matrix with the given precision specified by the tol parameter.

        Return:
        -------
        val: float
            The smallest eigenvalue of the matrix.

        Raise:
        ------
        ConvergenceError:
            Got an invariant subspace or Krylov 
            space larger than the orginal space.
        """

        v_old = 0.0
        v = np.array(self.v0[:])
        v_new = self.HM.dot(v)
        
        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []

        E = c1 
        for i in range(self.dim):
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
            v_new = self.HM.dot(v)
            c1 = np.vdot(v, v_new)
            c1s.append(c1)
            c0s.append(c0)

            tri = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
            val = eigvalsh(tri)[0]
            if abs(val - E) < self.tol:
                return val
            else:
                E = val

        raise ConvergenceError("Got Krylov subspace larger "
                               "than the original space.")
    # }}}

    def krylov_basis(self):# {{{
        """
        Generate the basis of the Krylov subspace.

        This method calculate the basis of the Krylov subspace as well as the
        representaion of the matrix in the base. 
        The projected matrix has the tridiagonal form and is return as the tri.
        The basis are stored as a ndarray with dimension (self.step, self.dim).
        Every row of the array represents a basis vector.

        Return:
        -------
        tri: ndarray
            The projection of the input matrix in the Krylov space.
        krylov_basis: ndarray
            The basis of the krylov space.
        """

        v_old = 0.0
        v = np.array(self.v0[:])
        v_new = self.HM.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []
        krylov_basis = v

        for i in range(1, self.step):
            v_new -= c1 * v
            v_new -= v_old

            c0 = norm(v_new)
            if c0 < VIEW_AS_ZERO:
                break
            
            v_new /= c0
            v_old = v
            v_old *= c0
            v = v_new
            v_new = self.HM.dot(v)
            c1 = np.vdot(v, v_new)
            krylov_basis = np.concatenate((krylov_basis, v), axis=1)
            c1s.append(c1)
            c0s.append(c0)
        tri = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
        return tri, krylov_basis
    # }}}

    def projection(self, vectors, v_init):# {{{
        """
        Return the representation of vectors and matrix in the Krylov space.
        
        The representation of self.HM in Krylov space is a tridiagonal matrix
        and is returned as a np.array. The representaion of the given vectors 
        in the Krylov space are stored as OrderedDict. The keys of the dict are
        the identifiers of these vectors and the values should be np.array with
        shape (self.step, 1).

        Parameter:
        ----------
        vectors: dict
            The vectors to be projected in the Krylov space. The keys of the 
            dict are the identifiers of these vectors. The values of this dict 
            should be np.array and of shape (self.dim, 1).
        v_init: ndarray
            Starting vector for iteration.

        Return:
        -------
        HM_proj: np.array
            The tridiagonal representaion of the matrix.
        vectors_proj: dict
            The projection of the vectors in the Krylov space.
        """

        v_norm = norm(v_init)
        if v_norm < VIEW_AS_ZERO:
            raise ValueError("The given v_init is a zero vector.")

        v_old = 0.0
        v = v_init / v_norm
        v_new = self.HM.dot(v)

        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []
        
        keys = vectors.keys()
        temp = dict()
        for key in keys:
            temp[key] = [np.vdot(v, vectors[key])]
        
        for i in range(1, self.step):
            v_new -= c1 * v
            v_new -= v_old

            c0 = norm(v_new)
            if c0 < VIEW_AS_ZERO:
                break
            v_new /= c0

            v_old = v
            v_old *= c0
            v = v_new
            v_new = self.HM.dot(v)
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

    def _pprojection(self, key, vectors, HM_proxy, vectors_proxy):# {{{
        """
        Used for multi-process parallel.
        """

        t0 = time()
        HM_proj, vectors_proj = self.projection(vectors, vectors[key])
        HM_proxy[key] = HM_proj
        vectors_proxy[key] = vectors_proj
        t1 = time()
        print("The current process: ", mp.current_process())
        print("The living time of this process: {0:.4f}s".format(t1 - t0))
        print("=" * 60)
    # }}}

    def __call__(self, vectors, *, procs_num=1):# {{{
        """
        Return the representation of vectors and matrix in the Krylov space.
        
        Choosing every vector in vectors as the initial vector, this method
        generate the krylov space, calculate the representation of the matrix 
        and all vectors in this space.

        Parameter:
        ----------
        vectors: dict
            The vectors to be projected in the Krylov space.
            The keys of the dict are the identifiers of these vectors.The 
            values of this dict should be np.array and of shape (self.dim, 1).
        procs_num: int, optional
            The number of process to use.
            default: 1

        Return:
        -------
        HM_projs: dict
            The representations of HM in the Krylov space.
        vectors_projs: dict
            The projections of the vectors in the Krylov space.
        """

        if not isinstance(vectors, dict):
            raise TypeError("The input vectors parameter is not OrderedDict.")
        else:
            for key in vectors.keys():
                if vectors[key].shape != (self.dim, 1):
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
                info = "The time spend on the {0}th ket: {1:.4f}."
                info = info.format(count, t1-t0)
                print(info)
                print("=" * len(info))
        else:
            with mp.Manager() as manager:
                HM_proxy = manager.dict()
                vectors_proxy = manager.dict()
                kwargs = {"vectors": vectors, "HM_proxy": HM_proxy,
                          "vectors_proxy": vectors_proxy}

                procs_alive = set()
                procs_dead = set()
                for i in range(procs_num):
                    if len(tasks) != 0:
                        p = mp.Process(target=self._pprojection,
                                args=(tasks.pop(), ), kwargs=kwargs)
                        p.start()
                        procs_alive.add(p)


                while len(procs_alive) != 0:
                    procs_new = set()
                    for p in procs_alive:
                        if not p.is_alive():
                            procs_dead.add(p)
                            p.join()
                            if len(tasks) != 0:
                                q = mp.Process(target=self._pprojection,
                                        args=(tasks.pop(), ), kwargs=kwargs)
                                q.start()
                                procs_new.add(q)
                    procs_alive.difference_update(procs_dead)
                    procs_alive.update(procs_new)

                for key in vectors.keys():
                    HM_projs[key] = HM_proxy[key]
                    vectors_projs[key] = vectors_proxy[key]

        return HM_projs, vectors_projs
    # }}}
# }}}
