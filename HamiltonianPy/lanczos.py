"""
The implemetation of the Lanczos Algorithm.
"""

from collections import OrderedDict
from numpy.linalg import eigvalsh, norm
from scipy.sparse import csr_matrix, isspmatrix_csr
from time import time

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
    matrix: csr_matrix
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

    Method:
    ------
    __init__(matrix, v0=None, step=200, tol=1e-12, vtype='rd')
    __call__(vecs)

    gs()
    krylov_basis()
    projection(vecs, v_init)
    """

    def __init__(self, matrix, v0=None, step=200, tol=1e-12, vtype='rd'):# {{{
        """
        Inits Lanczos class.

        See also the docstring of this class!
        """

        if not isspmatrix_csr(matrix):
            raise TypeError("The input matrix is not of csr type!")
        if (matrix - matrix.conjugate().transpose()).count_nonzero() != 0:
            raise TypeError("The input matrix is not a Hermitian matrix!")

        if v0 is None:# {{{
            if vtype.lower() == 'rd':
                v0 = np.zeros((matrix.shape[0], 1), dtype=np.complex128)
                v0[:, 0] = np.random.rand(matrix.shape[0])[:]
            else:
                v0 = np.ones((matrix.shape[0], 1), dtype=np.complex128)
            v0 /= norm(v0)
        else:
            if v0.shape != (matrix.shape[0], 1):
                raise ValueError("The dimension of v0 and matrix does not match!")
            v0_norm = norm(v0)
            if v0_norm < self.error:  #Ensure the input v0 is not a zero vector!
                raise ValueError("The norm of v0 is to small!")
            v0 = v0 / v0_norm# }}}

        self.v0 = v0
        self.M = matrix
        self.dim = matrix.shape[0]
        self.step = step
        self.tol = tol
    # }}}

    def gs(self):# {{{
        """
        Find samllest eigenvalue of the input matrix.

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

        v_old = np.zeros((self.dim, 1), dtype=np.float64)
        v = self.v0
        c0 = 0.0
        v_new = self.M.dot(v)
        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []

        E = c1 
        for i in range(self.dim):
            v_new -= c1 * v
            v_new -= v_old
            #The v_old is not needed anymore, delete it to save memory!
            del v_old
            c0 = norm(v_new)
            if c0 < VIEW_AS_ZERO:
                raise ConvergenceError("Got an invariant subspace!")
                #break
            v_new /= c0
            v_old = v
            v_old *= c0
            v = v_new
            v_new = self.M.dot(v)
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

        v_old = np.zeros((self.dim, 1), dtype=np.float64)
        v = self.v0
        c0 = 0.0
        v_new = self.M.dot(v)
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
            v_new = self.M.dot(v)
            c1 = np.vdot(v, v_new)
            krylov_basis = np.concatenate((krylov_basis, v), axis=1) 
            c1s.append(c1)
            c0s.append(c0)
        tri = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
        return tri, krylov_basis
    # }}}

    def projection(self, vecs, v_init):# {{{
        """
        Return the representation of vectors and matrix in the Krylov space.
        
        The projected matrix has the tridiagonal form and is return as the tri.
        The representaion of the vectors in the Krylov space are stored as a
        ndarray with dimension(self.step, stats_num). In the calculation, every
        column store a vector for effience.

        Parameter:
        ----------
        vecs: OrderedDict
            The vectors to be represented in the Krylov space.
            The keys of the dict are the identifieris of these vectors.
            The values of this dict should be ndarray and of shape (self.dim,1).
        v_init: ndarray
            Starting vector for iteration.

        Return:
        -------
        res_tri: ndarray
            The tridiagonal representaion of the matrix.
        res_vecs: ndarray
            The representaion of the vectors in the Krylov space.
        """

        v_norm = norm(v_init)
        if v_norm < VIEW_AS_ZERO:
            raise ValueError("The norm of v_init is to small!")

        v_old = np.zeros((self.dim, 1), dtype=np.float64)
        v = v_init / v_norm
        v_new = self.M.dot(v)
        c1 = np.vdot(v, v_new)
        c1s = [c1]
        c0s = []
        
        keys = vecs.keys()
        vecs_repr = OrderedDict()
        for key in keys:
            vecs_repr[key] = [np.vdot(v, vecs[key])]
        
        for i in range(1, self.step):
            v_new -= c1 * v
            v_new -= v_old
            #The v_old is not needed anymore, delete it to save memory!
            del v_old
            c0 = norm(v_new)
            if c0 < VIEW_AS_ZERO:
                break
            v_new /= c0

            v_old = v
            v_old *= c0
            v = v_new
            v_new = self.M.dot(v)
            c1 = np.vdot(v, v_new)

            c1s.append(c1)
            c0s.append(c0)

            for key in keys:
                vecs_repr[key].append(np.vdot(v, vecs[key]))

        res_vecs = OrderedDict()
        for key in keys:
            res_vecs[key] = np.array([vecs_repr[key]]).T
        res_tri = np.diag(c1s, 0) + np.diag(c0s, -1) + np.diag(c0s, 1)
        return res_tri, res_vecs
    # }}}

    def __call__(self, vecs):# {{{
        """
        Return the representation of vectors and matrix in the Krylov space.
        
        Choosing every vector in vecs as the initial vector, this method
        generate the krylov space, calculate the representation of matrix and
        all vectors in this space.

        Parameter:
        ----------
        vecs: OrderedDict
            The vectors to be represented in the Krylov space.
            The keys of the dict are the identifieris of these vectors.
            The values of this dict should be ndarray and of shape (self.dim,1).

        Return:
        -------
        res_tris: ndarray
            The tridiagonal representaion of the matrix.
        res_vecs_repr: ndarray
            The representaion of the vectors in the Krylov space.
        """

        keys = vecs.keys()
        for key in keys:
            if vecs[key].shape != (self.dim, 1):
                raise TypeError("The wrong vector dimension!")

        res_tris = OrderedDict() 
        res_vecs = OrderedDict()
        for count, key in enumerate(keys):
            t0 = time()
            v_init = vecs[key]
            res_tris[key], res_vecs[key] = self.projection(vecs, v_init)
            t1 = time()
            info = "The time spend on the {0}th ket: {1:.4f}."
            info = info.format(count, t1-t0)
            print(info)
            print("=" * len(info))
        return res_tris, res_vecs
    # }}}
# }}}
