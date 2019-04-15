"""
Description of common lattice with translation symmetry
"""


__all__ = [
    "Lattice",
    "lattice_generator",
    "special_cluster",
    "set_float_point_precision",
]


from itertools import product
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist

import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy.bond import Bond


# Useful global constant
_ZOOM = 10000
_VIEW_AS_ZERO = 1e-4
################################################################################


def set_float_point_precision(precision):
    """
    Set the precision for processing float point number

    The float-point precision affects the internal implementation of the
    Lattice class. If you want to change the default value, you must call
    this function before creating any Lattice instance.

    Parameters
    ----------
    precision : int
        The number of digits precision after the decimal point
    """

    assert isinstance(precision, int) and precision >= 0

    global  _ZOOM, _VIEW_AS_ZERO
    _ZOOM = 10 ** precision
    _VIEW_AS_ZERO = 10 ** (-precision)


class Lattice:
    """
    Unified description of a cluster with translation symmetry

    Attributes
    ----------
    points : array-like, shape (n, m)
        A collection of the coordinates of the points in the cluster and
        every row represents a point.
    vectors : array-like, shape (k, m)
        A collection of the translation vectors of the lattice and every row
        represents a translation vector.
    bs : array-like, shape (k, m)
        A collection of the base vectors of the reciprocal space and every
        row represents a vector.
    point_num : int
        The number of point in the cluster.
    space_dim : int
        The dimension of the space in which the points are described.
    trans_dim : int
        The number of linear independent directions along which the lattice
        is translational invariant.
    name : str
        The name of this cluster
    """

    def __init__(self, points, vectors, name=""):
        """
        Customize the newly created instance

        Parameters
        ----------
        points : array-like, shape (n, m)
            A collection of the coordinates of the points in the cluster and
            every row represents a point. There should be no duplicate
            points in the collection.
        vectors : array-like, shape (k, m)
            A collection of the translation vectors of the lattice and every
            row represents a translation vector. There should be no duplicate
            vectors in the collection.
        name : str, optional
            The name of this cluster.
            default: ""(empty string)
        """

        assert isinstance(points, np.ndarray) and points.ndim == 2
        assert isinstance(vectors, np.ndarray) and vectors.ndim == 2

        point_num, space_dim = points.shape
        trans_dim, tmp = vectors.shape
        if space_dim > 3:
            raise ValueError("Not supported space dimension.")

        if trans_dim > space_dim:
            raise ValueError(
                "The number of translation vectors should not be greater "
                "than the space dimension!"
            )
        if tmp != space_dim:
            raise ValueError(
                "The translation vectors should have the same space "
                "dimension as the points!"
            )

        # Check is there duplicate translation vectors in the given `vectors`
        if np.any(pdist(vectors) < _VIEW_AS_ZERO):
            raise ValueError(
                "There are duplicate translation vectors in the given `vectors`"
            )

        # Check is there duplicate points in the given `points`
        tmp = pdist(points)
        if np.any(tmp < _VIEW_AS_ZERO):
            raise ValueError("There are duplicate points in the given `points`")
        else:
            dists = np.insert(np.unique(np.ceil(tmp * _ZOOM)) / _ZOOM, 0, 0.0)

        # Cache dists as `_all_distances attribute` for later use
        self._all_distances = dists

        self.name = name
        self._point_num = point_num
        self._space_dim = space_dim
        self._trans_dim = trans_dim

        self._points = np.array(points, copy=True)
        self._vectors = np.array(vectors, copy=True)
        self._points.setflags(write=False)
        self._vectors.setflags(write=False)

        # bs = np.matmul(coeff, As)
        # np.matmul(bs, As.T) == 2 * np.pi * I
        self._bs = 2 * np.pi * np.linalg.solve(
            np.matmul(vectors, vectors.T), vectors
        )
        self._bs.setflags(write=False)

    @property
    def points(self):
        """
        The `points` attribute of the instance
        """

        return np.array(self._points, copy=True)

    @property
    def vectors(self):
        """
        The `vectors` attribute of the instance
        """

        return np.array(self._vectors, copy=True)

    @property
    def bs(self):
        """
        The `bs` attribute of the instance
        """

        return np.array(self._bs, copy=True)

    @property
    def point_num(self):
        """
        The `point_num` attribute of the instance
        """

        return self._point_num

    @property
    def space_dim(self):# {{{
        """
        The `space_dim` attribute of the instance
        """

        return self._space_dim

    @property
    def trans_dim(self):
        """
        The `trans_dim` attribute of the instance
        """

        return self._trans_dim

    def __str__(self):
        """
        Return a string that describes the content of the instance
        """

        info = "Point number: {0}\n".format(self._point_num)
        info += "Space dimension: {0}\n".format(self._space_dim)
        info += "Translation dimension: {0}\n".format(self._trans_dim)
        info += "points:\n" + str(self._points) + '\n'
        info += "vectors:\n" + str(self._vectors) + '\n'
        info += "bs:\n" + str(self._bs) + '\n'
        return info

    def getIndex(self, site, *, fold=False):
        """
        Return the index corresponding to the given site

        Parameters
        ----------
        site : np.ndarray
            The site whose index is queried
        fold : boolean, keyword-only, optional
            Whether to fold the given site back to the cluster
            default: False

        Returns
        -------
        index : int
            The index of the given site if it belong the cluster
        """

        assert isinstance(site, np.ndarray) and site.shape == (self._space_dim,)

        if fold:
            site, trash = self.decompose(site)

        dist, index = cKDTree(self._points).query(site)
        if dist < _VIEW_AS_ZERO:
            return index
        else:
            raise KeyError("The given site does not belong the lattice")

    def getSite(self, index):
        """
        Return the site corresponding to the given index.

        Parameters
        ----------
        index : int
            The index of the site to query.

        Returns
        -------
        res : np.array
            The site corresponding to the index.
        """

        assert isinstance(index, int), "The `index` should be an integer"

        if index < 0:
            raise ValueError("The `index` should be none negative integer")
        elif index < self._point_num:
            return np.array(self._points[index], copy=True)
        else:
            raise ValueError("The `index` is larger than the number of sites")

    def _guess_scope(self, ref_distance):
        repeat = self._trans_dim
        scope = -1
        while True:
            inner_mesh = product(range(-scope, scope+1), repeat=repeat)
            outer_mesh = product(range(-scope-1, scope+2), repeat=repeat)
            edge_dRs = np.matmul(
                list(set(outer_mesh).difference(set(inner_mesh))), self._vectors
            )
            edge_points = np.reshape(
                edge_dRs[:, np.newaxis, :] + self._points,
                newshape=(-1, self._space_dim)
            )
            if np.min(cdist(edge_points, self._points)) > ref_distance:
                break
            else:
                scope += 1
        return scope

    def _searching(self, database, displaces):
        matches = np.nonzero(database.query(displaces)[0] < _VIEW_AS_ZERO)[0]
        if len(matches) == 1:
            index = matches[0]
            return np.array(self._points[index], copy=True), displaces[index]
        else:
            return None

    def decompose(self, site):
        """
        Decompose the given site with respect to the translation vectors and
        points in the cluster.

        This method describes how to arrive the given site through translating
        one point of the cluster along the translation vectors. Fractional
        translation along any translation vectors is not allowed.

        The basic idea is that: we first calculate a collection of possible 
        translations for the given site, and then for every point in the
        cluster we search the collection to determine if we can reach the given
        site.

        Parameters
        ----------
        site : ndarray
            The site to be decomposed

        Returns
        -------
        site : ndarray
            The equivalent point in the cluster
        dR : ndarray
            The translation vector
        """

        assert isinstance(site, np.ndarray) and site.shape == (self.space_dim,)

        if not hasattr(self, "_dRs_DataBase"):
            mesh = product([-1, 0, 1], repeat=self._trans_dim)
            self._dRs_DataBase = cKDTree(np.matmul(list(mesh), self._vectors))

        # First search in a small area, if success then return the result,
        #  if failed, calculate the possible scope and search in that area
        all_displaces = site - self._points
        dRs_DataBase = self._dRs_DataBase
        res = self._searching(dRs_DataBase, all_displaces)
        if res is not None:
            return res

        scope = self._guess_scope(
            np.max(np.linalg.norm(all_displaces, axis=-1))
        )
        mesh = product(range(-scope, scope + 1), repeat=self._trans_dim)
        dRs_DataBase = cKDTree(np.matmul(list(mesh), self._vectors))
        res = self._searching(dRs_DataBase, all_displaces)
        if res is not None:
            return res
        else:
            raise RuntimeError(
                "Failed to decompose the input site. It might not belong "
                "the lattice. Check carefully!"
            )

    def show(self, scope=0, savefig=False):
        """
        Show all the points in the cluster

        Parameters
        ----------
        scope : int, optional
            Determine the number of clusters to show
            default: 0
        savefig : boolean, optional
            Determine whether to save or display the figure
            default: False
        """

        assert isinstance(scope, int) and scope >= 0

        if self._space_dim >= 3:
            raise RuntimeError("Not supported space dimension!")

        clusters = [
            self._points + np.matmul(tmp, self._vectors)
            for tmp in product(range(-scope, scope+1), repeat=self._trans_dim)
        ]

        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.set_aspect("equal")
        if self._space_dim == 1:
            ys = np.zeros(shape=self._points.shape)
            for cluster in clusters:
                ax.plot(cluster, ys, marker="o", ls="None", ms=8)
        else:
            for cluster in clusters:
                ax.plot(
                    cluster[:, 0], cluster[:, 1], marker="o", ls="None", ms=8
                )

        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        half = max(right - left, top - bottom) / 2
        x_center = (right + left) / 2
        y_center = (top + bottom) / 2
        ax.set_xlim(left=x_center-half, right=x_center+half)
        ax.set_ylim(bottom=y_center-half, top=y_center+half)
        ax.set_title("{0}_show-scope={1}".format(self.name, scope))

        if savefig:
            name = Path(".") / "{0}_scope={1}.png".format(self.name, scope)
            fig.savefig(name)
            plt.close()
            print("The figure has been saved to:\n{0}".format(name.resolve()))
        else:
            plt.show()

    def _sites_factory(self, scope, no_inversion=True):
        mesh = product(range(-scope, scope+1), repeat=self._trans_dim)
        if no_inversion:
            translations = set()
            for config in mesh:
                if tuple(-x for x in config) not in translations:
                    translations.add(config)
        else:
            translations = set(mesh)

        origin = (0, ) * self._trans_dim
        translations.discard(origin)
        dRs = np.matmul([origin, ] + list(translations), self._vectors)
        sites = np.reshape(
            dRs[:, np.newaxis, :] + self._points, newshape=(-1, self._space_dim)
        )
        return sites

    def neighbor_distance(self, nth):
        """
        Return the `nth` neighbor distance

        In this method, nth=0 represents onsite, nth=1 nearest-neighbor,
        nth=2 next-nearest neighbor, etc.

        Parameters
        ----------
        nth : int
            Specify which neighbor distance to return.

        Returns
        -------
        result : float
            The nth neighbor distance.
        """

        assert isinstance(nth, int) and nth >= 0

        if len(self._all_distances) <= nth:
            sites = self._sites_factory(scope=nth, no_inversion=False)
            self._all_distances = np.insert(
                np.unique(np.ceil(pdist(sites) * _ZOOM)) / _ZOOM, 0, 0.0
            )
        return self._all_distances[nth]

    def bonds(self, nth, *, only=True, fold=False):
        """
        Return all bonds specified by the given parameters.

        Parameters
        ----------
        nth : int
            0 means onsite, 1 represents nearest neighbor, 2 represents
            next-nearest neighbor, etc.
        only : boolean, keyword-only, optional
            If True, only these bonds which length equals to the nth neighbor
            distance. If False, all the bonds which length equal or less than
            the nth neighbor distance.
            default: True
        fold : boolean, keyword-only, optional
            Whether to fold the boundary bond
            default: False

        Returns
        -------
        intra : list of Bond
            A collection of the bonds that both endpoints belong the cluster.
        inter : list of Bond
            A collection of the bonds that one endpoints belong the cluster and
            the other does not.
        """

        assert isinstance(nth, int) and nth >= 0

        if nth == 0:
            intra = [Bond(p, p, directional=True) for p in self._points]
            return intra, None

        judge = self.neighbor_distance(nth=nth)
        scope = self._guess_scope(judge)
        sites = self._sites_factory(scope=scope, no_inversion=True)

        tree = cKDTree(sites)
        pairs_outer = tree.query_pairs(r=judge + _VIEW_AS_ZERO)
        if only and nth > 1:
            judge = self.neighbor_distance(nth=nth - 1)
            pairs_inner = tree.query_pairs(r=judge + _VIEW_AS_ZERO)
        else:
            pairs_inner = set()

        pairs = pairs_outer.difference(pairs_inner)

        intra = []
        inter = []
        # index0 < index1 is granted by the cKDTree.query_pairs() method
        for index0, index1 in pairs:
            p0, p1 = sites[[index0, index1]]
            if index1 < self._point_num:
                intra.append(Bond(p0, p1, directional=True))
            elif index0 < self.point_num:
                p1_eqv, trash = self.decompose(p1)
                if fold:
                    bond = Bond(p0, p1_eqv, directional=True)
                else:
                    bond = Bond(p0, p1, directional=True)
                inter.append(bond)
        return intra, inter

    def in_cluster(self, site):
        """
        Determine whether the input site belong the cluster.

        Parameters
        ----------
        site : ndarray
            The site which is to be checked.

        Returns
        -------
        result : boolean
            If site belong to the initial cluster return true, else false.
        """

        try:
            self.getIndex(site, fold=False)
            return True
        except KeyError:
            return False


# database for some commonly used cluster
_dtype = np.float64

_chain_cell_info = {
    "points": np.array([[0.0]], dtype=_dtype),
    "vectors": np.array([[1.0]], dtype=_dtype)
}

_square_cell_info = {
    "points": np.array([[0.0, 0.0]], dtype=_dtype),
    "vectors": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=_dtype)
}

_triangle_cell_info = {
    "points": np.array([[0.0, 0.0]], dtype=_dtype),
    "vectors": np.array([[1.0, 0.0], [0.5, np.sqrt(3)/2]], dtype=_dtype)
}

_honeycomb_cell_info = {
    "points": np.array([[0.0, 0.0], [0.0, 1/np.sqrt(3)]], dtype=_dtype),
    "vectors": np.array([[1.0, 0.0], [0.5, np.sqrt(3)/2]], dtype=_dtype)
}

_kagome_cell_info = {
    "points": np.array(
        [[0, 0], [0.25, np.sqrt(3)/4], [0.5, 0.0]], dtype=_dtype
    ),
    "vectors": np.array([[1, 0], [0.5, np.sqrt(3)/2]], dtype=_dtype)
}

_cubic_cell_info = {
    "points": np.array([[0.0, 0.0, 0.0]], dtype=_dtype),
    "vectors": np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=_dtype
    )
}

_common_cells_info = {
    "chain": _chain_cell_info,
    "square": _square_cell_info,
    "triangle": _triangle_cell_info,
    "honeycomb": _honeycomb_cell_info,
    "kagome": _kagome_cell_info,
    "cubic": _cubic_cell_info,
}

_square_cross_info = {
    "points": np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0],
         [2.0, 2.0], [2.0, 1.0], [3.0, 1.0], [3.0, 0.0],
         [2.0, 0.0], [2.0, -1.0], [1.0, -1.0], [1.0, 0.0]],
        dtype=_dtype
    ),
    "vectors": np.array([[3.0, 2.0], [3.0, -2.0]], dtype=_dtype)
}

_square_z_info = {
    "points": np.array(
        [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 2.0], [1.0, 3.0],
         [2.0, 3.0], [2.0, 2.0], [2.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        dtype=_dtype
    ),
    "vectors": np.array([[3.0, 1.0], [1.0, -3.0]], dtype=_dtype)
}

_triangle_star_info = {
    "points": np.array(
        [[0.0, 0.0],
         [0.0, -np.sqrt(3)], [-0.5, -np.sqrt(3)/2], [-1.5, -np.sqrt(3)/2],
         [-1.0, 0.0], [-1.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
         [0.0, np.sqrt(3)], [0.5, np.sqrt(3)/2], [1.5, np.sqrt(3)/2],
         [1.0, 0.0], [1.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]],
        dtype=_dtype
    ),
    "vectors": np.array(
        [[3.5, np.sqrt(3)/2], [2.5, -1.5*np.sqrt(3)]], dtype=_dtype
    )
}

_honeycomb_benzene_info = {
    "points": np.array(
        [[0.0, 0.0], [0.0, 1/np.sqrt(3)], [0.5, np.sqrt(3)/2],
         [1.0, 1/np.sqrt(3)], [1.0, 0.0], [0.5, -0.5/np.sqrt(3)]],
        dtype=_dtype
    ),
    "vectors": np.array(
        [[1.5, np.sqrt(3)/2], [1.5, -np.sqrt(3)/2]], dtype=_dtype
    )
}

_honeycomb_diphenyl_info = {
    "points": np.array(
        [[0.0, 0.0], [0.0, 1/np.sqrt(3)], [0.5, np.sqrt(3)/2],
         [1.0, 1/np.sqrt(3)], [1.5, np.sqrt(3)/2], [2.0, 1/np.sqrt(3)],
         [2.0, 0.0], [1.5, -0.5/np.sqrt(3)], [1.0, 0.0],
         [0.5, -0.5/np.sqrt(3)]],
        dtype=_dtype
    ),
    "vectors": np.array(
        [[2.5, np.sqrt(3)/2], [2.5, -np.sqrt(3)/2]], dtype=_dtype
    )
}

_honeycomb_gear_info = {
    "points": np.array(
        [[0.0, 0.0], [0.0, 1/np.sqrt(3)], [0.5, np.sqrt(3)/2],
         [0.5, 2.5/np.sqrt(3)], [1.0, np.sqrt(3)], [1.5, 2.5/np.sqrt(3)],
         [2.0, np.sqrt(3)], [2.5, 2.5/np.sqrt(3)], [2.5, np.sqrt(3)/2],
         [3.0, 1/np.sqrt(3)], [3.0, 0.0], [2.5, -0.5/np.sqrt(3)],
         [2.5, -np.sqrt(3)/2], [2.0, -2/np.sqrt(3)], [1.5, -np.sqrt(3)/2],
         [1.0, -2/np.sqrt(3)], [0.5, -np.sqrt(3)/2], [0.5, -0.5/np.sqrt(3)],
         [1.0, 0.0], [1.0, 1/np.sqrt(3)], [1.5, np.sqrt(3)/2],
         [2.0, 1/np.sqrt(3)], [2.0, 0.0], [1.5, -0.5/np.sqrt(3)]],
        dtype=_dtype
    ),
    "vectors": np.array([[3, np.sqrt(3)], [3, -np.sqrt(3)]], dtype=_dtype)
}

_special_clusters_info = {
    "square_cross": _square_cross_info,
    "square_12": _square_cross_info,
    "cross": _square_cross_info,

    "square_z": _square_z_info,
    "square_10": _square_z_info,
    "z": _square_z_info,

    "triangle_star": _triangle_star_info,
    "triangle_13": _triangle_star_info,
    "star": _triangle_star_info,

    "honeycomb_benzene": _honeycomb_benzene_info,
    "honeycomb_6": _honeycomb_benzene_info,
    "benzene": _honeycomb_benzene_info,

    "honeycomb_diphenyl": _honeycomb_diphenyl_info,
    "honeycomb_10": _honeycomb_diphenyl_info,
    "diphenyl": _honeycomb_diphenyl_info,

    "honeycomb_gear": _honeycomb_gear_info,
    "honeycomb_24": _honeycomb_gear_info,
    "gear": _honeycomb_gear_info,
}


def special_cluster(which):
    """
    Generating some special cluster

    Parameters
    ----------
    which : str
        Which special lattice to generate
        Currently supported special lattice:
            "square_cross" | "square_z" | "triangle_star" |
            "honeycomb_benzene" | "honeycomb_diphenyl" | "honeycomb_gear"
        Alias:
            "square_cross" | "square_12" | "cross";
            "square_z" | "square_10" | "z";
            "triangle_star" | "triangle_13" | "star";
            "honeycomb_benzene" | "honeycomb_6"| "benzene";
            "honeycomb_diphenyl" | "honeycomb_10" | "diphenyl";
            "honeycomb_gear" | "honeycomb_24" | "gear"

    Returns
    -------
    res : Lattice
        The corresponding cluster
    """

    try:
        cluster_info = _special_clusters_info[which]
        return Lattice(**cluster_info, name=which)
    except KeyError:
        raise KeyError("Unrecognized special lattice name!")


def lattice_generator(which, num0=1, num1=1, num2=1):
    """
    Generating a common cluster with translation symmetry

    Parameters
    ----------
    which : str
        Which  type of lattice to generate.
        Legal value:
            "chain" | "square" | "triangle" | "honeycomb" | "kagome" | "cubic"
    num0 : int, optional
        The number of unit cell along the first translation vector
        default: 1
    num1 : int, optional
        The number of unit cell along the second translation vector. It only
        takes effect for 2D and 3D lattice.
        default: 1
    num2 : int, optional
        The number of unit cell along the second translation vector. It only
        takes effect for 3D lattice.
        default : 1

    Returns
    -------
    res : Lattice
        The corresponding cluster with translation symmetry.
    """

    assert isinstance(num0, int) and num0 >= 1
    assert isinstance(num1, int) and num1 >= 1
    assert isinstance(num2, int) and num2 >= 1

    try:
        cell_info = _common_cells_info[which]
        cell_points = cell_info["points"]
        cell_vectors = cell_info["vectors"]
    except KeyError:
        raise KeyError("Unrecognized lattice type!")

    if which == "chain":
        if num0 == 1:
            return Lattice(**cell_info, name="chain_cell")
        else:
            name = "chain({0})".format(num0)
            vectors = cell_vectors * np.array([[num0]])
            mesh = product(range(num0))
    elif which == "cubic":
        if num0 == 1 and num1 == 1 and num2 == 1:
            return Lattice(**cell_info, name="cubic_cell")
        else:
            name = "cubic({0},{1},{2})".format(num0, num1, num2)
            vectors = cell_vectors * np.array([[num0], [num1], [num2]])
            mesh = product(range(num0), range(num1), range(num2))
    else:
        if num0 == 1 and num1 == 1:
            return Lattice(**cell_info, name="{0}_cell".format(which))
        else:
            name = "{0}({1},{2})".format(which, num0, num1)
            vectors = cell_vectors * np.array([[num0], [num1]])
            mesh = product(range(num0), range(num1))

    dim = cell_points.shape[1]
    dRs = np.matmul(list(mesh), cell_vectors)
    points = np.reshape(dRs[:, np.newaxis, :] + cell_points, newshape=(-1, dim))

    return Lattice(points=points, vectors=vectors, name=name)


def KPath(points, min_num=100, loop=True):
    """
    Generate k-points on the path specified by the given `points`

    If `loop` is set to `False`, the k-path is generated as follow:
        points[0] ->  ... -> points[i] -> ... -> points[N-1]
    If `loop` is set to `True`, the k-path is generated as follow:
        points[0] -> ... -> points[i] -> ... -> points[N-1] -> points[0]
    The k-points between the given `points` are generated linearly

    Parameters
    ----------
    points : sequence of 1D arrays
        Special points on the k-path
        It is assumed that the adjacent points should be different
    min_num : int, optional
        The number of k-point on the shortest k-path segment
        The number of k-point on other k-path segments are scaled according
        to their length
        default: 100
    loop : boolean, optional
        Whether to generate a k-loop or not
        default: True

    Returns
    -------
    kpoints : 2D array with shape (N, 2) or (N, 3)
        A collection of k-points on the path specified by the given `points`
    indices : list
        The indices of the given `points` in the returned `kpoints` array
    """

    assert isinstance(min_num, int) and min_num >= 1
    assert len(points) > 1, "At least two points are required"
    assert all(point.shape in [(2,), (3,)] for point in points)

    point_num = len(points)
    points = np.concatenate((points, points), axis=0)
    end = (point_num + 1) if loop else point_num
    dRs = points[1:end] - points[0:end-1]
    lengths = np.linalg.norm(dRs, axis=-1)

    min_length = np.min(lengths)
    if min_length < 1e-4:
        raise ValueError("Identical adjacent points")

    sampling_nums = [
        int(min_num * length / min_length) for length in lengths
    ]
    kpoints = [
        np.linspace(0, 1, num=num, endpoint=False)[:, np.newaxis] * dR + start
        for num, dR, start in zip(sampling_nums, dRs, points)
    ]
    kpoints.append(points[[end - 1]])
    kpoints = np.concatenate(kpoints, axis=0)
    indices = [0, *np.cumsum(sampling_nums)]
    return kpoints, indices


# TODO: Add high symmetry points information for common lattice


if __name__ == "__main__":
    for cell in ["chain", "square", "triangle", "honeycomb", "kagome"]:
        lattice = lattice_generator(cell)
        lattice.show()
        lattice.show(scope=1)

    for cluster in ["cross", "z", "star", "benzene", "diphenyl", "gear"]:
        lattice = special_cluster(cluster)
        lattice.show()
        lattice.show(1)

    lattice_generator("chain", num0=10).show()
    lattice_generator("square", num0=6, num1=6).show()
    lattice_generator("honeycomb", num0=6, num1=6).show()
    lattice_generator("triangle", num0=6, num1=6).show()
    lattice_generator("kagome", num0=6, num1=6).show()

    As = np.array([[1.0, 0.0], [0.0, 1.0]])
    Gamma = np.array([0.0, 0.0])
    X = np.array([np.pi, 0.0])
    M = np.array([np.pi, np.pi])
    kpoints, indices = KPath([Gamma, X, M], loop=True)
    labels = [r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"]

    Es = -2 * np.sum(np.cos(np.dot(kpoints, As.T)), axis=-1)
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.plot(kpoints[:, 0], kpoints[:, 1])
    ax0.set_axis_off()
    ax0.set_aspect("equal")

    ax1.plot(Es)
    ax1.set_xlim(0, len(Es)-1)
    ax1.set_xticks(indices)
    ax1.set_xticklabels(labels)
    ax1.grid(axis="x", ls="dashed")
    plt.show()
