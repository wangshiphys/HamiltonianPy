"""
This module provide description of lattice with translation symmetry.
"""

from copy import deepcopy
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm, solve
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy.arrayformat import arrayformat

__all__ = ["Lattice"]

#Useful constant in this module.
INDENT = 6
PRECISION = 4
###############################

class Lattice:
    """
    This class provide a description of a cluster with translation symmetry!

    The unified description of a lattice, the index of a given site, the
    neighbors of a given site, the relative location of a given site in the
    lattice with respect to the reference cluster, etc. This class provide
    some method to give answer to these most concerned questions!

    Attributes:
    -----------
    points: ndarray
        A collection of the coordinates of the sites in the cluster.
        It is a two dimension numpy array and every row represents a site.
    tvs: ndarry
        A collection of the translation vectors of the lattice.
        It is a two dimension numpy array and every row represents a vector.
    bs: ndarray
        A collection of the base vectors of the reciprocal space.
        It is a two dimension numpy array and every row represents a vector.
    drs: ndarray
        A collection of the relative displace of every site in the cluster to
        the zeroth site in the cluster.
        It is a two dimension numpy array and every row represents a displace.
    num: int
        The number of site in the cluster.
    dim: int
        The dimension of the lattice.
    distance: list
        All possible distance between any two sites in the cluster.

    Methods:
    --------
    __init__(points, tvs)
    __str__()
    setdist(scope=None)
    decompose(site)
    sites(site, scope=1)
    show_lattice(site=None, n=1)
    neighbor_dist(nth)
    neighbors(site, nth)
    bonds(max_neighbor)
    incluster(site)
    """

    def __init__(self, points, tvs):# {{{
        """
        Initialize the instance of this class.

        See also the docstring of this class!
        """

        if not (isinstance(points, np.ndarray) and isinstance(tvs, np.ndarray)):
            raise TypeError("The input parameter is not ndarray!")

        num, dim = points.shape
        dim1, dim2 = tvs.shape

        if dim != dim1 or dim != dim2:
            raise ValueError("The dimension of the cluster points and "
                             "the translation vectors does not match!")
        elif dim > 3:
            raise ValueError("Unsupported space dimension!")

        self.drs = np.round(points - points[0], decimals=PRECISION)
        self.bs = 2 * np.pi * solve(tvs, np.identity(dim)).T
        self.points = points
        self.tvs = tvs
        self.num = num
        self.dim = dim

        self.points.flags.writeable = False
        self.tvs.flags.writeable = False
        self.drs.flags.writeable = False
        self.bs.flags.writeable = False
    # }}}

    def __str__(self):# {{{
        """
        Return the printing string of instance if the class.
        """

        prefix = "\n" + " " * INDENT
        info = "\npoints:" + prefix
        info += arrayformat(self.points).replace("\n", prefix)
        info += "\ntvs:" + prefix + arrayformat(self.tvs).replace("\n", prefix)
        info += "\ndrs:" + prefix + arrayformat(self.drs).replace("\n", prefix)
        info += "\nbs:" + prefix + arrayformat(self.bs).replace("\n", prefix)
        info += "\nSite number: {0}".format(self.num)
        info += "\nDimension: {0}\n".format(self.dim)
        return info
    # }}}

    def setdist(self, scope=None):# {{{
        """
        Set all the possible distance between two points in the scope!
        """

        if scope is None:
            sites = self.points
        elif isinstance(scope, int):
            sites = self.sites(self.points[0], scope)
        else:
            raise TypeError("The input scope is not an integer!")

        dist_set = set(np.round(pdist(sites), decimals=PRECISION))
        dist_set.add(0.0)
        dist_list = list(dist_set)
        dist_list.sort()
        self.dist = tuple(dist_list)
    # }}}

    def decompose(self, site):# {{{
        """
        Decompse the site with respect to the translation vectors and incluster
        displace.
        
        This method describle how to reach the given site through translates
        along the tanslation vectors and incluster displace. In this method,
        fractional translation along a translation vector is not allowed.

        Parameter:
        ----------
        site: ndarray
            The site to be decompsed.

        Return:
        -------
        result: tuple
            The elements of the tuple are ndarrays.
            The first ndarray is the coordinate of the point within the cluster
            which is equivalent to the input site. The second ndarray is the
            displace between the input site and the equivalent site, which is an
            integer composation of the translation vectors.
        """

        if not isinstance(site, np.ndarray):
            raise TypeError("The input site is not a ndarray!")
        if site.shape != (self.dim, ):
            raise ValueError("The dimension of the input site "
                             "and the lattice does not match!")
       
        relative = site - self.points[0]
        guess = solve(self.tvs.T, relative).astype(int)
        scope = (0, 1, -1, 2)
        for cfg in product(scope, repeat=self.dim):
            dR = np.dot(guess + np.array(cfg), self.tvs)
            ds = np.round(relative - dR, decimals=PRECISION)
            for dr, eqv_site in zip(self.drs, self.points):
                if np.all(ds==dr):
                    return deepcopy(eqv_site), dR

        raise ValueError("Failed to decompse the input site."
                         "It might not belong to the lattice.")
    # }}}

    def sites(self, site, scope=1):# {{{
        """
        Return the neighboring sites of the given site.

        Parameter:
        ----------
        site: ndarray
            The original site coordinate.
        scope: int, optional
            Specify the range of the neighboring sites.
            default: 1

        Return:
        -------
        result: ndarray
            The coordinates of the neighboring sites.
        """

        if scope < 0:
            raise ValueError("Got a negative input scope!")

        eqv_site, dR = self.decompose(site)
        cluster = self.points + dR

        buff = []
        dR_scope = list(range(0, scope+1)) + list(range(-scope, 0))
        for cfg in product(dR_scope, repeat=self.dim):
            buff.append(cluster + np.dot(np.array(cfg), self.tvs))
        res = np.concatenate(buff, axis=0)
        return res
    # }}}

    def show_lattice(self, site=None, n=1):# {{{
        """
        Plot the lattice.

        Parameter:
        ----------
        site: ndarray, optional
            The original site of the plot.
            default: self.points[0]
        n: int, optional
            Specify the range of the sites to be plotted.
            default: 1
        """

        if site is None:
            site = self.points[0]
        sites = self.sites(site, n)

        x = []
        y = []
        z = []
        fig = plt.figure()
        if self.dim == 1:
            for buff in sites:
                x.append(buff[0])
                y.append(0.0)
            plt.scatter(x, y, s=100, c='r', marker='o', edgecolor='r')
        elif self.dim == 2:
            for buff in sites:
                x.append(buff[0])
                y.append(buff[1])
            plt.scatter(x, y, s=100, c='r', marker='o', edgecolor='r')
        elif self.dim == 3:
            for buff in sites:
                x.append(buff[0])
                y.append(buff[1])
                z.append(buff[2])
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, s=100, c='r', marker='o', edgecolor='r')
        else:
            raise TypeError("Unsupportted dimension!")

        plt.show()
    # }}}

    def neighbor_dist(self, nth):# {{{
        """
        Return the neighbor distance specified by the parameter nth.

        In this method, nth=0 represents onsite, nth=1 
        nearest-neighbor, nth=2 next-nearest neighbor, etc.

        Parameter:
        ---------
        nth: int
            Specify which neighbor distance to return.

        Return:
        -------
        result: float
            The nth neighbor distance.
        """

        if hasattr(self, 'dist'):
            if nth >= len(self.dist):
                self.setdist(scope=nth+1)
        else:
            self.setdist(scope=nth+1)
        return self.dist[nth]
    # }}}

    def neighbors(self, site, nth):# {{{
        """
        Return the neighboring sites of the input site, specified by nth.

        The meaning of nth is the same as in method neighbor_distance.

        Paramter:
        --------
        site: ndarray
            The site whose neighbors is  to be calculated.
        nth: int
            Specify up to how far to be inclued.

        Return:
        -------
        result: ndarray
            The neighboring sites of site specified by nth.
        """

        sites = self.sites(site, nth + 2)
        judge = self.neighbor_dist(nth)
        tree = KDTree(sites)
        result = sites[tree.query_ball_point(site, r=judge+1e-4)]
        return result
    # }}}

    def bonds(self, max_neighbor):# {{{
        judge = self.neighbor_dist(nth=max_neighbor)
        sites = self.sites(self.points[0], max_neighbor+1)
        tree = KDTree(sites)
        pairs = tree.query_pairs(r=judge+5e-4)
        intra = []
        inter = []
        for pair in pairs:
            p0, p1 = sites[pair, :]
            if pair[0] < self.num and pair[1] < self.num:
                intra.append((p0, p1))
            elif pair[0] < self.num:
                inter.append((p0, p1))
            elif pair[1] < self.num:
                inter.append((p1, p0))
        return intra, inter
    # }}}

    def incluster(self, site):# {{{
        """
        Determine whether the input site belong to the initial cluster.

        Parameter:
        ----------
        site: ndarray
            The site which is to be checked.

        Return:
        -------
        result: boolean
            If site belong to the initial cluster return true, else false.
        """

        if not isinstance(site, np.ndarray):
            raise TypeError("The input site is not a ndarray!")
        if site.shape != (self.dim, ):
            raise ValueError("The dimension of the input site and "
                             "the lattice does not match!")

        round_site = np.round(site, decimals=PRECISION)
        result = False
        for site1 in self.points:
            round_site1 = np.round(site1, decimals=PRECISION)
            if np.all(round_site == round_site1):
                result = True
                break
        return result
    # }}}


if __name__ == "__main__":
    sites = np.array([range(10)]).T
    tvs = np.array([[10]])
    l = Lattice(sites, tvs)
    intra, inter = l.bonds(1)
    print(intra)
    print("=" * 20)
    print(inter)

