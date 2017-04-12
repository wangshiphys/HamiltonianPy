"""
This module provide description of lattice with translation symmetry.
"""

from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm, solve
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy.bond import Bond
from HamiltonianPy.constant import NDIGITS

#Useful constant in module
ERR = 1e-4
##########################

__all__ = ["Lattice"]

class Lattice:# {{{
    """
    This class provide a description of a cluster with translation symmetry!

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
    dist: tuple
        A collection of possible distance between two points of the lattice.

    Methods:
    --------
    Special methods:
        __init__(points, tvs)
        __str__()
    General methods:
        getPoints()
        getTVs()
        getBs()
        decompose(site)
        sitesFactory(site, scope=1)
        show(site=None, n=1)
        neighborDist(nth)
        neighbors(site, nth)
        bonds(neighbor, only=False)
        incluster(site)
    """

    def __init__(self, points, tvs):# {{{
        """
        Initialize the instance of this class.

        See also the docstring of this class!
        """

        if isinstance(points, np.ndarray) and len(points.shape) == 2:
            num, dim = points.shape
            if dim <= 3:
                self.num = num
                self.dim = dim
                self.points = np.array(points[:, :])
            else:
                raise ValueError("Not supported space dimension.")
        else:
            raise TypeError("The invalid points parameter.")

        if isinstance(tvs, np.ndarray) and tvs.shape == (dim, dim):
            self.tvs = np.array(tvs[:, :])
        else:
            raise TypeError("The invalid tvs parameter.")

        self.drs = points - points[0]
        self.bs = 2 * np.pi * solve(tvs, np.identity(dim)).T
        tmp = set(np.around(pdist(points), decimals=NDIGITS))
        tmp.add(0.0)
        self.dist = tuple(sorted(tmp))
    # }}}

    def __str__(self):# {{{
        """
        Return a string that descriibles the content of the instance.
        """

        info = "Site number: {0}\n".format(self.num)
        info += "Space dimension: {0}\n".format(self.dim)
        info += "points:\n" + str(self.points) + '\n'
        info += "tvs:\n" + str(self.tvs) + '\n'
        info += "drs:\n" + str(self.drs) + '\n'
        info += "bs:\n" + str(self.bs) + '\n'
        return info
    # }}}

    def getPoints(self):# {{{
        """
        Access the points attribute of instance of this class.
        """

        return np.array(self.points[:, :])
    # }}}

    def getTVs(self):# {{{
        """
        Access the tvs attribute of instance of this class.
        """

        return np.array(self.tvs[:, :])
    # }}}

    def getBs(self):# {{{
        """
        Access the bs attribute of instance of this class.
        """

        return np.array(self.bs[:, :])
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
            displace between the input site and the equivalent site
        """

        if not isinstance(site, np.ndarray):
            raise TypeError("The input site is not a ndarray!")
        if site.shape != (self.dim, ):
            raise ValueError("The dimension of the input site "
                             "and the lattice does not match!")
       
        scope = (0, 1, -1, 2, -2)
        relative = site - self.points[0]
        guess = solve(self.tvs.T, relative).astype(int)
        cfgs = guess + np.array(list(product(scope, repeat=self.dim)))
        dRs = np.dot(cfgs, self.tvs)
        relative = np.around(relative, decimals=NDIGITS)
        for dR in dRs:
            for dr, eqv_site in zip(self.drs, self.points):
                tmp = np.around(dR + dr, decimals=NDIGITS)
                if np.all(tmp == relative):
                    return np.array(eqv_site[:]), dR
        raise ValueError("Failed to decompse the input site."
                         "It might not belong to the lattice.")
    # }}}

    def sitesFactory(self, site, scope=1):# {{{
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

        if isinstance(scope, int) and scope >= 1:
            trash, dR = self.decompose(site)
            cluster = self.points + dR

            tmp = []
            dR_scope = list(range(0, scope+1)) + list(range(-scope, 0))
            dRs = np.dot(list(product(dR_scope, repeat=self.dim)), self.tvs)
            for dR in dRs:
                tmp.append(cluster + dR)
            res = np.concatenate(tmp, axis=0)
            return res
        else:
            raise ValueError("The invalid scope parameter.")
    # }}}

    def show(self, site=None, n=1):# {{{
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
        sites = self.sitesFactory(site, n)

        markersize = 200
        fig = plt.figure()
        if self.dim == 1:
            for x in sites:
                plt.scatter(x, 0.0, marker='o', s=markersize)
        elif self.dim == 2:
            for x, y in sites:
                plt.scatter(x, y, marker='o', s=markersize)
        elif self.dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            for x, y, z in sites:
                ax.scatter(x, y, z, marker='o', s=markersize)
        else:
            raise TypeError("Unsupportted dimension!")

        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("off")
        plt.show()
    # }}}

    def neighborDist(self, nth):# {{{
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
        
        if not isinstance(nth, int):
            raise TypeError("The nth parameter is not of type int.")

        if len(self.dist) <= nth:
            sites = self.sitesFactory(self.points[0], scope)
            tmp = set(np.around(pdist(sites), decimals=NDIGITS))
            tmp.add(0.0)
            self.dist = tuple(sorted(tmp))
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

        sites = self.sitesFactory(site, nth + 2)
        judge = self.neighborDist(nth)
        tree = KDTree(sites)
        result = sites[tree.query_ball_point(site, r=judge+ERR)]
        return result
    # }}}

    def bonds(self, neighbor, only=False):# {{{
        """
        Return all bonds specified by the neighbor and only parameter.

        Parameter:
        ----------
        neighbor: int
            Specify the distance order of two points belong the lattice.
            0 means onsite, 1 represents nearest neighbor, 2 represents
            next-nearest neighbor, etc.
        only: boolean, optional
            If True, only these bonds whose length equals to the length
            specified by neighbor parameter is concerned, if False, all the
            bonds whose length equal or less than the length specified by
            neighbor parameter is concerned is concerned.

        Return:
        -------
        intra: list of Bond
            A collection of the bonds that both endpoints belong the cluster.
        inter: list of Bond
            A collection of the bonds that one endpoints belong the cluster and
            the other does not.
        """

        judge = self.neighborDist(nth=neighbor)
        sites = self.sitesFactory(self.points[0], neighbor+1)
        tree = KDTree(sites)
        pairs_outer = tree.query_pairs(r=judge+ERR)
        if only and neighbor > 1:
            judge = self.neighborDist(nth=neighbor-1)
            pairs_inner = tree.query_pairs(r=judge+ERR)
        else:
            pairs_inner = set()

        pairs = pairs_outer.difference(pairs_inner)

        intra = []
        inter = []
        for index0, index1 in pairs:
            p0 = sites[index0, :]
            p1 = sites[index1, :]
            if index0 < self.num and index1 < self.num:
                intra.append(Bond(p0, p1, directional=True))
            elif index0 < self.num:
                inter.append(Bond(p0, p1, directional=True))
            elif index1 < self.num:
                inter.append(Bond(p1, p0, directional=True))
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

        round_site = np.around(site, decimals=NDIGITS)
        result = False
        for tmp in self.points:
            tmp = np.around(tmp, decimals=NDIGITS)
            if np.all(round_site == tmp):
                result = True
                break
        return result
    # }}}
# }}}

if __name__ == "__main__":
    numx = 2
    numy = 2
    cell_points = np.array([[0, 0]])
    cell_tvs = np.array([[1, 0], [0, 1]])
    points = [cell_points + np.dot([x, y], cell_tvs) for x in range(numx) for y in range(numy)]
    points = np.concatenate(points, axis=0)
    tvs = cell_tvs * np.array([[numx, numx], [numy, numy]])

    cluster = Lattice(points, tvs)
    intra, inter = cluster.bonds(2, only=True)
    for bond in intra:
        print(bond)
    print("=" * 20)
    for bond in inter:
        print(bond)
