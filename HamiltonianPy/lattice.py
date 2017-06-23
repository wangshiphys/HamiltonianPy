"""
This module provide description of lattice with translation symmetry.
"""

from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv, norm
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy.bond import Bond
from HamiltonianPy.constant import NDIGITS

#Useful constant in module
ERR = 1e-6
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
    sitenum: int
        The number of site in the cluster.
    spacedim: int
        The dimension of the space in which the points are described.
    perioddim: int
        The number of linear independent directions along which the lattice is
        translational invariant.

    Methods:
    --------
    Special methods:
        __init__(points, tvs)
        __str__()
    General methods:
        getPoints()
        getTVs()
        getBs()
        getSite(index)
        getIndex(site, *, fold=False)
        decompose(site, *, extend=3)
        sitesFactory(site=None, scope=1)
        show(site=None, n=1)
        neighborDist(nth)
        neighbors(site, nth)
        bonds(neighbor, *, only=False, periodic=False, remove_duplicate=True)
        incluster(site)
    """

    def __init__(self, points, tvs):# {{{
        """
        Initialize the instance of this class.

        See also the document string of this class!
        """

        if isinstance(points, np.ndarray) and points.ndim == 2:
            sitenum, spacedim = points.shape
            if spacedim <= 3:
                self.sitenum = sitenum
                self.spacedim = spacedim
                self.points = np.array(points[:, :])
            else:
                raise ValueError("Not supported space dimension.")
        else:
            raise TypeError("The invalid points parameter.")

        if (isinstance(tvs, np.ndarray) and tvs.ndim == 2 and 
            tvs.shape[0] <= spacedim and tvs.shape[1] == spacedim):
            self.perioddim = tvs.shape[0]
            self.tvs = np.array(tvs[:, :])
        else:
            raise TypeError("The invalid tvs parameter.")

        self.bs = 2 * np.pi * np.dot(inv(np.dot(tvs, tvs.T)), tvs)
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "Site number: {0}\n".format(self.sitenum)
        info += "Space dimension: {0}\n".format(self.spacedim)
        info += "Translation dimension: {0}\n".format(self.perioddim)
        info += "points:\n" + str(self.points) + '\n'
        info += "tvs:\n" + str(self.tvs) + '\n'
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

    def getIndex(self, site, *, fold=False):# {{{
        """
        Return the index corresponding to the given site.
        
        Parameter:
        ----------
        fold: boolean, optional, keyword argument only
            Whether to decompose the given site to the cluster.
            default: False
        """
        
        if not isinstance(site, np.ndarray):
            raise TypeError("The input site is not a ndarray!")
        elif site.shape != (self.spacedim, ):
            raise ValueError("The dimension of the input site "
                             "and the lattice does not match!") 

        if fold:
            site, trash = self.decompose(site)

        dists = norm(self.points - site, axis=1)
        indices = np.nonzero(dists < ERR)[0]
        num = len(indices)
        if num == 1:
            return indices[0]
        elif num == 0:
            raise ValueError("The given site does not belong the cluster.")
        else:
            raise ValueError("There are duplicate point in the cluster.")
    # }}}

    def getSite(self, index):# {{{
        """
        Return the site corresponding to the given index.
        """

        if not isinstance(index, int):
            raise TypeError("The input index is not an integer.")
        elif index < 0:
            raise ValueError("The given index should be none negative.")
        elif index >= self.sitenum:
            raise ValueError("The given index is larger than the number of sites.")

        return np.array(self.points[index, :])
    # }}}

    def _scopeGuess(self, site, extend=3):# {{{
        """
        Guess the scope where we can find the given site. The local variable 
        max_trans means the possible maximum translation along a given 
        translation vector.
        
        This method is only for internal use.
        """

        scopes = []
        distance = norm(site - self.points[0])
        for tv in self.tvs:
            max_trans = int(distance / norm(tv)) + extend
            scopes.append(range(-max_trans, max_trans+1))
        self._dRsTree = KDTree(np.dot(list(product(*scopes)), self.tvs))
    # }}}

    def decompose(self, site, *, extend=3):# {{{
        """
        Decompse the site with respect to the translation vectors and site in
        the cluster.
        
        This method describle how to reach the given site through translates
        one site of the cluster along the tanslation vectors. Fractional 
        translation along a translation vector is invalid.
        The basic idea is that: we first calculate a collection of possible 
        translations for the given site,  and then for every site in the 
        cluster we search the collection to determine if we can reach the given
        site.
        The possible translations is calculated as follow: The distance between
        the given site and the zeroth site in the cluster is calculated. For
        every translation vector, we divide this distance by the length of the
        translation vector, round off the quotient and add the "extend" and then
        we get the possible maximum translation  along the translation vector.

        Parameter:
        ----------
        site: ndarray
            The site to be decompsed.
        extend: int, optional, keyword argument only
            Determine the scope of the possible maximum translation.
            default: 3

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
        if site.shape != (self.spacedim, ):
            raise ValueError("The dimension of the input site "
                             "and the lattice does not match!") 
        if not isinstance(extend, int) or extend < 0:
            raise ValueError("The invalid extend parameter.")

        EXC_MSG = "Failed to decompse the input site. "
        EXC_MSG += "It might not belong to the lattice or not located in "
        EXC_MSG += "the scope determined by this method. Check carefully."

        if hasattr(self, "_dRsTree"):
            for i in range(2):
                for eqv_site in self.points:
                    dR = site - eqv_site
                    ds, index = self._dRsTree.query(dR)
                    if ds < ERR:
                        return np.array(eqv_site[:]), dR
                if i == 0:
                    self._scopeGuess(site, extend=extend)
                else:
                    raise ValueError(EXC_MSG)
        else:
            self._scopeGuess(site, extend=extend)
            for eqv_site in self.points:
                dR = site - eqv_site
                ds, index = self._dRsTree.query(dR)
                if ds < ERR:
                    return np.array(eqv_site[:]), dR
            raise ValueError(EXC_MSG)
    # }}}

    def sitesFactory(self, site=None, scope=1):# {{{
        """
        Return the neighboring sites of the given site.

        Parameter:
        ----------
        site: ndarray, optional
            The original site coordinate.
            default: None
        scope: int, optional
            Specify the range of the neighboring sites.
            default: 1

        Return:
        -------
        result: ndarray
            The coordinates of the neighboring sites.
        """

        if site is None:
            dR = 0
        else:
            trash, dR = self.decompose(site)

        if isinstance(scope, int) and scope >= 1:
            cluster = self.points + dR
            tmp = []
            #This first element of dR_scope is 0. It ensures that these points in 
            #the cluster are place in the res before points not in the cluster.
            dR_scope = sorted(range(-scope, scope+1), key=lambda item: abs(item))
            dRs = np.dot(list(product(dR_scope, repeat=self.perioddim)), self.tvs)
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

        sites = self.sitesFactory(site, n)

        markersize = 200
        fig = plt.figure()
        if self.spacedim == 1:
            for x in sites:
                plt.scatter(x, 0.0, marker='o', s=markersize)
        elif self.spacedim == 2:
            for x, y in sites:
                plt.scatter(x, y, marker='o', s=markersize)
        elif self.spacedim == 3:
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

        if not hasattr(self, "_dist") or len(self._dist) <= nth:
            sites = self.sitesFactory(scope=nth+1)
            tmp = set(np.around(pdist(sites), decimals=NDIGITS))
            tmp.add(0.0)
            self._dist = tuple(sorted(tmp))
        return self._dist[nth]
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

    def bonds(self, neighbor, *, only=False, periodic=False, remove_duplicate=True):# {{{
        """
        Return all bonds specified by the neighbor and only parameter.

        Parameter:
        ----------
        neighbor: int
            Specify the distance order of two points belong the lattice.
            0 means onsite, 1 represents nearest neighbor, 2 represents
            next-nearest neighbor, etc.
        only: boolean, optional, keyword argument only
            If True, only these bonds which length equals to the length
            specified by neighbor parameter is concerned, if False, all the
            bonds which length equal or less than the length specified by
            neighbor parameter is concerned.
        remove_duplicate: boolean, optional, keyword argument only
            For every bond that connects the cluster and the environment, there
            exist another bond that is equivalent to it because of translation
            symmetry. If this parameter is set to True, then theequivalent bond
            will be removed from the results.
            default: True
        periodic: boolean, optional, keyword argument only
            Whether to decompose the inter bonds.
            default: False

        Return:
        -------
        intra: list of Bond
            A collection of the bonds that both endpoints belong the cluster.
        inter: list of Bond
            A collection of the bonds that one endpoints belong the cluster and
            the other does not.
        """

        judge = self.neighborDist(nth=neighbor)
        sites = self.sitesFactory(scope=neighbor+1)
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
        judge = set()
        for index0, index1 in pairs:
            p0 = sites[index0, :]
            p1 = sites[index1, :]
            if index0 < self.sitenum and index1 < self.sitenum:
                intra.append(Bond(p0, p1, directional=True))
            elif index0 < self.sitenum or index1 < self.sitenum:
                p0_eqv, trash = self.decompose(p0)
                p1_eqv, trash = self.decompose(p1)
                bond = Bond(p0, p1, directional=True)
                bond_eqv = Bond(p0_eqv, p1_eqv, directional=True)
                if remove_duplicate:
                    key = sorted([self.getIndex(p0_eqv), self.getIndex(p1_eqv)])
                    key = tuple(key)
                    if key not in judge:
                        if periodic:
                            inter.append(bond_eqv)
                        else:
                            inter.append(bond)
                        judge.add(key)
                else:
                    if periodic:
                        inter.append(bond_eqv)
                    else:
                        inter.append(bond)

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
        if site.shape != (self.spacedim, ):
            raise ValueError("The dimension of the input site and "
                             "the lattice does not match!")
        return np.any(norm(site - self.points, axis=1) < ERR)
    # }}}
# }}}

if __name__ == "__main__":
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    tvs = np.array([[2, 0], [0, 2]])
    cluster = Lattice(points, tvs)
    print(cluster.getIndex(np.array([80, 65]), fold=False))
