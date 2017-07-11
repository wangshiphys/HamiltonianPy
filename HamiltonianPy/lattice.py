"""
This module provide description of lattice with translation symmetry.
"""

from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv, norm
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy.bond import Bond

#Useful constant in module
ERR = 1e-6
ZOOM = 10000
##########################

__all__ = ["Lattice"]

class DuplicateError(Exception):# {{{
    def __init__(self, msg):
        self.msg = msg
# }}}

class Lattice:# {{{
    """
    This class provide a description of a cluster with translation symmetry!

    Attributes
    ----------
    points : ndarray
        A collection of the coordinates of the sites in the cluster.
        It is a two dimension numpy array and every row represents a site.
    tvs : ndarry
        A collection of the translation vectors of the lattice.
        It is a two dimension numpy array and every row represents a vector.
    bs : ndarray
        A collection of the base vectors of the reciprocal space.
        It is a two dimension numpy array and every row represents a vector.
    sitenum : int
        The number of site in the cluster.
    spacedim : int
        The dimension of the space in which the points are described.
    transdim : int
        The number of linear independent directions along which the lattice is
        translational invariant.

    Methods:
    --------
    Public methods:
        getPoints()
        getTVs()
        getBs()
        getSite(index)
        getIndex(site, *, fold=False)
        decompose(site, *, extend=2)
        sitesFactory(*, site=None, scope=1)
        show(*, site=None, scope=1)
        neighborDist(nth)
        neighbors(site, nth, *, only=False, onsite=False)
        bonds(nth, *, only=False, periodic=False, remove_duplicate=True)
        incluster(site)
    Special methods:
        __init__(points, tvs)
        __str__()
    """

    def __init__(self, points, tvs):# {{{
        """
        Initialize instance of this class.

        See also the docstring of this class!
        """

        if isinstance(points, np.ndarray) and points.ndim == 2:
            sitenum, spacedim = points.shape
            if spacedim <= 3:
                self._sitenum = sitenum
                self._spacedim = spacedim
                self._points = np.array(points, copy=True)
                self._points.setflags(write=False)
            else:
                raise ValueError("Not supported space dimension.")
        else:
            raise TypeError("The invalid points parameter.")

        if (isinstance(tvs, np.ndarray) and tvs.ndim == 2 and 
            tvs.shape[0] <= spacedim and tvs.shape[1] == spacedim):
            self._transdim = tvs.shape[0]
            self._tvs = np.array(tvs, copy=True)
            self._tvs.setflags(write=False)
        else:
            raise TypeError("The invalid tvs parameter.")
        
        #Check is there duplicate point or vector in the given points or tvs.
        #If no duplications, all the possible distance between any point in the
        #cluster are returned and are cached as _dists attribute for later use.
        self._dists = self._verify()

        self._bs = 2 * np.pi * np.dot(inv(np.dot(tvs, tvs.T)), tvs)
        self._bs.setflags(write=False)
    # }}}

    def _verify(self):# {{{
        errmsg = "There are duplicate {0} in the given {1}."

        #Check is there duplicate translation vector in the given tvs.
        dists = pdist(self._tvs)
        if np.any(dists < ERR):
            raise DuplicateError(errmsg.format("translation vector", "tvs"))

        #Check is there duplicate point in the given points.
        #If True raise DuplicateError, if False, return all the possible
        #distance between ant two point in the cluster.
        dists = pdist(self._points)
        if np.any(dists < ERR):
            raise DuplicateError(errmsg.format("point", "points"))
        else:
            dists = set(np.ceil(dists * ZOOM) / ZOOM)
            dists.add(0.0)
            return sorted(dists)
    # }}}

    @property
    def points(self):# {{{
        """
        The points attribute of instance of this class.
        """

        return np.array(self._points, copy=True)
    # }}}

    @property
    def tvs(self):# {{{
        """
        The tvs attribute of instance of this class.
        """

        return np.array(self._tvs, copy=True)
    # }}}

    @property
    def bs(self):# {{{
        """
        The bs attribute of instance of this class.
        """

        return np.array(self._bs, copy=True)
    # }}}

    @property
    def sitenum(self):# {{{
        """
        The sitenum attribute of instance of this class.
        """

        return self._sitenum
    # }}}

    @property
    def spacedim(self):# {{{
        """
        The spacedim attribute of instance of this class.
        """

        return self._spacedim
    # }}}

    @property
    def transdim(self):# {{{
        """
        The transdim attribute of instance of this class.
        """

        return self._transdim
    # }}}

    def __str__(self):# {{{
        """
        Return a string that describes the content of the instance.
        """

        info = "Site number: {0}\n".format(self._sitenum)
        info += "Space dimension: {0}\n".format(self._spacedim)
        info += "Translation dimension: {0}\n".format(self._transdim)
        info += "points:\n" + str(self._points) + '\n'
        info += "tvs:\n" + str(self._tvs) + '\n'
        info += "bs:\n" + str(self._bs) + '\n'
        return info
    # }}}

    def getPoints(self):# {{{
        """
        Access the points attribute of instance of this class.
        """

        return np.array(self._points, copy=True)
    # }}}

    def getTVs(self):# {{{
        """
        Access the tvs attribute of instance of this class.
        """

        return np.array(self._tvs, copy=True)
    # }}}

    def getBs(self):# {{{
        """
        Access the bs attribute of instance of this class.
        """

        return np.array(self._bs, copy=True)
    # }}}

    def getIndex(self, site, *, fold=False):# {{{
        """
        Return the index corresponding to the given site.
        
        Parameters
        ----------
        site : np.ndarray
            The site of whose index is queried.
        fold : boolean, optional, keyword argument only
            Whether to fold the given site back to the cluster.
            default: False

        Returns
        -------
        res : int
            The index of the given site if it belong the cluster.
        """
        
        if not isinstance(site, np.ndarray):
            raise TypeError("The input site is not a ndarray!")
        elif site.shape != (self._spacedim, ):
            raise ValueError("The dimension of the input site "
                             "and the lattice does not match!") 

        if fold:
            site, trash = self.decompose(site)

        dist, index = cKDTree(self._points).query(site)
        if dist < ERR:
            return index
        else:
            raise KeyError("The given site does not belong the cluster.")
    # }}}

    def getSite(self, index):# {{{
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

        if not isinstance(index, int):
            raise TypeError("The input index is not an integer.")
        elif index < 0:
            raise ValueError("The given index should be none negative integer.")
        elif index >= self._sitenum:
            raise ValueError("The given index is larger than the number of sites.")

        return np.array(self._points[index], copy=True)
    # }}}

    def _scopeGuess(self, site, extend=2):# {{{
        #Guess the scope where we can find the given site. The local variable
        #max_trans means the possible maximum translation along a given
        #translation vector.
        
        scopes = []
        distance = max(norm(self._points - site, axis=1))
        for tv in self._tvs:
            max_trans = int(distance / norm(tv)) + extend
            scopes.append(range(-max_trans, max_trans+1))
        self._dRsTree = cKDTree(np.dot(list(product(*scopes)), self._tvs))
    # }}}

    def decompose(self, site, *, extend=2):# {{{
        """
        Decompse the site with respect to the translation vectors and site in
        the cluster.
        
        This method describe how to reach the given site through translates
        one site of the cluster along the tanslation vectors. Fractional 
        translation along a translation vector is invalid.
        The basic idea is that: we first calculate a collection of possible 
        translations for the given site,  and then for every site in the 
        cluster we search the collection to determine if we can reach the given
        site.
        The possible translations is calculated as follow: The maximum distance
        between the given site and site in the cluster is calculated. For
        every translation vector, we divide this distance by the length of the
        translation vector, round off the quotient and add the "extend" and then
        we get the possible maximum translation along the translation vector.

        Parameters
        ----------
        site : ndarray
            The site to be decompsed.
        extend : int, optional, keyword argument only
            Determine the scope of the possible maximum translation.
            default: 2

        Returns
        -------
        result : tuple
            The elements of the tuple are ndarrays.
            The first ndarray is the coordinate of the point within the cluster
            which is equivalent to the input site. The second ndarray is the
            displace between the input site and the equivalent site.
        """

        if not isinstance(site, np.ndarray):
            raise TypeError("The input site is not a ndarray!")
        if site.shape != (self._spacedim, ):
            raise ValueError("The dimension of the input site "
                             "and the lattice does not match!") 
        if not isinstance(extend, int) or extend < 0:
            raise ValueError("The invalid extend parameter.")

        errmsg = "Failed to decompse the input site. "
        errmsg += "It might not belong to the lattice or not located in "
        errmsg += "the scope determined by this method. Check carefully."

        if hasattr(self, "_dRsTree"):
            for i in range(2):
                dRs = site - self._points
                dists, trash = self._dRsTree.query(dRs)
                matches = np.nonzero(dists < ERR)[0]
                if len(matches) == 1:
                    index = matches[0]
                    return np.array(self._points[index], copy=True), dRs[index]
                else:
                    if i == 0:
                        self._scopeGuess(site, extend=extend)
                    else:
                        raise ValueError(errmsg)
        else:
            self._scopeGuess(site, extend=extend)
            dRs = site - self._points
            dists, trash = self._dRsTree.query(dRs)
            matches = np.nonzero(dists < ERR)[0]
            if len(matches) == 1:
                index = matches[0]
                return np.array(self._points[index], copy=True), dRs[index]
            else:
                raise ValueError(errmsg)
    # }}}

    def sitesFactory(self, *, site=None, scope=1):# {{{
        """
        Return the neighboring sites of the given site.

        Parameters
        ----------
        site : ndarray, keyword only, optional
            The original site coordinate.
            default: None
        scope : int, keyword only, optional
            Specify the range of the neighboring sites.
            default: 1

        Returns
        -------
        result : ndarray
            The coordinates of the neighboring sites.
        """

        if not isinstance(scope, int) or scope < 1:
            raise ValueError("The invalid scope parameter.")

        if site is None:
            dR = 0
        else:
            trash, dR = self.decompose(site)
        cluster = self._points + dR

        #The first element of dRs is zero-vector. It ensures that these points
        #in the cluster are place in the res before points not in the cluster.
        mesh = product(range(-scope, scope+1), repeat=self._transdim)
        dRs = np.dot(sorted(mesh, key=lambda item: norm(item)), self._tvs)
        return np.concatenate([cluster + dR for dR in dRs], axis=0)
    # }}}

    def show(self, *, site=None, scope=1):# {{{
        """
        Show the lattice.

        Parameters
        ----------
        site : ndarray, keyword only, optional
            The original site of the plot.
            default: self.points[0]
        scope : int, keyword only, optional
            Specify the range of the sites to be plotted.
            default: 1
        """

        sites = self.sitesFactory(site=site, scope=scope)

        markersize = 200
        fig = plt.figure()
        if self._spacedim == 1:
            for x in sites:
                plt.scatter(x, 0.0, marker='o', s=markersize)
        elif self._spacedim == 2:
            for x, y in sites:
                plt.scatter(x, y, marker='o', s=markersize)
        elif self._spacedim == 3:
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

        Parameters
        ----------
        nth : int
            Specify which neighbor distance to return.

        Returns
        -------
        result : float
            The nth neighbor distance.
        """
        
        if not isinstance(nth, int) or nth < 0:
            raise TypeError("The nth parameter must be nonnegative integer.")

        if len(self._dists) <= nth:
            sites = self.sitesFactory(scope=nth)
            dists = set(np.ceil(pdist(sites) * ZOOM) / ZOOM)
            dists.add(0.0)
            self._dists = sorted(dists)
        return self._dists[nth]
    # }}}

    def neighbors(self, site, nth, *, only=False, onsite=False):# {{{
        """
        Return the neighboring sites of the input site, specified by nth.

        The meaning of nth is the same as in method neighborDist.

        Paramters
        ---------
        site : ndarray
            The site whose neighbors is  to be calculated.
        nth : int
            Specify up to how far to be inclued.
        only : boolean, optional, keyword argument only
            If True, only these sites which the distance to the given site
            equals to the nth neighbor distance, if False, all the sites
            which the distance to the given site not larger than the
            nth neighbor distance.
        onsite : boolean, optional, keyword argument only
            Whether the given site itself is included in the result.

        Returns
        -------
        result : ndarray
            The neighboring sites of site specified by the parameters.
        """

        judge = self.neighborDist(nth)
        scope = max([int(judge/norm(tv)) + 1 for tv in self._tvs])
        sites = self.sitesFactory(site=site, scope=scope)
        tree = cKDTree(sites)
        onsite_index = tree.query_ball_point(site, r=ERR)[0]
        outer_indices = set(tree.query_ball_point(site, r=judge+ERR))
        if not onsite:
            outer_indices.remove(onsite_index)
        if only and nth > 0:
            judge = self.neighborDist(nth-1)
            inner_indices = set(tree.query_ball_point(site, r=judge+ERR))
        else:
            inner_indices = set()
        indices = list(outer_indices.difference(inner_indices))
        return sites[indices]
    # }}}

    def bonds(self, nth, *, only=False, periodic=False, remove_duplicate=True):# {{{
        """
        Return all bonds specified by the given parameters.

        Parameters
        ----------
        nth : int
            0 means onsite, 1 represents nearest neighbor, 2 represents
            next-nearest neighbor, etc.
        only : boolean, optional, keyword argument only
            If True, only these bonds which length equals to the nth neighbor
            distance, if False, all the bonds which length equal or less than
            the nth neighbor distance.
        remove_duplicate : boolean, optional, keyword argument only
            For every bond that connects the cluster and the environment, there
            exist another bond that is equivalent to it because of translation
            symmetry. If this parameter is set to True, then these equivalent
            bonds will be removed from the results.
            default: True
        periodic : boolean, optional, keyword argument only
            Whether to decompose the inter bonds.
            default: False

        Returns
        -------
        intra : list of Bond
            A collection of the bonds that both endpoints belong the cluster.
        inter : list of Bond
            A collection of the bonds that one endpoints belong the cluster and
            the other does not.
        """

        judge = self.neighborDist(nth=nth)
        scope = max([int(judge/norm(tv)) + 1 for tv in self._tvs])
        sites = self.sitesFactory(scope=scope)
        tree = cKDTree(sites)
        pairs_outer = tree.query_pairs(r=judge+ERR)
        if only and neighbor > 1:
            judge = self.neighborDist(nth=nth-1)
            pairs_inner = tree.query_pairs(r=judge+ERR)
        else:
            pairs_inner = set()

        pairs = pairs_outer.difference(pairs_inner)

        intra = []
        inter = []
        rule = set()
        for index0, index1 in pairs:
            p0 = sites[index0]
            p1 = sites[index1]
            if index0 < self._sitenum and index1 < self._sitenum:
                intra.append(Bond(p0, p1, directional=True))
            elif index0 < self._sitenum or index1 < self._sitenum:
                p0_eqv, trash = self.decompose(p0)
                p1_eqv, trash = self.decompose(p1)
                bond = Bond(p0, p1, directional=True)
                bond_eqv = Bond(p0_eqv, p1_eqv, directional=True)
                if remove_duplicate:
                    key = sorted([self.getIndex(p0_eqv), self.getIndex(p1_eqv)])
                    key = tuple(key)
                    if key not in rule:
                        if periodic:
                            inter.append(bond_eqv)
                        else:
                            inter.append(bond)
                        rule.add(key)
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
    # }}}
# }}}


if __name__ == "__main__":
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    tvs = np.array([[2, 0], [0, 2]])
    cluster = Lattice(points, tvs)
    print("Attribute points:\n", cluster.points)
    print("Getpoints:\n", cluster.getPoints())
    print("Attribute tvs:\n", cluster.tvs)
    print("Gettvs:\n", cluster.getTVs())
    print("Attribute bs:\n", cluster.bs)
    print("Getbs:\n", cluster.getBs())
    print("Attribute sitenum:\n", cluster.sitenum)
    print("Attribute spacedim:\n", cluster.spacedim)
    print("Attribute transdim:\n", cluster.transdim)

    try:
        cluster.points = None
    except AttributeError:
        print("Can not set point attribute.")
    try:
        cluster.tvs = None
    except AttributeError:
        print("Can not set tvs attribute.")
    try:
        cluster.bs = None
    except AttributeError:
        print("Can not set bs attribute.")
    try:
        cluster.sitenum = None
    except AttributeError:
        print("Can not set sitenum attribute.")
    try:
        cluster.spacedim = None
    except AttributeError:
        print("Can not set spacedim attribute.")
    try:
        cluster.transdim = None
    except AttributeError:
        print("Can not set transdim attribute.")

    for site in points:
        print("The index of site{0} is: {1}".format(site, cluster.getIndex(site)))
    for index in range(len(points)):
        print("The {0}th point is: {1}".format(index, cluster.getSite(index)))
    
    site = np.random.randint(0, 100, size=2)
    eqv_site, dR = cluster.decompose(site)
    print("The orginal site: ", site)
    print("The eqv_site: ", eqv_site)
    print("The translation: ", dR)
    neighbors = cluster.neighbors(site, nth=2, only=False, onsite=True)
    for neighbor in neighbors:
        print(neighbor)

    intra, inter = cluster.bonds(nth=1)
    for bond in intra:
        print("Intra bond:\n", bond)
        print()
    for bond in inter:
        print("Inter bond:\n", bond)
        print()

    from time import time
    numx = 100
    numy = 100
    points = np.array(list(product(range(numx), range(numy))))
    tvs = np.array([[numx, 0], [0, numy]])
    t0 = time()
    cluster = Lattice(points, tvs)
    t1 = time()
    print("The time spend on construct the cluster: ", t1 - t0)
    
    t0 = time()
    intra, inter = cluster.bonds(nth=1)
    t1 = time()
    print("The time spend on find all the bonds: ", t1 - t0)
    print("The found number of intra bond: ", len(intra))
    print("The actual number of intra bond: ", 2 * numx * numy - numx - numy)
    print("The found number of inter bond: ", len(inter))
    print("The actual number of found inter bond: ", numx + numy)

    t0 = time()
    site = np.random.randint(0, 5 * numx, size=2)
    eqv_site, dR = cluster.decompose(site=site)
    print("The orginal site: ", site)
    print("The equvalient site: ", eqv_site)
    print("The translation : ", dR)
    t1 = time()
    print(t1 - t0)
