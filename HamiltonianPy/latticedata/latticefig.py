"""
This module provide function to show the honeycomb, kagome, square, triangular
lattice.
"""

from scipy.spatial import KDTree, distance

import matplotlib.pyplot as plt
import numpy as np

from HamiltonianPy.latticedata import honeycomb, kagome, square, triangular

__all__ = ["show"]

#Useful constant in this module
ERR = 1e-4
###############################

def sitesGenerator(numx, numy, lattice):# {{{
    """
    This function generate sites coordinates specified by the input parameters.

    Parameter:
    ----------
    numx: int
        The number of sites along the first space base vector.
    numy: int
        The number of sites along the second space base vector.
    lattice: str
        The type of the lattice.
        It can be only "square", "honeycomb", "kagome", "triangular"

    Return:
    -------
    sites: np.array
        The coordinates of the sites.
        Every row represents a site.
    """

    lattice = lattice.lower()
    if lattice in ("honeycomb", 'h'):
        cellinfo = honeycomb.cellinfo
    elif lattice in ("kagome", 'k'):
        cellinfo = kagome.cellinfo
    elif lattice in ("square", 's'):
        cellinfo = square.cellinfo
    elif lattice in ("triangular", 't'):
        cellinfo = triangular.cellinfo
    else:
        raise ValueError("The invalid lattice parameter.")

    ps = cellinfo["points"]
    tvs = cellinfo["tvs"]
    tmp = [ps + np.dot([x, y], tvs) for x in range(numx) for y in range(numy)]
    return np.concatenate(tmp, axis=0)
# }}}

def bondsGenerator(sites):# {{{
    """
    Find all nearest pairs of the collection of sites.

    Parameter:
    ----------
    sites: np.array
        A collection of sites.
        Every row represents a site.

    Return:
    -------
    bonds: np.array.
        The shape of bonds should be (n, 2, 2), where n is the number of nearest
        bonds. The first dimension of bonds specifies different bonds, the
        second dimension specifies the two sites that consisting a bond, and the
        third dimension specifies the x, y coordinates of the site.
    """

    #The minimum distance between these sites, i.e. the nearest neighbor distance.
    min_dist = min(distance.pdist(sites))
    tree = KDTree(sites)
    pairs = list(tree.query_pairs(r=min_dist+ERR))
    bonds = sites[pairs, :]
    return bonds
# }}}

def show(lattice, numx=4, numy=4, link=True, 
         pcolor='k', psize=6, lcolor='k',lwidth=2):# {{{
    """
    Show the lattice specified by lattice parameter.

    Parameter:
    ----------
    lattice: str
        The type of the lattice.
        It can be only "square", "honeycomb", "kagome", "triangular"
    numx: int, optional
        The number of sites along the first space base vector.
        default: 4
    numy: int, optional
        The number of sites along the second space base vector.
        default: 4
    link: boolean, optional
        Determine whether to show both sites and bonds or just sites.
        default: True
    pcolor: str, optional
        The color of the site.
        default: 'k'
    psize: float, optional
        The size of the site.
        default: 6
    lcolor: str, optional
        The color of the bond.
        default: 'k'
    lsize: float, optional
        The width of the bond.
        default: 2
    """

    sites = sitesGenerator(numx=numx, numy=numy, lattice=lattice)
    bonds = bondsGenerator(sites=sites)
    
    plt.figure()
    if link:
        for (x0, y0), (x1, y1) in bonds:
            plt.plot([x0, x1], [y0, y1], color=lcolor, linewidth=lwidth)

    for x, y in sites:
        plt.plot(x, y, marker='o', color=pcolor, markersize=psize)
    
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.show()
# }}}
