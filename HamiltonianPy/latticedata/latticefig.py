"""
This module provide function to show the honeycomb, kagome, square, triangular
lattice.
"""

from scipy.spatial import KDTree, distance

import matplotlib.pyplot as plt
import numpy as np
from time import time

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
        It can be only "square", "honeycomb", "kagome", "triangular", 's', 'h',
        't', 'k'

    Return:
    -------
    sites: np.array
        The coordinates of the sites.
        Every row represents a site.
    """

    lattice = lattice.lower()
    tvs = np.array([[1, 0], [1.0/2, np.sqrt(3)/2]])

    if lattice in ("square", 's'):
        ps = np.array([[0, 0]])
        tvs = np.array([[1, 0], [0, 1]])
    elif lattice in ("triangular", 't'):
        ps = np.array([[0, 0]])
    elif lattice in ("honeycomb", 'h'):
        ps = np.array([[0, 0], [0, 1/np.sqrt(3)]])
    elif lattice in ("kagome", 'k'):
        ps = np.array([[0, 0], [1.0/4, np.sqrt(3)/4], [1.0/2, 0]])
    else:
        raise ValueError("The invalid lattice parameter.")

    dRs = np.dot([(x, y) for x in range(numx) for y in range(numy)], tvs)
    return np.concatenate([dRs + p for p in ps], axis=0)
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

def show(lattice, numx=9, numy=9, link=True, 
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

    plt.figure()
    if link:
        bonds = bondsGenerator(sites=sites)
        for (x0, y0), (x1, y1) in bonds:
            plt.plot([x0, x1], [y0, y1], color=lcolor, linewidth=lwidth)

    for x, y in sites:
        plt.plot(x, y, marker='o', color=pcolor, markersize=psize)
    
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.show()
# }}}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
    description="Draw some 2D lattice according to the lattice parameter.")

    parser.add_argument("lattice", help="Which kind of 2D lattice to draw.",
    choices=("square", "triangular", "honeycomb", "kagome", 's', 't','h', 'k'),
    type=str)

    parser.add_argument("--scope", type=int, default=[9, 9], nargs=2, 
    help="The number of cells in the a0 and a1 direction(default: %(default)s).")

    parser.add_argument("-l", "--link", action="store_true", 
    help="Whether to draw the nearest neighbor bond.")

    parser.add_argument("--pcolor", default='k', 
    help="Color of the point(default : black).")

    parser.add_argument("--lcolor", default='k', 
    help="Color of the bond(default: black).")

    parser.add_argument("--psize", default=6, type=int, 
    help="Size of the point(default: %(default)s).")

    parser.add_argument("--lwidth", default=2, type=int, 
    help="Width of the bond(default: %(default)s).")

    args = parser.parse_args()
    kwargs = {"lattice": args.lattice, "numx": args.scope[0], 
              "numy": args.scope[1], "link": args.link, "pcolor": args.pcolor, 
              "psize": args.psize, "lcolor": args.lcolor, "lwidth": args.lwidth}
    show(**kwargs)
