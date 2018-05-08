"""
This module provide function to show the honeycomb, kagome, square, triangle lattice.
"""

from scipy.spatial import KDTree, distance

import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    "show",
]


#Useful constant in this module
ERR = 1e-4
###############################


_cell_square = np.array([[0.0, 0.0]], dtype=np.float64)
_cell_triangle = np.array([[0.0, 0.0]], dtype=np.float64)
_cell_honeycomb = np.array([[0.0, 0.0], [0.0, 1/np.sqrt(3)]], dtype=np.float64)
_cell_kagome = np.array(
    [[0.0, 0.0], [1.0/4, np.sqrt(3)/4], [0.5, 0.0]],
    dtype=np.float64
)

_tvs_square = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
_tvs_triangle = np.array([[1.0, 0.0], [0.5, np.sqrt(3)/2]], dtype=np.float64)
_tvs_honeycomb = _tvs_triangle
_tvs_kagome = _tvs_triangle


_cells = {
    "square": _cell_square,
    "s": _cell_square,
    "triangle": _cell_triangle,
    "t": _cell_triangle,
    "honeycomb": _cell_honeycomb,
    "h": _cell_honeycomb,
    "kagome": _cell_kagome,
    "k": _cell_kagome,
}

_tvs = {
    "square": _tvs_square,
    "s": _tvs_square,
    "triangle": _tvs_triangle,
    "t": _tvs_triangle,
    "honeycomb": _tvs_honeycomb,
    "h": _tvs_honeycomb,
    "kagome": _tvs_kagome,
    "k": _tvs_kagome,
}


def sitesGenerator(numx, numy, lattice):# {{{
    """
    This function generate sites coordinates specified by the input parameters.

    Parameter:
    ----------
    numx: int
        The number of sites along the first base vector.
    numy: int
        The number of sites along the second base vector.
    lattice: str
        The type of the lattice.
        It can be only "square", "honeycomb", "kagome", "triangle", 's', 'h',
        't', 'k'

    Return:
    -------
    sites: np.array
        The coordinates of the sites.
        Every row represents a site.
    """

    lattice = lattice.lower()
    ps = _cells[lattice]
    tvs = _tvs[lattice]

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
         pcolor='r', psize=6, lcolor='k',lwidth=2):# {{{
    """
    Show the lattice specified by lattice parameter.

    Parameter:
    ----------
    lattice: str
        The type of the lattice.
        It can be only "square", "honeycomb", "kagome", "triangle"
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

    fig, ax = plt.subplots()

    if link:
        bonds = bondsGenerator(sites=sites)
        for (x0, y0), (x1, y1) in bonds:
            ax.plot([x0, x1], [y0, y1], color=lcolor, linewidth=lwidth)

    for x, y in sites:
        ax.plot(x, y, marker='o', color=pcolor, markersize=psize)

    ax.axis("equal")
    ax.axis("off")
    fig.tight_layout()
    plt.show()
# }}}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw some 2D lattice according to the lattice parameter."
    )

    parser.add_argument(
        "lattice",
        help = "Which kind of 2D lattice to draw.",
        choices = (
            "square", "triangle", "honeycomb", "kagome", 's', 't','h', 'k'
        ),
        type=str
    )

    parser.add_argument(
        "--scope",
        type = int,
        default = [9, 9],
        nargs = 2,
        help = "The number of cells in the a0 and a1 direction(default: %(default)s)."
    )

    parser.add_argument(
        "-l",
        "--link",
        action = "store_true",
        help = "Whether to draw the nearest neighbor bond."
    )

    parser.add_argument(
        "--pcolor",
        default = 'k',
        help  ="Color of the point(default : black)."
    )

    parser.add_argument(
        "--lcolor",
        default = 'k',
        help = "Color of the bond(default: black)."
    )

    parser.add_argument(
        "--psize",
        default = 6,
        type = int,
        help = "Size of the point(default: %(default)s)."
    )

    parser.add_argument(
        "--lwidth",
        default = 2,
        type = int,
        help = "Width of the bond(default: %(default)s)."
    )

    args = parser.parse_args()

    kwargs = {
        "lattice": args.lattice,
        "numx": args.scope[0],
        "numy": args.scope[1],
        "link": args.link,
        "pcolor": args.pcolor,
        "psize": args.psize,
        "lcolor": args.lcolor,
        "lwidth": args.lwidth,
    }

    show(**kwargs)
