"""
Draw some commonly used 2D lattice
"""


from itertools import product
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    "show",
]


# Useful constant in this module
_ERR = 1e-4
################################################################################


_dtype = np.float64

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

_common_cells_info = {
    "s": _square_cell_info,
    "square": _square_cell_info,

    "t": _triangle_cell_info,
    "triangle": _triangle_cell_info,

    "h": _honeycomb_cell_info,
    "honeycomb": _honeycomb_cell_info,

    "k": _kagome_cell_info,
    "kagome": _kagome_cell_info,
}


def _sites_factory(which, num0=1, num1=1):
    try:
        cell_info = _common_cells_info[which]
        cell_points = cell_info["points"]
        cell_vectors = cell_info["vectors"]
    except KeyError:
        raise KeyError("Unrecognized lattice type!")

    dim = cell_points.shape[1]
    dRs = np.matmul(list(product(range(num0), range(num1))), cell_vectors)
    points = np.reshape(dRs[:, np.newaxis, :] + cell_points, newshape=(-1, dim))
    return points


def _bonds_generator(sites):
    # The nearest neighbor distance
    min_dist = min(pdist(sites))
    pairs = np.array(list(cKDTree(sites).query_pairs(r=min_dist + _ERR)))

    # bonds.shape = (n, 2, 2) where `n` is the number of nearest-neighbor
    # bonds. The first dimension specifies different bonds, the second
    # dimension specifies the two end-points of a bond and the third
    # dimension specifies the x,y coordinates of the end-points
    bonds = sites[pairs]
    return bonds


def show(
        which, num0=9, num1=9, *,
        link=False, mc='black', ms=6, ls="solid", lc="black", lw=2
):
    """
    Draw the 2D lattice specified by `which` parameter

    Parameter:
    ----------
    which : str
        The name of the lattice to draw
        Valid values:
            "square" | "s"
            "honeycomb" | "h"
            "kagome" | "k"
            "triangle" | "t"
    num0 : int, optional
        The number of unit-cell along the first translation vector
        default: 10
    num1 : int, optional
        The number of unit-cell along the second translation vector
        default: 10
    link : boolean, optional, keyword-only
        Determine whether to draw the nearest-neighbor bonds
        default: False
    mc : str, optional, keyword-only
        The color of the points
        default: "black"
    ms : float, optional, keyword-only
        The size of the points
        default: 6
    ls : str, optional, keyword-only
        The line style of the nearest-neighbor bonds
        default: "solid"
    lc : str, optional, keyword-only
        The color of the nearest-neighbor bonds
        default: "black"
    lw : float, optional, keyword-only
        The line-width of the nearest-neighbor bonds
        default: 2
    """

    sites = _sites_factory(which=which, num0=num0, num1=num1)

    fig, ax = plt.subplots()
    if link:
        bonds = _bonds_generator(sites=sites)

        # The `x` and `y` parameters of `Axes.plot` can also be
        # 2-dimensional. Then, the columns represent separate data sets.
        ax.plot(
            bonds[:, :, 0].T, bonds[:, :, 1].T,
            marker="o", mec=mc, mfc=mc, ms=ms,
            ls=ls, color=lc, lw=lw
        )
    else:
        ax.plot(
            sites[:, 0], sites[:, 1],
            marker="o", mec=mc, mfc=mc, ms=ms,
            ls="None"
        )

    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw some commonly used 2D lattice."
    )

    parser.add_argument(
        "lattice",
        help = "Which kind of 2D lattice to draw.",
        choices = (
            "square", "triangle", "honeycomb", "kagome", "s", "t", "h", "k"
        ),
        type=str
    )

    parser.add_argument(
        "--scope",
        type = int,
        default = [10, 10],
        nargs = 2,
        help = "The number of unit-cell along the 1st and 2nd translation "
               "vector(default: %(default)s)."
    )

    parser.add_argument(
        "-l", "--link",
        action = "store_true",
        help = "Whether to draw the nearest-neighbor bonds."
    )

    parser.add_argument(
        "--markercolor",
        default = "black",
        help  = "Color of the points(default: %(default)s)."
    )

    parser.add_argument(
        "--markersize",
        default = 6,
        type = float,
        help = "Size of the points(default: %(default)s)."
    )

    parser.add_argument(
        "--linestyle",
        default = "solid",
        help = "Line style of the nearest-neighbor bonds(default: %(default)s)."
    )

    parser.add_argument(
        "--linecolor",
        default = "black",
        help = "Color of the nearest-neighbor bonds(default: %(default)s)."
    )

    parser.add_argument(
        "--linewidth",
        default = 2,
        type = float,
        help = "Line width of the nearest-neighbor bonds(default: %(default)s)."
    )

    args = parser.parse_args()

    kwargs = {
        "which": args.lattice,
        "num0": args.scope[0],
        "num1": args.scope[1],
        "link": args.link,
        "mc": args.markercolor,
        "ms": args.markersize,
        "ls": args.linestyle,
        "lc": args.linecolor,
        "lw": args.linewidth,
    }

    show(**kwargs)
