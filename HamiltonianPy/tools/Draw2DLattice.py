"""
Draw some commonly used 2D lattice.

Currently supported lattice:
    square-lattice
    triangle-lattice
    honeycomb-lattice
    kagome-lattice
"""


import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist


__all__ = [
    "show",
]


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
    except KeyError:
        raise KeyError("Unrecognized lattice type!")

    cell_points = cell_info["points"]
    cell_vectors = cell_info["vectors"]
    dim = cell_points.shape[1]
    dRs = np.matmul(
        list(product(range(-num0, num0+1), range(-num1, num1+1))),
        cell_vectors
    )
    return np.reshape(dRs[:, np.newaxis, :] + cell_points, newshape=(-1, dim))


def _bonds_generator(sites):
    # The nearest neighbor distance
    min_dist = min(pdist(sites)) + (1E-4)
    pairs = list(cKDTree(sites).query_pairs(r=min_dist))
    return sites[pairs]


def show(
        which, num0=8, num1=8, *,
        mc="black", ms=6, lc="gray", lw=2, ls="solid",
        link=False, trim=False,
        save=True, path="", name=None,
):
    """
    Draw the 2D lattice specified by `which` parameter.

    Parameter
    ---------
    which : ["s", "square", "t", "triangle", "h", "honeycomb", "k" or "kagome"]
        The name of the lattice.
    num0, num1 : int, optional
        The number of unit-cell along the 1st and 2nd translation vector.
        Default: 8.
    mc : str, optional, keyword-only
        The color of lattice-sites.
        Default: "black".
    ms : int or float, optional, keyword-only
        The size of lattice-sites.
        Default: 6.
    lc : str, optional, keyword-only
        The color of nearest-neighbor bonds.
        Default: "gray".
    lw : int or float, optional, keyword-only
        The line-width of nearest-neighbor bonds.
        Default: 2.
    ls : str, optional, keyword-only
        The line-style of nearest-neighbor bonds.
        Default: "solid".
    link : bool, optional, keyword-only
        Whether to draw nearest-neighbor bonds.
        Default: False.
    trim : bool, optional, keyword-only
        Whether to trim the figure into square form.
        Default: False.
    save : bool, optional, keyword-only
        Whether to save the figure.
        Default: True.
    path : str, optional, keyword-only
        Where to save the figure. This parameter only takes effect when
        `save` is set to True. Empty string(default) implies the current
        working directory.
        Default: ''(empty-string).
    name : str or None, optional, keyword-only
        The name of the saved figure. This parameter only takes effect when
        `save` is set to True. If `name` is set to `None`, then the figure name
        will be determined according to the `which` parameter.
        Default: None.
    """

    sites = _sites_factory(which=which, num0=num0, num1=num1)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_aspect("equal")

    if link:
        bonds = _bonds_generator(sites=sites)
        ax.plot(bonds[:, :, 0].T, bonds[:, :, 1].T, color=lc, lw=lw, ls=ls)
    ax.plot(
        sites[:, 0], sites[:, 1], marker="o", color=mc, ms=ms, ls="None"
    )

    if trim:
        left, right = ax.get_xlim()
        bottom, top = ax.get_ylim()
        ax.set_xlim(left/2, right/2)
        ax.set_ylim(bottom/2, top/2)

    try:
        plt.get_current_fig_manager().window.showMaximized()
    except Exception:
        pass
    plt.show()

    if save:
        Path(path).mkdir(parents=True, exist_ok=True)
        if name is None:
            if which in ("s", "square"):
                name = "square.pdf"
            elif which in ("t", "triangle"):
                name = "triangle.pdf"
            elif which in ("h", "honeycomb"):
                name = "honeycomb.pdf"
            elif which in ("k", "kagome"):
                name = "kagome.pdf"
            else:
                name = which + ".pdf"
        fig.savefig(path + name, dpi=200)
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw some commonly used 2D lattice."
    )

    parser.add_argument(
        "lattice", type=str, help="Which kind of lattice to draw.",
        choices=(
            "square", "triangle", "honeycomb", "kagome", "s", "t", "h", "k",
        ),
    )
    parser.add_argument(
        "--scope", type=int, default=[8, 8], nargs=2,
        help="The number of unit-cell along the 1st and 2nd translation "
               "vector(default: %(default)s)."
    )
    parser.add_argument(
        "--mc", type=str, default="black",
        help="The color of lattice-sites(default: %(default)s)."
    )
    parser.add_argument(
        "--ms", type=float, default=6,
        help="The size of lattice-sites(default: %(default)s)."
    )
    parser.add_argument(
        "--lc", type=str, default="gray",
        help="The color of nearest-neighbor bonds(default: %(default)s)."
    )
    parser.add_argument(
        "--lw", type=float, default=2,
        help="The line-width of nearest-neighbor bonds(default: %(default)s)."
    )
    parser.add_argument(
        "--ls", type=str, default="solid",
        help="The line-style of nearest-neighbor bonds(default: %(default)s)."
    )
    parser.add_argument(
        "-l", "--link", action="store_true",
        help="Draw the nearest-neighbor bonds."
    )
    parser.add_argument(
        "--trim", action="store_true", help="Trim the figure into square form."
    )
    parser.add_argument(
        "--nosave", action="store_true", help="Don't save the figure."
    )
    parser.add_argument(
        "--path", type=str, default="", help="Where to save the figure."
    )
    parser.add_argument(
        "--name", default=None, help="The name of the saved figure."
    )

    args = parser.parse_args()
    kwargs = {
        "which": args.lattice, "num0": args.scope[0], "num1": args.scope[1],
        "mc": args.mc, "ms": args.ms,
        "lc": args.lc, "ls": args.ls, "lw": args.lw,
        "link": args.link, "trim": args.trim,
        "save": (not args.nosave), "path": args.path, "name": args.name,
    }
    show(**kwargs)
