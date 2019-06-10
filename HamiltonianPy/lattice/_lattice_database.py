"""
Database for some commonly used cluster
"""


from itertools import product

import numpy as np

from HamiltonianPy.lattice.lattice import Lattice


__all__ = [
    "CHAIN_CELL_POINTS", "CHAIN_CELL_AS", "CHAIN_CELL_BS",
    "CHAIN_CELL_GAMMA", "CHAIN_CELL_ES",

    "SQUARE_CELL_POINTS", "SQUARE_CELL_AS", "SQUARE_CELL_BS",
    "SQUARE_CELL_GAMMA", "SQUARE_CELL_XS", "SQUARE_CELL_MS",

    "TRIANGLE_CELL_POINTS", "TRIANGLE_CELL_AS", "TRIANGLE_CELL_BS",
    "TRIANGLE_CELL_GAMMA", "TRIANGLE_CELL_MS", "TRIANGLE_CELL_KS",

    "HONEYCOMB_CELL_POINTS", "HONEYCOMB_CELL_AS", "HONEYCOMB_CELL_BS",
    "HONEYCOMB_CELL_GAMMA", "HONEYCOMB_CELL_MS", "HONEYCOMB_CELL_KS",

    "KAGOME_CELL_POINTS", "KAGOME_CELL_AS", "KAGOME_CELL_BS",
    "KAGOME_CELL_GAMMA", "KAGOME_CELL_MS", "KAGOME_CELL_KS",

    "CUBIC_CELL_POINTS", "CUBIC_CELL_AS", "CUBIC_CELL_BS",
    "CUBIC_CELL_GAMMA", "CUBIC_CELL_XS", "CUBIC_CELL_MS", "CUBIC_CELL_KS",

    "SQUARE_CROSS_POINTS", "SQUARE_CROSS_AS", "SQUARE_CROSS_BS",
    "SQUARE_CROSS_GAMMA",

    "SQUARE_Z_POINTS", "SQUARE_Z_AS", "SQUARE_Z_BS",
    "SQUARE_Z_GAMMA", "SQUARE_Z_XS", "SQUARE_Z_MS",

    "SQUARE_S_POINTS", "SQUARE_S_AS", "SQUARE_S_BS",
    "SQUARE_S_GAMMA", "SQUARE_S_XS", "SQUARE_S_MS",

    "TRIANGLE_STAR_POINTS", "TRIANGLE_STAR_AS", "TRIANGLE_STAR_BS",
    "TRIANGLE_STAR_GAMMA", "TRIANGLE_STAR_MS", "TRIANGLE_STAR_KS",

    "HONEYCOMB_BENZENE_POINTS", "HONEYCOMB_BENZENE_AS", "HONEYCOMB_BENZENE_BS",
    "HONEYCOMB_BENZENE_GAMMA", "HONEYCOMB_BENZENE_MS", "HONEYCOMB_BENZENE_KS",

    "HONEYCOMB_DIPHENYL_POINTS", "HONEYCOMB_DIPHENYL_AS",
    "HONEYCOMB_DIPHENYL_BS", "HONEYCOMB_DIPHENYL_GAMMA",

    "HONEYCOMB_GEAR_POINTS", "HONEYCOMB_GEAR_AS", "HONEYCOMB_GEAR_BS",
    "HONEYCOMB_GEAR_GAMMA", "HONEYCOMB_GEAR_MS", "HONEYCOMB_GEAR_KS",

    "special_cluster", "lattice_generator",
]


# Calculate the translation vectors in k-space
# from translation vectors in real-space
_BSCalculator = lambda AS: 2 * np.pi * np.linalg.inv(AS.T)

# The following matrices are defined for calculating
# the high-symmetry points in the first Brillouin Zone(1st-BZ)
# When the two real-space translation vectors a0, a1 have the same length and
# the angle between them is 90 degree, then the 1st-BZ is a square. The
# middle-points of the edges of the 1st-BZ are called X-points and the
# corner-points of the 1st-BZ are called M-points. The following two matrices
# are useful for calculating the coordinates of X- and M-points.
_SQUARE_XS_COEFF = [[0, -1], [-1, 0], [0, 1], [1, 0]]
_SQUARE_MS_COEFF = [[-1, -1], [-1, 1], [1, 1], [1, -1]]

# When the two real-space translation vectors a0, a1 have the same length and
# the angle between them is 60 or 120 degree, then the 1st-BZ is a
# regular hexagon(The six edges have the same length).
# The middle-points of the edges of the 1st-BZ are called M-points and the
# corner-points of the 1st-BZ are called K-points. The following four
# matrices are useful for calculating the coordinates of M- and K-points.
_HEXAGON_MS_COEFF_60 = [[0, -1], [-1, -1], [-1, 0], [0, 1], [1, 1], [1, 0]]
_HEXAGON_KS_COEFF_60 = [[1, -1], [-1, -2], [-2, -1], [-1, 1], [1, 2], [2, 1]]
# In this module, the relevant angle between a0 and a1 is chosen to be 60
# degree, so the following two matrices are not used in this module
_HEXAGON_MS_COEFF_120 = [[0, -1], [-1, 0], [-1, 1], [0, 1], [1, 0], [1, -1]]
_HEXAGON_KS_COEFF_120 = [[1, -2], [-1, -1], [-2, 1], [-1, 2], [1, 1], [2, -1]]

_dtype = np.float64

# Database for some commonly used cluster
# In the following, these variables ended with `_POINTS` are the coordinates
# of the points in the cluster; `_AS` are the translation vectors in
# real-space; `_BS` are the translation vectors in reciprocal-space(k-space);
# `_GAMMA` are the center of the 1st-BZ.

# Unit-cell of 1D chain
CHAIN_CELL_POINTS = np.array([[0.0]], dtype=_dtype)
CHAIN_CELL_AS = np.array([[1.0]], dtype=_dtype)
CHAIN_CELL_BS = _BSCalculator(CHAIN_CELL_AS)
CHAIN_CELL_GAMMA = np.array([0.0], dtype=_dtype)
# Endpoints of the 1st-BZ
CHAIN_CELL_ES = np.dot([[-1], [1]], CHAIN_CELL_BS) / 2
################################################################################

# Unit-cell of square lattice
SQUARE_CELL_POINTS = np.array([[0.0, 0.0]], dtype=_dtype)
SQUARE_CELL_AS = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=_dtype)
SQUARE_CELL_BS = _BSCalculator(SQUARE_CELL_AS)
# The corresponding 1st-BZ is a square
# Xs are the middle-points of the edges of the 1st-BZ
# Ms are the corner-points of the 1st-BZ
SQUARE_CELL_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
SQUARE_CELL_XS = np.dot(_SQUARE_XS_COEFF, SQUARE_CELL_BS) / 2
SQUARE_CELL_MS = np.dot(_SQUARE_MS_COEFF, SQUARE_CELL_BS) / 2
################################################################################

# Unit-cell of triangle lattice
TRIANGLE_CELL_POINTS = np.array([[0.0, 0.0]], dtype=_dtype)
TRIANGLE_CELL_AS = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], dtype=_dtype)
TRIANGLE_CELL_BS = _BSCalculator(TRIANGLE_CELL_AS)
# The corresponding 1st-BZ is a regular hexagon
# Ms are the middle-points of the edges of the 1st-BZ
# Ks are the corner-points of the 1st-BZ
TRIANGLE_CELL_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
TRIANGLE_CELL_MS = np.dot(_HEXAGON_MS_COEFF_60, TRIANGLE_CELL_BS) / 2
TRIANGLE_CELL_KS = np.dot(_HEXAGON_KS_COEFF_60, TRIANGLE_CELL_BS) / 3
################################################################################

# Unit-cell of honeycomb lattice
HONEYCOMB_CELL_POINTS = np.array(
    [[0.0, -0.5 / np.sqrt(3)], [0.0, 0.5 / np.sqrt(3)]], dtype=_dtype
)
HONEYCOMB_CELL_AS = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], dtype=_dtype)
HONEYCOMB_CELL_BS = _BSCalculator(HONEYCOMB_CELL_AS)
# The corresponding 1st-BZ is a regular hexagon
# Ms are the middle-points of the edges of the 1st-BZ
# Ks are the corner-points of the 1st-BZ
HONEYCOMB_CELL_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
HONEYCOMB_CELL_MS = np.dot(_HEXAGON_MS_COEFF_60, HONEYCOMB_CELL_BS) / 2
HONEYCOMB_CELL_KS = np.dot(_HEXAGON_KS_COEFF_60, HONEYCOMB_CELL_BS) / 3
################################################################################

# Unit-cell of Kagome lattice
KAGOME_CELL_POINTS = np.array(
    [[0.0, 0.0], [0.25, np.sqrt(3) / 4], [0.5, 0.0]], dtype=_dtype
)
KAGOME_CELL_AS = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]], dtype=_dtype)
KAGOME_CELL_BS = _BSCalculator(KAGOME_CELL_AS)
# The corresponding 1st-BZ is a regular hexagon
# Ms are the middle-points of the edges of the 1st-BZ
# Ks are the corner-points of the 1st-BZ
KAGOME_CELL_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
KAGOME_CELL_MS = np.dot(_HEXAGON_MS_COEFF_60, KAGOME_CELL_BS) / 2
KAGOME_CELL_KS = np.dot(_HEXAGON_KS_COEFF_60, KAGOME_CELL_BS) / 3
################################################################################

# Unit-cell of the cubic lattice
CUBIC_CELL_POINTS = np.array([[0.0, 0.0, 0.0]], dtype=_dtype)
CUBIC_CELL_AS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=_dtype
)
CUBIC_CELL_BS = _BSCalculator(CUBIC_CELL_AS)
# The corresponding 1st-BZ is a cubic
# Xs are the center-points of the surfaces of the 1st-BZ
# Ms are the middle-points of the edges of the 1st-BZ
# Ks are the corner-points of the 1st-BZ
CUBIC_CELL_GAMMA = np.array([0.0, 0.0, 0.0], dtype=_dtype)
CUBIC_CELL_XS = np.dot(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]],
    CUBIC_CELL_BS
) / 2
CUBIC_CELL_MS = np.dot(
    [
        [-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0],
        [-1, 0, -1], [-1, 0, 1], [1, 0, 1], [1, 0, -1],
        [0, -1, -1], [0, -1, 1], [0, 1, 1], [0, 1, -1],
    ], CUBIC_CELL_BS
) / 2
CUBIC_CELL_KS = np.dot(
    [
        [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
        [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1],
    ], CUBIC_CELL_BS
) / 2
################################################################################

# 12-sites cluster division of the square lattice
# The appearance of this cluster looks like a plus symbol
SQUARE_CROSS_POINTS = np.array(
    [
        # The inner 4 points
        [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5],
        # The outer 8 points
        [-0.5, -1.5], [-1.5, -0.5], [-1.5, 0.5], [-0.5, 1.5],
        [0.5, 1.5], [1.5, 0.5], [1.5, -0.5], [0.5, -1.5],
    ], dtype=_dtype
)
SQUARE_CROSS_AS = np.array([[3.0, -2.0], [3.0, 2.0]], dtype=_dtype)
SQUARE_CROSS_BS = _BSCalculator(SQUARE_CROSS_AS)
# The corresponding 1st-BZ is a hexagon but now regular
SQUARE_CROSS_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
################################################################################

# 10-sites cluster division of the square lattice
# The appearance of this cluster looks like the 'z' character
SQUARE_Z_POINTS = np.array(
    [
        # The top three points
        [-1.5, 1.0], [-0.5, 1.0], [0.5, 1.0],
        # The middle four points
        [-1.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0],
        # The bottom three points
        [-0.5, -1.0], [0.5, -1.0], [1.5, -1.0],
    ], dtype=_dtype
)
SQUARE_Z_AS = np.array([[1.0, -3.0], [3.0, 1.0]], dtype=_dtype)
SQUARE_Z_BS = _BSCalculator(SQUARE_Z_AS)
# The corresponding 1st-BZ is a square
# Xs are the middle-points of the edges of the 1st-BZ
# Ms are the corner-points of the 1st-BZ
SQUARE_Z_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
SQUARE_Z_XS = np.dot(_SQUARE_XS_COEFF, SQUARE_Z_BS) / 2
SQUARE_Z_MS = np.dot(_SQUARE_MS_COEFF, SQUARE_Z_BS) / 2
################################################################################

# 10-sites cluster division of the square lattice
# The appearance of this cluster looks like the 's' character
SQUARE_S_POINTS = np.array(
    [
        # The top three points
        [-0.5, 1.0], [0.5, 1.0], [1.5, 1.0],
        # The middle four points
        [-1.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0],
        # The bottom three points
        [-1.5, -1.0], [-0.5, -1.0], [0.5, -1.0],
    ], dtype=_dtype
)
SQUARE_S_AS = np.array([[3.0, -1.0], [1.0, 3.0]], dtype=_dtype)
SQUARE_S_BS = _BSCalculator(SQUARE_S_AS)
# The corresponding 1st-BZ is a square
# Xs are the middle-points of the edges of the 1st-BZ
# Ms are the corner-points of the 1st-BZ
SQUARE_S_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
SQUARE_S_XS = np.dot(_SQUARE_XS_COEFF, SQUARE_S_BS) / 2
SQUARE_S_MS = np.dot(_SQUARE_MS_COEFF, SQUARE_S_BS) / 2
################################################################################

# 13-sites cluster division of the triangle lattice
# It is called Star-of-David(SD)
TRIANGLE_STAR_POINTS = np.array(
    [
        # The central point of the star
        [0.0, 0.0],
        # The inner 6 points of the star
        [-0.5, -np.sqrt(3) / 2], [-1.0, 0.0], [-0.5, np.sqrt(3) / 2],
        [0.5, np.sqrt(3) / 2], [1.0, 0.0], [0.5, -np.sqrt(3) / 2],
        # The outer 6 points of the star
        [0.0, -np.sqrt(3)], [-1.5, -np.sqrt(3) / 2], [-1.5, np.sqrt(3) / 2],
        [0.0, np.sqrt(3)], [1.5, np.sqrt(3) / 2], [1.5, -np.sqrt(3) / 2],
    ], dtype=_dtype
)
TRIANGLE_STAR_AS = np.array(
    [[3.5, np.sqrt(3) / 2], [1.0, 2 * np.sqrt(3)]], dtype=_dtype
)
TRIANGLE_STAR_BS = _BSCalculator(TRIANGLE_STAR_AS)
# The corresponding 1st-BZ is a regular hexagon
# Ms are the middle-points of the edges of the 1st-BZ
# Ks are the corner-points of the 1st-BZ
TRIANGLE_STAR_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
TRIANGLE_STAR_MS = np.dot(_HEXAGON_MS_COEFF_60, TRIANGLE_STAR_BS) / 2
TRIANGLE_STAR_KS = np.dot(_HEXAGON_KS_COEFF_60, TRIANGLE_STAR_BS) / 3
################################################################################

# 6-sites cluster division of the honeycomb lattice
HONEYCOMB_BENZENE_POINTS = np.array(
    [
        [0.0, -1 / np.sqrt(3)], [-0.5, -0.5 / np.sqrt(3)],
        [-0.5, 0.5 / np.sqrt(3)],
        [0.0, 1 / np.sqrt(3)], [0.5, 0.5 / np.sqrt(3)],
        [0.5, -0.5 / np.sqrt(3)],
    ], dtype=_dtype
)
HONEYCOMB_BENZENE_AS = np.array(
    [[1.5, -np.sqrt(3) / 2], [1.5, np.sqrt(3) / 2]], dtype=_dtype
)
HONEYCOMB_BENZENE_BS = _BSCalculator(HONEYCOMB_BENZENE_AS)
# The corresponding 1st-BZ is a regular hexagon
# Ms are the middle-points of the edges of the 1st-BZ
# Ks are the corner-points of the 1st-BZ
HONEYCOMB_BENZENE_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
HONEYCOMB_BENZENE_MS = np.dot(_HEXAGON_MS_COEFF_60, HONEYCOMB_BENZENE_BS) / 2
HONEYCOMB_BENZENE_KS = np.dot(_HEXAGON_KS_COEFF_60, HONEYCOMB_BENZENE_BS) / 3
################################################################################

# 10-sites cluster division of the honeycomb lattice
HONEYCOMB_DIPHENYL_POINTS = np.array(
    [
        [-0.5, -1.0 / np.sqrt(3)], [-1.0, -0.5 / np.sqrt(3)],
        [-1.0, 0.5 / np.sqrt(3)], [-0.5, 1.0 / np.sqrt(3)],
        [0.0, 0.5 / np.sqrt(3)], [0.5, 1.0 / np.sqrt(3)],
        [1.0, 0.5 / np.sqrt(3)], [1.0, -0.5 / np.sqrt(3)],
        [0.5, -1.0 / np.sqrt(3)], [0.0, -0.5 / np.sqrt(3)],
    ], dtype=_dtype
)
HONEYCOMB_DIPHENYL_AS = np.array(
    [[2.5, -np.sqrt(3) / 2], [2.5, np.sqrt(3) / 2]], dtype=_dtype
)
HONEYCOMB_DIPHENYL_BS = _BSCalculator(HONEYCOMB_DIPHENYL_AS)
# The corresponding 1st-BZ is a hexagon but not regular
HONEYCOMB_DIPHENYL_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
################################################################################

# 24-sites cluster division of the honeycomb lattice
# The appearance of this cluster looks like a gear
HONEYCOMB_GEAR_POINTS = np.array(
    [
        # The inner 6 points
        [0.0, -1.0 / np.sqrt(3)], [-0.5, -0.5 / np.sqrt(3)],
        [-0.5, 0.5 / np.sqrt(3)], [0.0, 1.0 / np.sqrt(3)],
        [0.5, 0.5 / np.sqrt(3)], [0.5, -0.5 / np.sqrt(3)],
        # The outer 18 points
        [0.0, -2.0 / np.sqrt(3)], [-0.5, -2.5 / np.sqrt(3)],
        [-1.0, -2.0 / np.sqrt(3)], [-1.0, -1.0 / np.sqrt(3)],
        [-1.5, -0.5 / np.sqrt(3)], [-1.5, 0.5 / np.sqrt(3)],
        [-1.0, 1.0 / np.sqrt(3)], [-1.0, 2.0 / np.sqrt(3)],
        [-0.5, 2.5 / np.sqrt(3)], [0.0, 2.0 / np.sqrt(3)],
        [0.5, 2.5 / np.sqrt(3)], [1.0, 2.0 / np.sqrt(3)],
        [1.0, 1.0 / np.sqrt(3)], [1.5, 0.5 / np.sqrt(3)],
        [1.5, -0.5 / np.sqrt(3)], [1.0, -1.0 / np.sqrt(3)],
        [1.0, -2.0 / np.sqrt(3)], [0.5, -2.5 / np.sqrt(3)],
    ], dtype=_dtype
)
HONEYCOMB_GEAR_AS = np.array(
    [[3.0, -np.sqrt(3)], [3.0, np.sqrt(3)]], dtype=_dtype
)
HONEYCOMB_GEAR_BS = _BSCalculator(HONEYCOMB_GEAR_AS)
# The corresponding 1st-BZ is a regular hexagon
# Ms are the middle-points of the edges of the 1st-BZ
# Ks are the corner-points of the 1st-BZ
HONEYCOMB_GEAR_GAMMA = np.array([0.0, 0.0], dtype=_dtype)
HONEYCOMB_GEAR_MS = np.dot(_HEXAGON_MS_COEFF_60, HONEYCOMB_GEAR_BS) / 2
HONEYCOMB_GEAR_KS = np.dot(_HEXAGON_KS_COEFF_60, HONEYCOMB_GEAR_BS) / 3
################################################################################


_chain_cell_info = {
    "points": CHAIN_CELL_POINTS,
    "vectors": CHAIN_CELL_AS,
}

_square_cell_info = {
    "points": SQUARE_CELL_POINTS,
    "vectors": SQUARE_CELL_AS,
}

_triangle_cell_info = {
    "points": TRIANGLE_CELL_POINTS,
    "vectors": TRIANGLE_CELL_AS,
}

_honeycomb_cell_info = {
    "points": HONEYCOMB_CELL_POINTS,
    "vectors": HONEYCOMB_CELL_AS,
}

_kagome_cell_info = {
    "points": KAGOME_CELL_POINTS,
    "vectors": KAGOME_CELL_AS,
}

_cubic_cell_info = {
    "points": CUBIC_CELL_POINTS,
    "vectors": CUBIC_CELL_AS,
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
    "points": SQUARE_CROSS_POINTS,
    "vectors": SQUARE_CROSS_AS,
}

_square_z_info = {
    "points": SQUARE_Z_POINTS,
    "vectors": SQUARE_Z_AS,
}

_square_s_info = {
    "points": SQUARE_S_POINTS,
    "vectors": SQUARE_S_AS,
}

_triangle_star_info = {
    "points": TRIANGLE_STAR_POINTS,
    "vectors": TRIANGLE_STAR_AS,
}

_honeycomb_benzene_info = {
    "points": HONEYCOMB_BENZENE_POINTS,
    "vectors": HONEYCOMB_BENZENE_AS,
}

_honeycomb_diphenyl_info = {
    "points": HONEYCOMB_DIPHENYL_POINTS,
    "vectors": HONEYCOMB_DIPHENYL_AS,
}

_honeycomb_gear_info = {
    "points": HONEYCOMB_GEAR_POINTS,
    "vectors": HONEYCOMB_GEAR_AS,
}

_special_clusters_info = {
    "square_cross": _square_cross_info,
    "square_12": _square_cross_info,
    "cross": _square_cross_info,

    "square_z": _square_z_info,
    "z": _square_z_info,

    "square_s": _square_s_info,
    "s": _square_s_info,

    "triangle_star": _triangle_star_info,
    "triangle_13": _triangle_star_info,
    "star": _triangle_star_info,
    "SD": _triangle_star_info,

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
    Generate special cluster.

    Parameters
    ----------
    which : str
        Which special lattice to generate.
        Currently supported special lattice:
            "square_cross"
            "square_z"
            "square_s"
            "triangle_star"
            "honeycomb_benzene"
            "honeycomb_diphenyl"
            "honeycomb_gear"
        Alias:
            "square_cross" | "square_12" | "cross";
            "square_z" | "z";
            "square_s" | "s";
            "triangle_star" | "triangle_13" | "star" | "SD";
            "honeycomb_benzene" | "honeycomb_6"| "benzene";
            "honeycomb_diphenyl" | "honeycomb_10" | "diphenyl";
            "honeycomb_gear" | "honeycomb_24" | "gear";

    Returns
    -------
    res : Lattice
        The corresponding cluster.
    """

    try:
        cluster_info = _special_clusters_info[which]
        return Lattice(**cluster_info, name=which)
    except KeyError:
        raise KeyError("Unrecognized special lattice name!")


def lattice_generator(which, num0=1, num1=1, num2=1):
    """
    Generate a common cluster with translation symmetry.

    Parameters
    ----------
    which : str
        Which type of lattice to generate.
        Current supported value:
            "chain" | "square" | "triangle" | "honeycomb" | "kagome" | "cubic"
    num0 : int, optional
        The number of unit cell along the first translation vector.
        Default: 1.
    num1 : int, optional
        The number of unit cell along the second translation vector.
        It only takes effect for 2D and 3D lattice.
        Default: 1.
    num2 : int, optional
        The number of unit cell along the second translation vector.
        It only takes effect for 3D lattice.
        Default: 1.

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


if __name__ == "__main__":
    for cell in ["chain", "square", "triangle", "honeycomb", "kagome"]:
        lattice = lattice_generator(cell)
        lattice.show()
        lattice.show(scope=1)

    for cluster in ["cross", "z", "s", "star", "benzene", "diphenyl", "gear"]:
        lattice = special_cluster(cluster)
        lattice.show()
        lattice.show(1)

    lattice_generator("chain", num0=10).show()
    lattice_generator("square", num0=6, num1=6).show()
    lattice_generator("honeycomb", num0=6, num1=6).show()
    lattice_generator("triangle", num0=6, num1=6).show()
    lattice_generator("kagome", num0=6, num1=6).show()
