"""
lattice
=======

Description of commonly used lattice in condensed matter physics.

Available Classes
-----------------
Lattice
    Description of common lattice with translation symmetry.

Available Functions
-------------------
KPath
    Generate k-points on the path specified by anchor points.
ShowFirstBZ
    Draw the first Brillouin Zone corresponding to the given real-space
    translation vectors.
special_cluster
    Generate special cluster.
lattice_generator
    Generate a common cluster with translation symmetry.
set_float_point_precision
    Set the float point precision for processing float point number.

Available Constants
-------------------
CHAIN_CELL_POINTS
    Coordinate of point in the unit-cell of 1D chain.
CHAIN_CELL_AS
    Real-space translation vector of 1D chain.
CHAIN_CELL_BS
    Reciprocal-space translation vector of 1D chain.
CHAIN_CELL_GAMMA
    Center of the 1st-BZ.
CHAIN_CELL_ES
    Endpoints of the 1st-BZ.

SQUARE_CELL_POINTS
    Coordinates of point in the unit-cell of square lattice.
    The corresponding 1st-BZ is a square.
SQUARE_CELL_AS
    Real-space translation vectors of square lattice.
SQUARE_CELL_BS
    Reciprocal-space translation vectors of square lattice.
SQUARE_CELL_GAMMA
    Center of the 1st-BZ.
SQUARE_CELL_XS
    Middle-points of the edges of the 1st-BZ.
SQUARE_CELL_MS
    Corner-points of the 1st-BZ.

TRIANGLE_CELL_POINTS
    Coordinates of point in the unit-cell of triangle lattice.
    The corresponding 1st-BZ is a regular hexagon.
TRIANGLE_CELL_AS
    Real-space translation vectors of triangle lattice.
TRIANGLE_CELL_BS
    Reciprocal-space translation vectors of triangle lattice.
TRIANGLE_CELL_GAMMA
    Center of the 1st-BZ.
TRIANGLE_CELL_MS
    Middle-points of the edges of the 1st-BZ.
TRIANGLE_CELL_KS
    Corner-points of the 1st-BZ.

HONEYCOMB_CELL_POINTS
    Coordinates of points in the unit-cell of honeycomb lattice.
    The corresponding 1st-BZ is a regular hexagon.
HONEYCOMB_CELL_AS
    Real-space translation vectors of honeycomb lattice.
HONEYCOMB_CELL_BS
    Reciprocal-space translation vectors of honeycomb lattice.
HONEYCOMB_CELL_GAMMA
    Center of the 1st-BZ.
HONEYCOMB_CELL_MS
    Middle-points of the edges of the 1st-BZ.
HONEYCOMB_CELL_KS
    Corner-points of the 1st-BZ.

KAGOME_CELL_POINTS
    Coordinates of points in the unit-cell of kagome lattice.
    The corresponding 1st-BZ is a regular hexagon.
KAGOME_CELL_AS
    Real-space translation vectors of kagome lattice.
KAGOME_CELL_BS
    Reciprocal-space translation vectors of kagome lattice.
KAGOME_CELL_GAMMA
    Center of the 1st-BZ.
KAGOME_CELL_MS
    Middle-points of the edges of the 1st-BZ.
KAGOME_CELL_KS
    Corner-points of the 1st-BZ.

CUBIC_CELL_POINTS
    Coordinates of point in the unit-cell of cubic lattice.
    The corresponding 1st-BZ is a cubic.
CUBIC_CELL_AS
    Real-space translation vectors of cubic lattice.
CUBIC_CELL_BS
    Reciprocal-space translation vectors of cubic lattice.
CUBIC_CELL_GAMMA
    Center of the 1st-BZ.
CUBIC_CELL_XS
    Center-points of the surfaces of the 1st-BZ.
CUBIC_CELL_MS
    Middle-points of the edges of the 1st-BZ.
CUBIC_CELL_KS
    Corner-points of the 1st-BZ.

SQUARE_CROSS_POINTS
    Coordinates of points in the 'square_cross' cluster.
    The 'square_cross' cluster is a 12-sites division of the square lattice.
    The appearance of this cluster looks like a plus symbol. The
    corresponding 1st-BZ is a hexagon but not regular.
SQUARE_CROSS_AS
    Real-space translation vectors of the 'square_cross' cluster.
SQUARE_CROSS_BS
    Reciprocal-space translation vectors of the 'square_cross' cluster.
SQUARE_CROSS_GAMMA
    Center of the 1st-BZ.

SQUARE_Z_POINTS
    Coordinates of points in the 'square_z' cluster.
    The 'square_z' cluster is a 10-sites division of the square lattice.
    The appearance of this cluster looks like the 'z' character. The
    corresponding 1st-BZ is a square.
SQUARE_Z_AS
    Real-space translation vectors of the 'square_z' cluster.
SQUARE_Z_BS
    Reciprocal-space translation vectors of the 'square_z' cluster.
SQUARE_Z_GAMMA
    Center of the 1st-BZ.
SQUARE_Z_XS
    Middle-points of the edges of the 1st-BZ.
SQUARE_Z_MS
    Corner-points of the 1st-BZ.

SQUARE_S_POINTS
    Coordinates of points in the 'square_s' cluster.
    The 'square_s' cluster is a 10-sites division of the square lattice.
    The appearance of this cluster looks like the 's' character. The
    corresponding 1st-BZ is a square.
SQUARE_S_AS
    Real-space translation vectors of the 'square_s' cluster.
SQUARE_S_BS
    Reciprocal-space translation vectors of the 'square_s' cluster.
SQUARE_S_GAMMA
    Center of the 1st-BZ.
SQUARE_S_XS
    Middle-points of the edges of the 1st-BZ.
SQUARE_S_MS
    Corner-points of the 1st-BZ.

TRIANGLE_STAR_POINTS
    Coordinates of points in the 'triangle_star' cluster.
    The 'triangle_star' cluster is a 13-sites division of the triangle lattice.
    The appearance of this cluster looks like a star and it is called
    Star-of-David(SD). The corresponding 1st-BZ is a regular hexagon.
TRIANGLE_STAR_AS
    Real-space translation vectors of the 'triangle_star' cluster.
TRIANGLE_STAR_BS
    Reciprocal-space translation vectors of the 'triangle_star' cluster.
TRIANGLE_STAR_GAMMA
    Center of the 1st-BZ.
TRIANGLE_STAR_MS
    Middle-points of the edges of the 1st-BZ.
TRIANGLE_STAR_KS
    Corner-points of the 1st-BZ.

HONEYCOMB_BENZENE_POINTS
    Coordinates of points in the 'honeycomb_benzene' cluster.
    The 'honeycomb_benzene' cluster is a 6-sites division of the honeycomb
    lattice. The appearance of this cluster looks like a benzene ring. The
    corresponding 1st-BZ is a regular hexagon.
HONEYCOMB_BENZENE_AS
    Real-space translation vectors of the 'honeycomb_benzene' cluster.
HONEYCOMB_BENZENE_BS
    Reciprocal-space translation vectors of the 'honeycomb_benzene' cluster.
HONEYCOMB_BENZENE_GAMMA
    Center of the 1st-BZ.
HONEYCOMB_BENZENE_MS
    Middle-points of the edges of the 1st-BZ.
HONEYCOMB_BENZENE_KS
    Corner-points of the 1st-BZ.

HONEYCOMB_DIPHENYL_POINTS
    Coordinates of points in the 'honeycomb_diphenyl' cluster.
    The 'honeycomb_diphenyl' cluster is a 10-sites division of the honeycomb
    lattice. The appearance of this cluster looks like a diphenyl. The
    corresponding 1st-BZ is a hexagon but not regular.
HONEYCOMB_DIPHENYL_AS
    Real-space translation vectors of the 'honeycomb_diphenyl' cluster.
HONEYCOMB_DIPHENYL_BS
    Reciprocal-space translation vectors of the 'honeycomb_diphenyl' cluster.
HONEYCOMB_DIPHENYL_GAMMA
    Center of the 1st-BZ.

HONEYCOMB_GEAR_POINTS
    Coordinates of points in the 'honeycomb_gear' cluster.
    The 'honeycomb_gear' cluster is a 24-sites division of the honeycomb
    lattice. The appearance of this cluster looks like a gear. The
    corresponding 1st-BZ is a regular hexagon.
HONEYCOMB_GEAR_AS
    Real-space translation vectors of the 'honeycomb_gear' cluster.
HONEYCOMB_GEAR_BS
    Reciprocal-space translation vectors of the 'honeycomb_gear' cluster.
HONEYCOMB_GEAR_GAMMA
    Center of the 1st-BZ.
HONEYCOMB_GEAR_MS
    Middle-points of the edges of the 1st-BZ.
HONEYCOMB_GEAR_KS
    Corner-points of the 1st-BZ.
"""


from ._lattice_database import *
from .lattice import *


__all__ = [
    "Lattice",
    "KPath",
    "ShowFirstBZ",
]
__all__ += _lattice_database.__all__
