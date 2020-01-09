"""
Polygon in 2 dimension.
"""


__all__ = [
    "Location",
    "Polygon",
]


from enum import Enum, auto

import line2d
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist


class Location(Enum):
    """
    Enumeration class that describe the location relationship between a point
    and a polygon in 2 dimension.
    """

    # The point is located at the corner of the polygon
    CORNER = auto()
    # The point is on the boundary of the polygon
    BOUNDARY = auto()
    # The point is inside the polygon
    INSIDE = auto()
    # The point is outside the polygon
    OUTSIDE = auto()


class Polygon:
    """
    A class that describe polygon in 2 dimension.

    Attributes
    ----------
    vertices : array with shape (n, 2)
        The coordinates of the vertices.
    vertices_num : int
        The number of vertices.
    """

    def __init__(self, vertices, *, tol=1E-10):
        """
        Customize the newly created instance.

        The boundary of the polygon is defined as follow:
            vertices[0] -> vertices[1] -> ... -> vertices[n-1] -> vertices[0]
        The order of vertices is important and left to the user to ensure
        the vertices are in the right order.

        Parameters
        ----------
        vertices : array like with shape (n, 2)
            The coordinates of the vertices of the polygon.
            Every row correspond to a vertex. To form a polygon, at least
            three vertices are required (n >= 3).
        tol : float, keyword-only, optional
            The tolerance for viewing float-point numbers as zero.
            Default: 1E-10.
        """

        vertices = np.array(vertices, copy=True)
        assert vertices.ndim == 2
        shape = vertices.shape
        assert shape[0] >= 3, "At least three vertices are required"
        assert shape[1] == 2, "Support only for 2D"

        if np.any(pdist(vertices) <= tol):
            raise ValueError(
                "There are duplicate points in the given `vertices`!"
            )

        self._vertices = vertices
        self._vertices_num = vertices.shape[0]

    @property
    def vertices(self):
        """
        The `vertices` attribute.
        """

        return np.array(self._vertices, copy=True)

    @property
    def vertices_num(self):
        """
        The `vertices_num` attribute.
        """

        return self._vertices_num

    def ShowPolygon(self):
        """
        Show the polygon as well as the vertices.
        """

        vertices = self._vertices
        boundary = np.concatenate([vertices, vertices[[0]]])

        fig, ax = plt.subplots(num="Polygon")
        ax.plot(boundary[:, 0], boundary[:, 1])
        ax.plot(vertices[:, 0], vertices[:, 1], ls="", marker="o")
        ax.set_aspect("equal")
        plt.show()
        plt.close("all")

    def LocationRelation(self, point, ref_point, *, tol=1E-10):
        """
        Determine the location relationship between the given `point`
        and the polygon.

        Parameters
        ----------
        point : array with shape (2, )
            The coordinate of the point.
        ref_point : array with shape (2, )
            An arbitrary point inside the polygon, not on the boundary.
        tol : float, keyword-only, optional
            The tolerance for viewing float-point numbers as zero.
            Default: 1E-10.

        Returns
        -------
        res : member of `Location` enumeration.
            The location relationship between the point and the polygon.
        """

        distances = np.linalg.norm(self._vertices - np.array(point), axis=-1)
        if np.any(distances <= tol):
            return Location.CORNER

        num = self._vertices_num
        vertices = self._vertices
        for i in range(num):
            p0 = vertices[i]
            p1 = vertices[(i + 1) % num]
            line = line2d.Line2D(p0, p1, tol=tol)
            location0 = line.LocationRelation(point, tol=tol)
            location1 = line.LocationRelation(ref_point, tol=tol)
            if location0 is line2d.Location.ON:
                tmp = [p0[0], p1[0]]
                x, y = point
                if np.min(tmp) <= x <= np.max(tmp):
                    return Location.BOUNDARY
                else:
                    return Location.OUTSIDE
            if not (location0 is location1):
                return Location.OUTSIDE
        return Location.INSIDE

    def ShowLocationRelation(self, point):
        """
        Show the location relationship between the given `point` and the
        polygon.
        """

        vertices = self._vertices
        boundary = np.concatenate([vertices, vertices[[0]]])

        fig, ax = plt.subplots(num="Polygon")
        ax.plot(boundary[:, 0], boundary[:, 1])
        ax.plot(vertices[:, 0], vertices[:, 1], ls="", marker="o")
        ax.plot(point[0], point[1], ls="", marker="s")
        ax.set_aspect("equal")
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    from lattice import TRIANGLE_CELL_KS, TRIANGLE_CELL_BS

    BZBoundary = TRIANGLE_CELL_KS[[*range(6), 0]]
    vertices = TRIANGLE_CELL_KS
    polygon = Polygon(vertices)
    polygon.ShowPolygon()
    polygon.ShowLocationRelation([0, 0])
    polygon.ShowLocationRelation(TRIANGLE_CELL_KS[0])
    polygon.ShowLocationRelation([-5, 0])

    num1 = 10
    num2 = 12
    kpoints_corner = []
    kpoints_inside = []
    kpoints_outside = []
    kpoints_boundary = []
    for i in range(-num1, num1 + 1):
        for j in range(-num2, num2 + 1):
            kpoint = np.dot([i/num1, j/num2], TRIANGLE_CELL_BS)
            location = polygon.LocationRelation(kpoint, [0, 0])
            if location is Location.CORNER:
                kpoints_corner.append(kpoint)
            elif location is Location.INSIDE:
                kpoints_inside.append(kpoint)
            elif location is Location.OUTSIDE:
                kpoints_outside.append(kpoint)
            else:
                kpoints_boundary.append(kpoint)

    print("The number of corner points:", len(kpoints_corner))
    print("The number of boundary points:", len(kpoints_boundary))
    print("The number of inside points:", len(kpoints_inside))
    print("The number of outside points:", len(kpoints_outside))

    fig, ax = plt.subplots()
    ax.plot(BZBoundary[:, 1], BZBoundary[:, 0])
    for kpoints in [
        kpoints_corner, kpoints_inside, kpoints_outside, kpoints_boundary
    ]:
        if kpoints:
            kpoints = np.array(kpoints)
            ax.plot(kpoints[:, 1], kpoints[:, 0], ls="", marker="o")
    ax.set_aspect("equal")
    plt.show()
    plt.close("all")
