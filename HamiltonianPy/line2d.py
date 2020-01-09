"""
Line in 2 dimension.
"""


__all__ = [
    "Location",
    "Line2D",
]


from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np


class Location(Enum):
    """
    Enumeration class that describe the location relationship between a point
    and a line in 2 dimension.
    """

    # The point is on the line
    ON = auto()
    # The point is below the line
    BELOW = auto()
    # The point is above the line
    ABOVE = auto()
    # The point is on the left side of the line
    LEFT = auto()
    # The point is on the right side of the line
    RIGHT = auto()


class Line2D:
    """
    A class that describe line in 2 dimension.

    Attributes
    ----------
    IsHorizontal : bool
        Whether the line is a horizontal line.
    IsVertical : bool
        Whether the line is a vertical line.

    Examples
    --------
    >>> from HamiltonianPy.line2d import Line2D
    >>> line = Line2D([0, 0], [1, 1])
    >>> line
    Line2D(p0=(0, 0), p1=(1, 1))
    >>> line.IsHorizontal
    False
    >>> line.IsVertical
    False
    >>> line.LocationRelation([0, 1])
    <Location.ABOVE: 3>
    >>> line.LocationRelation([0.5, 0.5])
    <Location.ON: 1>
    >>> line.LocationRelation([1, 0])
    <Location.BELOW: 2>
    >>> line(0.5, which="x")
    (0.5, 0.5)
    """

    def __init__(self, p0, p1, *, tol=1E-10):
        """
        Customize the newly created instance.

        Parameters
        ----------
        p0, p1 : list, tuple or 1D np.ndarray
            Two representative points on the line.
            `p0` and `p1` should be 1D array with length 2.
        tol : float, keyword-only, optional
            The tolerance for viewing float-point numbers as zero.
            Default: 1E-10.
        """

        p0 = np.array(p0, copy=True)
        p1 = np.array(p1, copy=True)
        assert p0.shape == (2, )
        assert p1.shape == (2, )

        dr = p1 - p0
        if np.all(np.abs(dr) <= tol):
            raise ValueError("The given `p0` and `p1`are identical!")

        self._p0 = p0
        self._p1 = p1
        self._dr = dr
        self._IsHorizontal = True if np.abs(dr[1]) <= tol else False
        self._IsVertical = True if np.abs(dr[0]) <= tol else False

    def __repr__(self):
        """
        Official string representation of the line.
        """

        return "Line2D(p0={0!r}, p1={1!r})".format(
            tuple(self._p0), tuple(self._p1)
        )

    def __str__(self):
        """
        Return the formula of the line.
        """

        x0, y0 = self._p0
        dx, dy = self._dr
        if self._IsHorizontal:
            return "y = {0:.4f}".format(y0)
        elif self._IsVertical:
            return "x = {0:.4f}".format(x0)
        else:
            return "y = {0:.4f}x + {1:.4f}".format(dy / dx, y0 - x0 * dy / dx)

    @property
    def IsHorizontal(self):
        """
        Whether the line is a horizontal line.
        """

        return self._IsHorizontal

    @property
    def IsVertical(self,):
        """
        Whether the line is a vertical line.
        """

        return self._IsVertical

    def PointsFactory(self, ratios=None):
        """
        Generate points on the line according to the given `ratios`.

        The points are generated using the following formula:
            (x, y) = (x0, y0) + ratio * (x1 - x0, y1 - y0)

        Parameters
        ----------
        ratios : 1D array, optional
            A collection of ratios.
            The default value `None` implies that:
                `ratios = np.linspace(0, 1, 101, endpoint=True)`.
            Default: None.

        Returns
        -------
        points : 2D array
            A collection of points.
        """

        if ratios is None:
            ratios = np.linspace(0, 1, num=101, endpoint=True)
        else:
            ratios = np.array(ratios)
            assert ratios.ndim == 1

        return self._p0 + ratios[:, np.newaxis] * self._dr

    def ShowLine(self):
        """
        Show the line as well as the two representative points `p0` and `p1`.
        """

        x0, y0 = self._p0
        x1, y1 = self._p1
        dx, dy = self._dr
        points = self.PointsFactory(np.linspace(-1, 2, num=301, endpoint=True))
        if self._IsHorizontal:
            title = "y = {0:.4f}".format(y0)
        elif self._IsVertical:
            title = "x = {0:.4f}".format(x0)
        else:
            title = "y = {0:.4f}x + {1:.4f}".format(dy / dx, y0 - x0 * dy / dx)

        fig, ax = plt.subplots()
        ax.plot(points[:, 0], points[:, 1])
        ax.plot(x0, y0, ls="", marker="o", ms=30, clip_on=False)
        ax.plot(x1, y1, ls="", marker="o", ms=30, clip_on=False)
        ax.set_title(title)
        ax.text(x0, y0, "$p_0$", ha="center", va="center", fontsize="xx-large")
        ax.text(x1, y1, "$p_1$", ha="center", va="center", fontsize="xx-large")
        # ax.set_aspect("equal")
        # ax.set_axis_off()
        plt.show()
        plt.close("all")

    def LocationRelation(self, point, *, tol=1E-10):
        """
        Determine the location relationship between the given `point` and
        the line.

        Parameters
        ----------
        point : array with shape (2, )
            The coordinate of the point.
        tol : float, keyword-only, optional
            The tolerance for viewing float-point numbers as zero.
            Default: 1E-10.

        Returns
        -------
        res : member of `Location` enumeration.
            The location relationship between the point and the line.
        """

        x, y = point
        x0, y0 = self._p0
        dx, dy = self._dr

        if self._IsHorizontal:
            if np.abs(y - y0) <= tol:
                return Location.ON
            elif y < y0:
                return Location.BELOW
            else:
                return Location.ABOVE
        elif self._IsVertical:
            if np.abs(x - x0) <= tol:
                return Location.ON
            elif x < x0:
                return Location.LEFT
            else:
                return Location.RIGHT
        else:
            y_on = (dy / dx) * (x - x0) + y0
            if np.abs(y - y_on) <= tol:
                return Location.ON
            elif y < y_on:
                return Location.BELOW
            else:
                return Location.ABOVE

    def ShowLocationRelation(self, point):
        """
        Show the location relationship between the given `point` and the line.
        """

        x, y = point
        x0, y0 = self._p0
        dx, dy = self._dr
        points = self.PointsFactory(np.linspace(-1, 2, num=301, endpoint=True))
        if self._IsHorizontal:
            title = "y = {0:.4f}".format(y0)
        elif self._IsVertical:
            title = "x = {0:.4f}".format(x0)
        else:
            title = "y = {0:.4f}x + {1:.4f}".format(dy / dx, y0 - x0 * dy / dx)

        fig, ax = plt.subplots()
        ax.plot(points[:, 0], points[:, 1])
        ax.plot(x, y, ls="", marker="o", ms=30, clip_on=False)
        ax.text(x, y, "p", ha="center", va="center", fontsize="xx-large")
        ax.set_title(title)
        # ax.set_aspect("equal")
        # ax.set_axis_off()
        plt.show()
        plt.close("all")

    def Coordinate(self, known, which="x"):
        """
        Calculate the coordinate of the point on the line from the given
        `known` and `which`.

        Parameters
        ---------
        known : int or float
            The known coordinate.
        which : ["x" | "y"], str, optional
            If set to "x", the given `known` is interpreted as x-coordinate;
            if set to "y", the given `known` is interpreted as y-coordinate.

        Returns
        -------
        (x, y) : tuple
            The x, y coordinate of the point on the line.
        """

        assert which in ("x", "y")
        x0, y0 = self._p0
        dx, dy = self._dr

        if self._IsHorizontal:
            if which == "x":
                x = known
                y = y0
            else:
                raise RuntimeError(
                    "Can't calculate `x` from known `y` for horizontal line!"
                )
        elif self._IsVertical:
            if which == "y":
                x = x0
                y = known
            else:
                raise RuntimeError(
                    "Can't calculate `y` from known `x` for vertical line!"
                )
        else:
            if which == "x":
                x = known
                y = (dy / dx) * (known - x0) + y0
            else:
                x = (dx / dy) * (known - y0) + x0
                y = known
        return (x, y)

    __call__ = Coordinate
