"""
A test script for the line2d module.
"""


import numpy as np
import pytest

from HamiltonianPy.line2d import Location, Line2D

class TestLine2D:
    def test_init(self):
        with pytest.raises(AssertionError):
            tmp = Line2D([0, 0, 0], [0, 0, 0])
        with pytest.raises(ValueError, match="identical"):
            tmp = Line2D([0, 0], [0, 0])

        line = Line2D([0, 1], [0, 2])
        assert line.IsVertical
        assert not line.IsHorizontal

        line = Line2D((0, 1), (2, 1))
        assert not line.IsVertical
        assert line.IsHorizontal

    def test_LocationRelation(self):
        p0 = [-1, -2]
        p1 = [0, 5]
        p2 = [1, 3]
        line = Line2D([0, 0], [0, 1])
        assert line.LocationRelation(p0) is Location.LEFT
        assert line.LocationRelation(p1) is Location.ON
        assert line.LocationRelation(p2) is Location.RIGHT

        p0 = [-1, -1]
        p1 = [1, 0]
        p2 = [2, 3]
        line = Line2D([0, 0], [5, 0])
        assert line.LocationRelation(p0) is Location.BELOW
        assert line.LocationRelation(p1) is Location.ON
        assert line.LocationRelation(p2) is Location.ABOVE

        line = Line2D([0, 1], [1, 0])
        p0 = [0, 0]
        p1 = [0.5, 0.5]
        p2 = [1, 1]
        assert line.LocationRelation(p0) is Location.BELOW
        assert line.LocationRelation(p1) is Location.ON
        assert line.LocationRelation(p2) is Location.ABOVE

    def test_Coordinate(self):
        line = Line2D([0, 0], [0, 1])
        x, y = line(5, which="y")
        assert x == 0 and y == 5
        with pytest.raises(
                RuntimeError,
                match="Can't calculate `y` from known `x` for vertical line!"
        ):
            line(5, which="x")

        line = Line2D([0, 0], [1, 0])
        x, y = line(5, which="x")
        assert x == 5 and y == 0
        with pytest.raises(
            RuntimeError,
            match="Can't calculate `x` from known `y` for horizontal line!"
        ):
            line.Coordinate(5, which="y")

        line = Line2D([0, 1], [1, 0])
        x, y = line(2, which="x")
        assert x == 2 and y == -1
        x, y = line.Coordinate(2, which="y")
        assert x == -1 and y == 2
