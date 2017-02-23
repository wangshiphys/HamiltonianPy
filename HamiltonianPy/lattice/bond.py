from numpy.linalg import norm

import numpy as np

from HamiltonianPy.arrayformat import arrayformat

__all__ = ['Bond']

#Useful constant in the module
INDENT = 6
PRECISION = 4
##############################

class Bond:
    """
    This class provide a unified description of the bond between two points.

    Attributes:
    -----------
    dim: int
        The space dimension of the bond.
    start: ndarray
        The starting point of the bond.
    end: ndarray
        The end point of the bond.
    distance: float
        The distance between the two points.
    displace: ndarray
        The vector pointing from start to end.
    theta: float
        In one dimension, theta is the angle between the bond and axis, which is
        either zero or pi.
        In two dimension, theta is the angle between the bond and the x axis,
        which range [-pi, pi]
        In three dimension, theta is the angle between the bond and the z aixs,
        which range [0, pi]
    phi: float
        In one or two dimension, the instance does not have this attribute.
        In three dimension, phi is the angle between the projection of the bond
        in xy-plane and the x axis, which range [-pi, pi].
    """

    def __init__(self, start, end):# {{{
        """
        Initilize this class!
        
        Parameter:
        ----------
        start: ndarray
            The starting point of the bond.
        end: ndarray
            The end point of the bond.
        """

        if not (isinstance(start, np.ndarray) and 
                isinstance(end, np.ndarray)):
            raise TypeError("The input start or end is not ndarray!")

        shape0 = start.shape
        shape1 = end.shape
        if shape0 != shape1:
            raise ValueError("The dimension of the two "
                             "input points does not match!")
        else:
            if shape0 == (1,):
                dim = 1
            elif shape0 == (2,):
                dim = 2
            elif shape0 == (3,):
                dim = 3
            else:
                raise ValueError("The unsupported space dimension!")
        
        self.dim = dim
        self.start = start
        self.end = end
    # }}}

    def setdistance(self):# {{{
        """
        Set the distance attribute.
        """

        distance = norm(self.start - self.end)
        self.distance = np.round(distance, decimals=PRECISION)
    # }}}

    def setdisplace(self):# {{{
        """
        Set the displace attribute.
        """

        displace = self.end - self.start
        self.displace = np.round(displace, decimals=PRECISION)
    # }}}

    def setazimuth(self, tag='degree'):# {{{
        """
        Set the theta and phi attribute.

        Parameter:
        ----------
        tag: string, optional
            Define the form of the angle.
            Default: degree.
        """

        if tag == 'degree':
            trans = 180 / np.pi
        else :
            trans = 1.0
        
        if self.dim == 1:
            if self.end >= self.start:
                self.theta = 0.0
            else:
                self.theta = np.round(np.pi * trans, decimals=PRECISION)
        elif self.dim == 2:
            x, y = self.end - self.start
            self.theta = np.round(np.arctan2(y, x) * trans, decimals=PRECISION)
        elif self.dim == 3:
            x, y, z = self.end - self.start
            theta = np.arctan2(np.sqrt(x**2 + y**2), z) * trans
            self.theta = np.round(theta, decimals=PRECISION)
            self.phi = np.round(np.arctan2(y, x) * trans, decimals=PRECISION)
    # }}}

    def setall(self, tag='degree'):# {{{
        """
        Set all the attributes of the instance of this class!

        tag: string, optional
            Define the form of the angle.
            Default: degree.
        """

        self.setdistance()
        self.setdisplace()
        self.setazimuth(tag=tag)
    # }}}

    def __str__(self):# {{{
        """
        Return the print string of instance of this class.
        """

        self.setall()
        prefix = "\n" + " " * INDENT
        info = prefix + "Space dimension: {0}".format(self.dim)
        info += prefix + "start: " + arrayformat(self.start)
        info += prefix + "end: " + arrayformat(self.end)
        info += prefix + "displace: dr: " + arrayformat(self.displace)
        info += prefix + "distance: ds = {0}".format(self.distance)
        info += prefix + "azimuth: theta = {0} degree.\n".format(self.theta)
        return info
    # }}}


#This is a test!
if __name__ == "__main__":
    p0 = np.array([0, 0])
    p1 = np.array([1, 1])
    bond = Bond(p0, p1)
    print(bond)
