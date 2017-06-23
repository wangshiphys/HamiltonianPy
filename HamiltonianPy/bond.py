"""
This module define Bond class that represents the bond connecting two points.
"""

import numpy as np

from HamiltonianPy.constant import FLOAT_TYPE, NDIGITS

__all__ = ["Bond"]

class Bond:# {{{
    """
    This class provide a unified description of a bond connecting two points.

    Attributes:
    -----------
    p0: np.array
        One endpoint of the bond.
    p1: np.array
        Another endpoint of the bond. p0 and p1 should be the same shape, 
        and only support shape (1,), (2,), (3,).
    dim: int
        The space dimension of the bond.
    directional: boolean
        Whether the direction of the bond should be considered. If True, the
        order of p0, p1 is concerned and Bond(p0, p1) != Bond(p1, p0) unless 
        p0 = p1, if False, the order of p0, p1 is not concerned and 
        Bond(p0, p1) = Bond(p1, p0).

    Methods:
    --------
    Special methods:
        __init__(p0, p1, directional=True)
        __hash__()
        __eq__()
        __ne__()
        __str__()
    General methods:
        getEndpoints()
        getLength(ndigits=None)
        getDisplace(ndigits=None)
        getAzimuth(radian=True, ndigits=None)
        tupleform()
        opposite()
        oppositeTo(other)
    """

    def __init__(self, p0, p1, directional=True):# {{{
        """
        Initialize instance of this class.
        """

        if isinstance(p0, np.ndarray) and p0.shape in [(1,), (2,), (3,)]:
            shape = p0.shape
            self.p0 = np.array(p0[:])
        else:
            raise TypeError("The invalid p0 parameter.")
        if isinstance(p1, np.ndarray) and p1.shape == shape:
            self.dim = shape[0]
            self.p1 = np.array(p1[:])
        else:
            raise TypeError("The invalid p1 parameter.")

        self.directional = directional
    # }}}

    def getEndpoints(self):# {{{
        """
        Access the p0 and p1 attribute of instance of this class.
        """

        p0 = np.array(self.p0[:])
        p1 = np.array(self.p1[:])
        return p0, p1
    # }}}

    def getLength(self, ndigits=None):# {{{
        """
        Return the length of bond.

        Parameter:
        ----------
        ndigits: None or int, optional
            Number of digits to preserve after the decimal point.
            default: None
        
        Return:
        -------
        res: The length of the bond.
        """
        
        length = np.linalg.norm(self.p0 - self.p1)
        if isinstance(ndigits, int):
            length = np.around(length, decimals=ndigits)
        return length
    # }}}

    def getDisplace(self, ndigits=None):# {{{
        """
        Return the displace from p0 to p1.

        For bond which is not directional, the method is meaningless.
        
        Parameter:
        ----------
        ndigits: None or int, optional
            Number of digits to preserve after the decimal point.
            default: None
        
        Return:
        -------
        res: The displace from p0 to p1.
        """
        
        if self.directional:
            dr = self.p1 - self.p0
            if isinstance(ndigits, int):
                dr = np.around(dr, decimals=ndigits)
            return dr
        else:
            raise NotImplementedError("This method should not be implemented "
                  "for bond which the direction is not concerned.")
    # }}}

    def getAzimuth(self, radian=False, ndigits=None):# {{{
        """
        Return the angle between the bond and the coordinate system.

        For bond which is not directional, the method is meaningless.
        
        Parameter:
        ----------
        radian: boolean, optional
            Determine the unit of angle, radian or degree.
            default: False
        ndigits: None or int, optional
            Number of digits to preserve after the decimal point.
            default: None
        
        Return:
        -------
        res: np.float64 or np.array
            For 1D or 2D system, the azimuth is the angle between the
            directional bond and the x axis. For 3D, there are two angles,
            alpha: the angle between the directional bond and the z axis, beta:
            the angle between the projection of bond on the xy plane and the x
            aixs.
        """

        if self.directional:
            if radian:
                coeff = 1.0
            else:
                coeff = 180 / np.pi

            dr = self.p1 - self.p0
            if self.dim == 1:
                if dr[0] >= 0:
                    theta = 0
                else:
                    theta = np.pi
            elif self.dim == 2:
                x, y = dr
                theta = np.arctan2(y, x)
            else:
                x, y, z = dr
                alpha = np.arctan2(np.sqrt(x**2+y**2), z)
                beta = np.arctan2(y, x)
                theta = np.array([alpha, beta])

            theta = coeff * theta
            if isinstance(ndigits, int):
                theta = np.around(theta, decimals=ndigits)
            return theta
        else:
            raise NotImplementedError("This method should not be implemented "
                  "for bond which the direction is not concerned.")
    # }}}

    def tupleform(self):# {{{
        """
        The tuple form of this bond.
        """

        factor = 10 ** NDIGITS
        if self.p0.dtype in FLOAT_TYPE:
            p0 = [int(factor * i) for i in self.p0]
        else:
            p0 = list(self.p0)
        if self.p1.dtype in FLOAT_TYPE:
            p1 = [int(factor * i) for i in self.p1]
        else:
            p1 = list(self.p1)

        res = p0 + p1
        if not self.directional:
            res.sort()
        res += [self.directional]
        return tuple(res)
    # }}}

    def __hash__(self):# {{{
        """
        Return the hash value of instance of this class.
        """

        return hash(self.tupleform())
    # }}}

    def __eq__(self, other):# {{{
        """
        Define the == operator between instance of this class.
        """

        if isinstance(other, self.__class__):
            return self.tupleform() == other.tupleform()
        else:
            raise TypeError("The right operand is not instance of this class.")
    # }}}

    def __ne__(self, other):# {{{
        """
        Define the != operator between instance of this class.
        """

        return not self.__eq__(other)
    # }}}

    def __str__(self):# {{{
        """
        Return a string that descriibles the content of the instance.
        """

        p0, p1 = self.getEndpoints()
        length = self.getLength(ndigits=NDIGITS)
        info = "p0: " + str(p0) + "\n"
        info += "p1: " + str(p1) + "\n"
        info += "length: " + str(length) + "\n"
        if self.directional:
            displace = self.getDisplace(ndigits=NDIGITS)
            azimuth = self.getAzimuth(ndigits=NDIGITS)
            info += "displace: " + str(displace) + "\n"
            info += "azimuth: " + str(azimuth) + "\n"
        return info
    # }}}

    def opposite(self):# {{{
        """
        Return a bond that is opposite to self.
        """

        if self.directional:
            return Bond(p0=self.p1, p1=self.p0, directional=True)
        else:
            raise NotImplementedError("This method should not be implemented "
                  "for bond which the direction is not concerned.")
    # }}}

    def oppositeTo(self, other):# {{{
        """
        Return whether the self bond is opposite to the other bond.
        """

        if self.directional:
            if isinstance(other, self.__class__):
                return self.opposite().__eq__(other)
            else:
                raise TypeError("The right operand is not instance of this class.")
        else:
            raise NotImplementedError("This method should not be implemented "
                  "for bond which the direction is not concerned.")
    # }}}
# }}}



if __name__ == "__main__":
    p0 = np.array([np.exp(-2/3), np.sin(2 * np.pi/5)])
    p1 = np.array([np.sqrt(2)/2, np.sqrt(3)/4])
    bond0 = Bond(p0=p0, p1=p1, directional=True)
    bond1 = Bond(p0=p0, p1=p1, directional=False)
    bond2 = Bond(p0=p1, p1=p0, directional=True)
    bond3 = Bond(p0=p1, p1=p0, directional=False)
    print(bond0)
    print(bond1)
    print(bond2)
    print(bond3)
    print(bond0 == bond1)
    print(bond0 == bond2)
    print(bond0 == bond3)
    print(bond1 == bond2)
    print(bond1 == bond3)
    print(bond2 == bond3)
