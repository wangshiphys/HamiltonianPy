"""
This module define Bond class that represents the bond connecting two points.
"""

import numpy as np

from HamiltonianPy.constant import VIEW_AS_ZERO

#Useful constant
_ZOOM = 10000
################

__all__ = ["Bond"]

class Bond:# {{{
    """
    This class provide a unified description of a bond connecting two points.

    Attributes
    -----------
    directional : boolean
        Whether the direction of the bond should be considered. If True, the
        order of p0, p1 is concerned and Bond(p0, p1) != Bond(p1, p0) unless 
        p0 = p1, if False, the order of p0, p1 is not concerned and 
        Bond(p0, p1) = Bond(p1, p0).

    Methods
    --------
    Public methods:
        getEndpoints()
        getLength(ndigits=None)
        getDisplace(ndigits=None)
        getAzimuth(radian=True, ndigits=None)
        opposite()
        oppositeTo(other)
    Special methods:
        __init__(p0, p1, *, directional=True)
        __hash__()
        __eq__()
        __ne__()
        __str__()
    """

    def __init__(self, p0, p1, *, directional=True):# {{{
        """
        Initialize instance of this class.

        Parameters
        ----------
        p0 : np.array
            One endpoint of the bond.
        p1 : np.array
            Another endpoint of the bond. p0 and p1 should be the same shape, 
            and only support shape (1,), (2,), (3,).
        directional: boolean, optional, keyword-only
            Whether the direction of the bond should be considered. If True, the
            order of p0, p1 is concerned and Bond(p0, p1) != Bond(p1, p0) unless 
            p0 = p1, if False, the order of p0, p1 is not concerned and 
            Bond(p0, p1) = Bond(p1, p0).
        """

        if isinstance(p0, np.ndarray) and p0.shape in [(1,), (2,), (3,)]:
            shape = p0.shape
            self._p0 = np.array(p0, copy=True)
            self._p0.setflags(write=False)
        else:
            raise TypeError("The invalid p0 parameter.")
        if isinstance(p1, np.ndarray) and p1.shape == shape:
            self._dim = shape[0]
            self._p1 = np.array(p1, copy=True)
            self._p1.setflags(write=False)
        else:
            raise TypeError("The invalid p1 parameter.")

        self._directional = directional
    # }}}

    @property
    def directional(self):# {{{
        """
        The directional attribute of instance of this class.
        """
        return self._directional
    # }}}

    def getEndpoints(self):# {{{
        """
        Access the p0 and p1 endpoints of the bond.
        """

        p0 = np.array(self._p0, copy=True)
        p1 = np.array(self._p1, copy=True)
        return p0, p1
    # }}}

    def getLength(self, ndigits=None):# {{{
        """
        Return the length of bond.

        Parameters
        ----------
        ndigits : None or int, optional
            Number of digits to preserve after the decimal point.
            default: None
        
        Returns
        -------
        res : The length of the bond.
        """
        
        length = np.linalg.norm(self._p0 - self._p1)
        if isinstance(ndigits, int):
            length = np.around(length, decimals=ndigits)
        return length
    # }}}

    def getDisplace(self, ndigits=None):# {{{
        """
        Return the displace from p0 to p1.
        
        Parameters
        ----------
        ndigits : None or int, optional
            Number of digits to preserve after the decimal point.
            default: None
        
        Returns
        -------
        res : The displace from p0 to p1.

        Raises
        ------
        NotImplementedError : 
            For bond which is not directional, the method is meaningless.
        """
        
        if self._directional:
            dr = self._p1 - self._p0
            if isinstance(ndigits, int):
                dr = np.around(dr, decimals=ndigits)
            return dr
        else:
            errmsg = "This method is not implemented for bond which the"
            errmsg += "direction is not concerned."
            raise NotImplementedError(errmsg)
    # }}}

    def getAzimuth(self, radian=False, ndigits=None):# {{{
        """
        Return the angle between the bond and the coordinate system.
        
        Parameters
        ----------
        radian : boolean, optional
            Determine the unit of angle, radian or degree.
            default: False
        ndigits : None or int, optional
            Number of digits to preserve after the decimal point.
            default: None
        
        Returns
        -------
        res : np.float64 or np.array
            For 1D or 2D system, the azimuth is the angle between the
            directional bond and the x axis. For 3D, there are two angles,
            alpha: the angle between the directional bond and the z axis, beta:
            the angle between the projection of bond on the xy plane and the x
            aixs.
        
        Raises
        ------
        NotImplementedError : 
            For bond which is not directional, the method is meaningless.
        """

        if self._directional:
            if radian:
                coeff = 1.0
            else:
                coeff = 180 / np.pi

            dr = self._p1 - self._p0
            if self._dim == 1:
                if dr[0] >= 0:
                    theta = 0
                else:
                    theta = np.pi
            elif self._dim == 2:
                x, y = dr
                theta = np.arctan2(y, x)
                #If theta is -pi, then it is equivalent to pi.
                if abs(theta + np.pi) < VIEW_AS_ZERO:
                    theta = np.pi
            else:
                x, y, z = dr
                alpha = np.arctan2(np.sqrt(x**2+y**2), z)
                beta = np.arctan2(y, x)
                #If beta is -pi, then it is equivalent to pi.
                if abs(beta + np.pi) < VIEW_AS_ZERO:
                    beta = np.pi
                theta = np.array([alpha, beta])

            theta = coeff * theta
            if isinstance(ndigits, int):
                theta = np.around(theta, decimals=ndigits)
            return theta
        else:
            errmsg = "This method is not implemented for bond which the"
            errmsg += "direction is not concerned."
            raise NotImplementedError(errmsg)
    # }}}

    def _tupleform(self):# {{{
        #Combine the original information of this bond into a tuple.
        #The tuple is then used to calculate the hash code and to define the
        #comparation logic between different instance.

        p0 = tuple([int(i) for i in _ZOOM * self._p0])
        p1 = tuple([int(i) for i in _ZOOM * self._p1])
        res = [p0, p1]
        if not self._directional:
            res.sort()
        res += [self._directional]
        return tuple(res)
    # }}}

    def __hash__(self):# {{{
        """
        Return the hash value of instance of this class.
        """

        return hash(self._tupleform())
    # }}}

    def __eq__(self, other):# {{{
        """
        Define the == operator between instance of this class.
        """

        if isinstance(other, self.__class__):
            return self._tupleform() == other._tupleform()
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
        Return a string that describes the content of the instance.
        """
        
        NDIGITS = 4
        p0, p1 = self.getEndpoints()
        length = self.getLength(ndigits=NDIGITS)
        info = "p0: " + str(p0) + "\n"
        info += "p1: " + str(p1) + "\n"
        info += "length: " + str(length) + "\n"
        if self._directional:
            displace = self.getDisplace(ndigits=NDIGITS)
            azimuth = self.getAzimuth(ndigits=NDIGITS)
            info += "displace: " + str(displace) + "\n"
            info += "azimuth: " + str(azimuth) + "\n"
        return info
    # }}}

    def opposite(self):# {{{
        """
        Return a bond that is opposite to self.
        
        Raises
        ------
        NotImplementedError : 
            For bond which is not directional, the method is meaningless.
        """

        if self._directional:
            return Bond(p0=self._p1, p1=self._p0, directional=True)
        else:
            errmsg = "This method is not implemented for bond which the"
            errmsg += "direction is not concerned."
            raise NotImplementedError(errmsg)
    # }}}

    def oppositeTo(self, other):# {{{
        """
        Return whether the self bond is opposite to the other bond.
        
        Raises
        ------
        NotImplementedError : 
            For bond which is not directional, the method is meaningless.
        """

        if self._directional:
            if isinstance(other, self.__class__):
                return self.opposite().__eq__(other)
            else:
                raise TypeError("The right operand is not instance of this class.")
        else:
            errmsg = "This method is not implemented for bond which the"
            errmsg += "direction is not concerned."
            raise NotImplementedError(errmsg)
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
