"""
Bond class that describes the bond connecting two points
"""


__all__ = [
    "Bond",
]


import numpy as np


# Useful constant
_ZOOM = 10000
_VIEW_AS_ZERO = 1E-4
################################################################################


class Bond:
    """
    Bond connecting two points

    Attributes
    ----------
    directional : boolean
        Whether the direction of the bond should be considered.
        If set to True, then the order of p0, p1 is concerned and
        Bond(p0, p1) != Bond(p1, p0) unless p0 = p1.
        If set to False, then the order of p0, p1 is not concerned and
        Bond(p0, p1), Bond(p1, p0) is equivalent.

    Examples
    --------
    >>> import numpy as np
    >>> from HamiltonianPy.bond import Bond
    >>> Bond(p0=np.array([0, 0]), p1=np.array([1, 1]))
    Bond(p0=array([0, 0]), p1=array([1, 1]), directional=True)

    >>> b = Bond(p0=np.array([0, 0]), p1=np.array([1, 1]))
    >>> b.getEndpoints()
    (array([0, 0]), array([1, 1]))

    >>> b.getLength()
    1.4142135623730951

    >>> b.getDisplace()
    array([1, 1])

    >>> b.getAzimuth()
    45.0

    >>> b.flip()
    Bond(p0=array([1, 1]), p1=array([0, 0]), directional=True)
    >>> b.oppositeTo(b.flip())
    True
    """

    def __init__(self, p0, p1, *, directional=True):
        """
        Customize the newly created Bond instance

        Parameters
        ----------
        p0, p1 : np.ndarray
            Endpoints of the bond
            p0 and p1 should be of the same shape and the supported shape is
            (1,), (2,), (3,)
        directional : boolean, optional, keyword-only
            Whether the direction of the bond should be considered
            If set to True, then the order of p0, p1 is concerned
            and Bond(p0, p1) != Bond(p1, p0) unless p0 = p1.
            If set to False, then the order of p0, p1 is not concerned
            and Bond(p0, p1), Bond(p1, p0) is equivalent.
        """

        assert (p0, np.ndarray) and p0.shape in ((1, ), (2, ), (3, ))
        assert (p1, np.ndarray) and p1.shape == p0.shape
        assert directional in (True, False)

        p0 = np.array(p0, copy=True)
        p1 = np.array(p1, copy=True)
        p0.setflags(write=False)
        p1.setflags(write=False)
        self._p0 = p0
        self._p1 = p1
        self._dim = p0.shape[0]
        self._directional = directional

        # Combine the original information of this bond into a tuple.
        # The tuple is then used to calculate the hash code and define the
        # comparison logic between different instances.
        tmp = [tuple(int(i) for i in _ZOOM * p) for p in [p0, p1]]
        if not directional:
            tmp.sort()
        tmp.append(directional)
        self._tuple_form = tuple(tmp)

    @property
    def directional(self):
        """
        Directional attribute of the bond
        """

        return self._directional

    def getEndpoints(self):
        """
        Access the p0 and p1 endpoints of the bond

        Returns
        -------
        res : tuple
            The p0 and p1 endpoints (p0, p1)
        """

        return np.array(self._p0, copy=True), np.array(self._p1, copy=True)

    def getLength(self, ndigits=None):
        """
        Return the length of the bond

        Parameters
        ----------
        ndigits : None or int, optional
            Number of digits to preserve after the decimal point
            default: None

        Returns
        -------
        res : The length of the bond
        """

        length = np.linalg.norm(self._p0 - self._p1)
        if ndigits is None:
            return length
        return np.around(length, decimals=ndigits)

    def getDisplace(self, ndigits=None):
        """
        Return the displace from p0 to p1

        Parameters
        ----------
        ndigits : None or int, optional
            Number of digits to preserve after the decimal point
            default: None

        Returns
        -------
        res : The displace from p0 to p1

        Raises
        ------
        NotImplementedError :
            For bond which is not directional, the method is meaningless
        """

        if self._directional:
            dr = self._p1 - self._p0
            if ndigits is None:
                return dr
            return np.around(dr, decimals=ndigits)
        else:
            raise NotImplementedError(
                "This method is meaningless for non-directional bond."
            )

    def getAzimuth(self, radian=False, ndigits=None):
        """
        Return the angle between the bond and the coordinate system

        Parameters
        ----------
        radian : boolean, optional
            Determine the unit of the angle, radian or degree
            default: False
        ndigits : None or int, optional
            Number of digits to preserve after the decimal point
            default: None

        Returns
        -------
        res : np.float64 or np.array
            For 1D or 2D system, the azimuth is the angle between the
            directional bond and the x-axis. The range of azimuth is [-pi, pi).
            For 3D, there are two angles,
            alpha: the angle between the directional bond and the z-axis. The
            range of alpha is [0, pi]
            beta: the angle between the projection of the bond on the xy plane
            and the x-axis. The range of beta is [-pi, pi)

        Raises
        ------
        NotImplementedError :
            For bond which is not directional, the method is meaningless
        """

        if self._directional:
            if radian:
                coeff = 1.0
            else:
                coeff = 180 / np.pi

            dr = self._p1 - self._p0
            if self._dim == 1:
                if dr[0] >= 0:
                    theta = 0.0
                else:
                    theta = -np.pi
            elif self._dim == 2:
                x, y = dr
                theta = np.arctan2(y, x)
                # If theta is pi, then it is equivalent to -pi.
                if abs(theta - np.pi) < _VIEW_AS_ZERO:
                    theta = -np.pi
            else:
                x, y, z = dr
                alpha = np.arctan2(np.sqrt(x**2+y**2), z)
                beta = np.arctan2(y, x)
                if abs(beta - np.pi) < _VIEW_AS_ZERO:
                    beta = -np.pi
                theta = np.array([alpha, beta])

            theta = coeff * theta
            if ndigits is None:
                return theta
            return np.around(theta, decimals=ndigits)
        else:
            raise NotImplementedError(
                "This method is meaningless for non-directional bond."
            )

    def __hash__(self):
        """
        Return the hash value of the bond
        """

        return hash(self._tuple_form)

    def __eq__(self, other):
        """
        Define the == operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form == other._tuple_form
        else:
            return False

    def __ne__(self, other):
        """
        Define the != operator between self and other
        """

        return not self.__eq__(other)

    def __repr__(self):
        """
        The official string representation of the bond
        """

        info = "Bond(p0={0!r}, p1={1!r}, directional={2!r})"
        return info.format(self._p0, self._p1, self._directional)

    def __str__(self):
        """
        Return a string that describes the content of the instance
        """

        fmt_func = repr
        #fmt_func = str
        NDIGITS = 4

        p0, p1 = self.getEndpoints()
        length = self.getLength(ndigits=NDIGITS)
        titles = ["P0", "P1", "Length"]
        contents = [fmt_func(p0), fmt_func(p1), fmt_func(length)]
        if self._directional:
            displace = self.getDisplace(ndigits=NDIGITS)
            azimuth = self.getAzimuth(ndigits=NDIGITS)
            titles.extend(["Azimuth", "Displace"])
            contents.extend([fmt_func(azimuth), fmt_func(displace)])
        width = max(len(item) for item in titles) + 1
        tmp = [t.ljust(width) + ": " + c for t, c in zip(titles, contents)]
        return "\n".join(tmp)

    def flip(self):
        """
        Return a bond that is opposite to itself

        Raises
        ------
        NotImplementedError :
            For bond which is not directional, the method is meaningless.
        """

        if self._directional:
            return self.__class__(p0=self._p1, p1=self._p0, directional=True)
        else:
            raise NotImplementedError(
                "This method is not implemented for non-directional bond."
            )

    def oppositeTo(self, other):
        """
        Return whether the self bond is opposite to the other bond

        Raises
        ------
        NotImplementedError :
            For bond which is not directional, the method is meaningless.
        """

        if self._directional:
            if isinstance(other, self.__class__):
                return self.flip().__eq__(other)
            else:
                raise TypeError(
                    "The right operand is not instance of this class."
                )
        else:
            raise NotImplementedError(
                "This method is not implemented for non-directional bond."
            )


if __name__ == "__main__":
    p0 = np.array([0, 0, 0])
    p1 = np.array([1, 1, 1])
    b0 = Bond(p0=p0, p1=p1, directional=True)
    b1 = Bond(p0=p0, p1=p1, directional=False)
    b2 = Bond(p0=p1, p1=p0, directional=True)
    b3 = Bond(p0=p1, p1=p0, directional=False)
    bs = [b0, b1, b2, b3]
    for b in bs:
        print(b)
        print("hash code: ", hash(b))
        print(repr(b))
        print("=" * 80)
    assert b0 != b1
    assert b0 != b2
    assert b0 != b3
    assert b1 != b2
    assert b1 == b3
    assert b2 != b3
    assert b0.oppositeTo(b2)
    assert b2.oppositeTo(b0)
