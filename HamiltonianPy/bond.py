"""
This module provides Bond class that describes the bond connecting two points
"""


__all__ = [
    "Bond",
    "set_float_point_precision",
]


import numpy as np

# Useful global constant
_ZOOM = 10000
################################################################################


def set_float_point_precision(precision=4):
    """
    Set the float-point precision for processing coordinates of endpoints

    In the internal implementation, the coordinates of endpoints are treated
    as float-point numbers no matter they are given as float-point numbers
    or integers. The float-point precision affects the internal
    implementation of the `Bond` class. If you want to change the default
    value, you must call this function before creating any Bond instance.
    The default value is: `precision=4`.

    Parameters
    ----------
    precision : int, optional
        The number of digits precision after the decimal point.
        Default: 4.
    """

    assert isinstance(precision, int) and precision >= 0

    global  _ZOOM
    _ZOOM = 10 ** precision


class Bond:
    """
    Bond connecting two points

    Attributes
    ----------
    endpoints : tuple
        The two endpoints (p0 and p1) of the bond
        `p0` and `p1` is 1D np.ndarray with length 1, 2 or 3
    directional : boolean
        Whether the direction of the bond should be considered.
        If set to True, then the order of p0, p1 is concerned and
        Bond(p0, p1) != Bond(p1, p0) unless p0 == p1.
        If set to False, then the order of p0, p1 is not concerned and
        Bond(p0, p1), Bond(p1, p0) is equivalent.

    Examples
    --------
    >>> from HamiltonianPy.bond import Bond
    >>> b0 = Bond(p0=[0, 0], p1=(1, 1))
    >>> b1 = Bond(p0=(1.35, 2.47), p1=(2.47, 1.35))
    >>> b0
    Bond(p0=(0, 0), p1=(1, 1), directional=True)
    >>> b1
    Bond(p0=(1.35, 2.47), p1=(2.47, 1.35), directional=True)
    >>> b0.endpoints
    (array([0, 0]), array([1, 1]))
    >>> b1.endpoints
    (array([1.35, 2.47]), array([2.47, 1.35]))
    >>> b0.getLength(ndigits=4)
    1.4142
    >>> b0.getDisplace()
    array([1, 1])
    >>> b0.getAzimuth()
    45.0
    >>> b0.flip()
    Bond(p0=(1, 1), p1=(0, 0), directional=True)
    >>> b0.oppositeTo(b0.flip())
    True
    """

    def __init__(self, p0, p1, *, directional=True):
        """
        Customize the newly created Bond instance

        Parameters
        ----------
        p0, p1 : list, tuple or 1D.np.ndarray
            Endpoints of the bond
            `p0` and `p1` should be 1D array with length 1, 2 or 3.
        directional : boolean, keyword-only, optional
            Whether the direction of the bond should be considered
            If set to True, then the order of p0, p1 is concerned
            and Bond(p0, p1) != Bond(p1, p0) unless p0 == p1;
            If set to False, then the order of p0, p1 is not concerned
            and Bond(p0, p1), Bond(p1, p0) is equivalent.
        """

        p0 = np.array(p0, copy=True)
        p1 = np.array(p1, copy=True)
        assert p0.shape in ((1, ), (2, ), (3, )), "Invalid shape"
        assert p1.shape == p0.shape, "Shape does not match"

        self._p0 = p0
        self._p1 = p1
        self._dim = p0.shape[0]
        self._directional = directional

        # Combine the original information of this bond into a tuple.
        # The tuple is then used to calculate the hash code and define the
        # compare logic between different instances.
        identity = [tuple(int(_ZOOM * i) for i in p) for p in [p0, p1]]
        if not directional:
            identity.sort()
        identity.append(directional)
        self._tuple_form = tuple(identity)

    @property
    def endpoints(self):
        """
        The `endpoints` attribute
        """

        return np.array(self._p0, copy=True), np.array(self._p1, copy=True)

    @property
    def directional(self):
        """
        The `directional` attribute
        """

        return self._directional

    def getLength(self, ndigits=None):
        """
        Return the length of the bond

        Parameters
        ----------
        ndigits : None or int, optional
            Number of digits to preserve after the decimal point
            The default value `None` implies not round the result
            default: None

        Returns
        -------
        length : np.float64
            The length of the bond
        """

        length = np.linalg.norm(self._p0 - self._p1)
        if ndigits is None:
            return length
        return np.around(length, decimals=ndigits)

    def getDisplace(self, ndigits=None):
        """
        Return the displace from p0 to p1

        This method is only implemented for directional bond.

        Parameters
        ----------
        ndigits : None or int, optional
            Number of digits to preserve after the decimal point
            The default value `None` implies not round the result
            default: None

        Returns
        -------
        dr : 1D np.ndarray
            The displace from p0 to p1

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

        This method is only implemented for directional bond.

        Parameters
        ----------
        radian : boolean, optional
            Determine the unit of the angle, radian or degree
            default: False
        ndigits : None or int, optional
            Number of digits to preserve after the decimal point
            The default value `None` implies not round the result
            default: None

        Returns
        -------
        res : float or array of floats
            For 1D or 2D system, the azimuth is the angle between the
            directional bond and the x-axis. The range of azimuth is [-pi, pi).
            For 3D, there are two angles,
            alpha: the angle between the directional bond and the z-axis. The
            range of alpha is [0, pi].
            beta: the angle between the projection of the bond on the xy plane
            and the x-axis. The range of beta is [-pi, pi).

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
                theta = 0.0 if dr[0] >= 0 else -np.pi
            elif self._dim == 2:
                theta = np.arctan2(dr[1], dr[0])
            else:
                x, y, z = dr
                alpha = np.arctan2(np.sqrt(x**2+y**2), z)
                beta = np.arctan2(y, x)
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
        Calculate the hash value of the bond
        """

        return hash(self._tuple_form)

    def __eq__(self, other):
        """
        Implement the `==` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form == other._tuple_form
        else:
            return NotImplemented

    def __ne__(self, other):
        """
        Implement the `!=` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form != other._tuple_form
        else:
            return NotImplemented

    def __repr__(self):
        """
        Official string representation of the bond
        """

        info = "Bond(p0={0!r}, p1={1!r}, directional={2!r})"
        return info.format(
            tuple(self._p0), tuple(self._p1), self._directional
        )

    def __str__(self):
        """
        Return a string that describes the content of the instance
        """
        
        ndigits = 4
        titles = ["P0", "P1", "Length", "Azimuth", "Displace"]
        contents = [self._p0, self._p1, self.getLength(ndigits)]
        if self._directional:
            contents += [self.getAzimuth(ndigits), self.getDisplace(ndigits)]
        else:
            contents += ["Undefined", "Undefined"]
        return "\n".join(
            "{0:<8} : {1!r}".format(t, c) for t, c in zip(titles, contents)
        )

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
