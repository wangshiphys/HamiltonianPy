"""
This module provides classes that describe lattice site as well as
single-particle state
"""


import numpy as np

from HamiltonianPy.indextable import IndexTable

# Useful global constants
_PRECISION = 4
_ZOOM = 10 ** _PRECISION
_NUMERIC_TYPES_REAL = (int, float, np.integer, np.floating)
################################################################################


def set_float_point_precision(precision):
    """
    Set the float-point precision for processing coordinate of lattice site

    In the internal implementation, the coordinate of lattice site is treated
    as float-point number no matter it is given as float-point number or
    integer. The float-point precision affects the internal implementation of
    the `SiteID` class as well as classes inherited from it. It also affects
    the behavior of classes which contain instance of `SiteID` or `StateID`
    as attribute. If you want to change the default value, you must call this
    function before using any other components defined in this subpackage.
    The default value is: `precision=4`.

    Parameters
    ----------
    precision : int
        The number of digits precision after the decimal point
    """

    assert isinstance(precision, int) and precision >= 0

    global  _PRECISION, _ZOOM
    _PRECISION = precision
    _ZOOM = 10 ** precision


class SiteID:
    """
    A unified description of lattice site

    Attributes
    ----------
    coordinate : tuple
        The coordinate of the lattice site in tuple form
    site : 1D np.ndarray
        The coordinate of the lattice site in np.ndarray form

    Examples
    --------
    >>> from HamiltonianPy.quantumoperator import SiteID
    >>> site0 = SiteID([0, 0])
    >>> site1 = SiteID((0.3, 0.75))
    >>> site0
    SiteID(site=(0, 0))
    >>> site1
    SiteID(site=(0.3, 0.75))
    >>> site0.coordinate
    (0, 0)
    >>> site1.site
    array([0.3 , 0.75])
    >>> site0 < site1
    True
    >>> site0.tolatex()
    '(0,0)'
    >>> site1.tolatex(site_index=1)
    '1'
    >>> site2 = SiteID((1/3, 2/3))
    >>> site2.tolatex()
    '(0.3333,0.6667)'
    >>> site2.tolatex(precision=6)
    '(0.333333,0.666667)'
    """

    def __init__(self, site):
        """
        Customize the newly created instance

        Parameters
        ----------
        site : list, tuple or 1D np.ndarray
            The coordinate of the lattice site.
        """

        site = tuple(site)
        assert len(site) in (1, 2, 3)
        assert all(isinstance(coord, _NUMERIC_TYPES_REAL) for coord in site)

        self._site = site
        # The tuple form of this instance
        # This internal attribute is useful for calculating the hash value of
        # the instance as well as defining compare logic between instances of
        # this class
        self._tuple_form = tuple(int(_ZOOM * coord) for coord in site)

    @property
    def coordinate(self):
        """
        The `coordinate` attribute
        """

        return self._site

    @property
    def site(self):
        """
        The `site` attribute
        """

        return np.array(self._site, copy=True)

    def __repr__(self):
        """
        Official string representation of the instance
        """

        return "SiteID(site={!r})".format(self._site)

    __str__ = __repr__

    def tolatex(self, *, site_index=None, precision=4, **kwargs):
        """
        Return the LaTex form of this instance

        Parameters
        ----------
        site_index : int or IndexTable, keyword-only, optional
            Determine how to format this instance
            If set to None, the instance is formatted as '(x)', '(x,y)'
            and '(x, y, z)' for 1, 2 and 3D respectively;
            If given as an integer, then `site_index` is the index of this
            instance and the LaTex form is the given integer;
            If given as an IndexTable, then `site_index` is a table that
            associate instances of SiteID with integer indices, the LaTex
            form is the index of this instance in the table.
            default: None
        precision : int
            The number of digits precision after the decimal point for
            processing float-point number.
        kwargs: other keyword arguments, optional
            Has no effect, do not use.

        Returns
        -------
        res : str
            The LaTex form of this instance
        """

        if isinstance(site_index, (int, np.integer)):
            latex_form = str(site_index)
        elif isinstance(site_index, IndexTable):
            latex_form = str(site_index(self))
        else:
            latex_form = "(" + ",".join(
                str(round(coord, ndigits=precision)) for coord in self._site
            ) + ")"
        return latex_form

    def __hash__(self):
        """
        Calculate the hash code of the instance
        """

        return hash(self._tuple_form)

    def __lt__(self, other):
        """
        Implement the `<` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form < other._tuple_form
        else:
            return NotImplemented

    def __eq__(self, other):
        """
        Implement the `==` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form == other._tuple_form
        else:
            return NotImplemented

    def __gt__(self, other):
        """
        Implement the `>` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form > other._tuple_form
        else:
            return NotImplemented

    def __le__(self, other):
        """
        Implement the `<=` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form <= other._tuple_form
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

    def __ge__(self, other):
        """
        Implement the `>=` operator between self and other
        """

        if isinstance(other, self.__class__):
            return self._tuple_form >= other._tuple_form
        else:
            return NotImplemented

    def getIndex(self, indices_table):
        """
        Return the index associated with the instance

        Parameters
        ----------
        indices_table : IndexTable
            A table that associate instances of SiteID with integer indices

        Returns
        -------
        index : int
            The index of this instance in the table
        """

        return indices_table(self)
