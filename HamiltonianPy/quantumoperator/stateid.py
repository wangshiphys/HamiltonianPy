"""
This module provides classes that describe lattice site and single-particle
state
"""


__all__ = [
    "SiteID",
    "StateID",
]


import numpy as np

from HamiltonianPy.indextable import IndexTable
from HamiltonianPy.quantumoperator.constant import NUMERIC_TYPES_REAL
from HamiltonianPy.quantumoperator.constant import SPIN_DOWN, SPIN_UP

# Useful global constants
_ZOOM = 10000
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

    global  _ZOOM
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
    >>> site2.tolatex(ndigits=6)
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
        assert all(isinstance(coord, NUMERIC_TYPES_REAL) for coord in site)

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

    def tolatex(self, *, site_index=None, ndigits=4, **kwargs):
        """
        Return the LaTex form of this instance

        Parameters
        ----------
        site_index : int, IndexTable or None, keyword-only, optional
            Determine how to format this instance
            If set to None, the coordinate is rounded to `ndigits` precision
            after the decimal point and formatted as '(x)', '(x,y)' and
            '(x,y,z)' for 1, 2 and 3D respectively;
            If given as an integer, then `site_index` is the index of this
            instance and the LaTex form is the given integer;
            If given as an IndexTable, then `site_index` is a table that
            associate instances of SiteID with integer indices, the LaTex
            form is the index of this instance in the table.
            default: None
        ndigits : int, keyword-only, optional
            The number of digits precision after the decimal point.
            This parameter only takes effect when `site_index` is None.
            default: 4
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
                str(round(coord, ndigits=ndigits)) for coord in self._site
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
            The index of this instance in the given table
        """

        return indices_table(self)


class StateID(SiteID):
    """
    A unified description of single-particle state

    Attributes
    ----------
    coordinate : tuple
        The coordinate of the localized single-particle state in tuple form
    site : 1D np.ndarray
        The coordinate of the localized single-particle state in np.ndarray form
    spin : int
        The spin index of the single-particle state
    orbit : int
        The orbit index of the single-particle state

    Examples
    --------
    >>> from HamiltonianPy.quantumoperator import StateID
    >>> state0 = StateID(site=[0, 0], spin=1)
    >>> state1 = StateID(site=(0.3, 0.75), spin=0)
    >>> state0
    StateID(site=(0, 0), spin=1, orbit=0)
    >>> state1
    StateID(site=(0.3, 0.75), spin=0, orbit=0)
    >>> state1.coordinate
    (0.3, 0.75)
    >>> state1.site
    array([0.3 , 0.75])
    >>> state0 < state1
    True
    >>> state0.tolatex()
    '(0,0),$\\\\uparrow$'
    >>> state1.tolatex(site_index=1)
    '1,$\\\\downarrow$'
    >>> state2 = StateID((1/3, 2/3))
    >>> state2.tolatex()
    '(0.3333,0.6667),$\\\\downarrow$'
    >>> state2.tolatex(precision=6)
    '(0.333333,0.666667),$\\\\downarrow$'
    """

    def __init__(self, site, spin=0, orbit=0):
        """
        Customize the newly created instance

        Parameters
        ----------
        site : list, tuple or 1D np.ndarray
            The coordinate of the localized single-particle state.
        spin : int, optional
            The spin index of the single-particle state
            default: 0
        orbit : int, optional
            The orbit index of the single-particle state
            default: 0
        """

        assert isinstance(spin, int) and spin >= 0
        assert isinstance(orbit, int) and orbit >= 0

        super().__init__(site=site)
        self._spin = spin
        self._orbit = orbit

        # The self._tuple_form on the right hand has already been set properly
        # by calling `super().__init__(site=site)`
        self._tuple_form = (self._tuple_form, spin, orbit)

    @property
    def spin(self):
        """
        The `spin` attribute
        """

        return self._spin

    @property
    def orbit(self):
        """
        The `orbit` attribute
        """

        return self._orbit

    def __repr__(self):
        """
        Official string representation of the instance
        """

        info = "StateID(site={0!r}, spin={1}, orbit={2})"
        return info.format(self._site, self._spin, self._orbit)

    __str__ = __repr__

    def tolatex(
            self, *, site_index=None, precision=4,
            spin_one_half=True, suppress_orbit=True, **kwargs
    ):
        """
        Return the LaTex form of this instance

        Parameters
        ----------
        site_index : int or IndexTable, keyword-only, optional
            Determine how to format the `site` attribute
            If set to None, the `site` attribute is formatted as '(x)', '(x,y)'
            and '(x, y, z)' for 1, 2 and 3D respectively;
            If given as an integer, then `site_index` is the index of the
            lattice-site and the LaTex form of the `site` attribute is the
            given integer;
            If given as an IndexTable, then `site_index` is a table that
            associate instances of SiteID with integer indices, the LaTex
            form of the `site` attribute is the index of the lattice-site in the
            table.
            default: None
        precision : int, keyword-only, optional
            The number of digits precision after the decimal point for
            processing float-point number.
            default: 4
        spin_one_half : boolean, keyword-only, optional
            Whether the concerned system is a spin-1/2 system.
            If set to True, the spin index is represented by down- or up-arrow;
            If set to False, the spin index is represented by an integer.
            default: True
        suppress_orbit : boolean, keyword-only, optional
            Whether to suppress the orbit degree of freedom.
            If set to True, the orbit index is not shown.
            default: True
        kwargs: other keyword arguments, optional
            Has no effect, do not use.

        Returns
        -------
        res : str
            The LaTex form of this instance
        """

        if isinstance(site_index, (int, np.integer)):
            latex_form_site = str(site_index)
        elif isinstance(site_index, IndexTable):
            latex_form_site = str(site_index(self))
        else:
            latex_form_site = "(" + ",".join(
                str(round(coord, ndigits=precision)) for coord in self._site
            ) + ")"

        if spin_one_half and self._spin in (SPIN_DOWN, SPIN_UP):
            if self._spin == SPIN_DOWN:
                latex_form_spin = r"$\downarrow$"
            else:
                latex_form_spin = r"$\uparrow$"
        else:
            latex_form_spin = str(self._spin)

        if suppress_orbit:
            latex_form = ",".join([latex_form_site, latex_form_spin])
        else:
            latex_form = ",".join(
                [latex_form_site, latex_form_spin, str(self._orbit)]
            )
        return latex_form
