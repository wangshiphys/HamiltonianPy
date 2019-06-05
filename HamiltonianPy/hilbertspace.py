"""
Bases of Hilbert space in the occupation-number representation

The design concept behind this module is that the concerned Hilbert space
is consisted of one or more disjoint subspaces and every subspace can be
described by a collection of single-particle states as well as the number of
particle belong the subspace. The subspaces are identified by specifiers and
there are three kinds of valid subspace specifier:
1. N
    N is a positive integer
    The available single-particle states: 0, 1, 2, ..., N-1
    The particle number is not conserved
2. (N, M)
    N is a positive integer
    M is an integer and M <= N
    The available single-particle states: 0, 1, 2, ..., N-1
    The particle number: M(if M < 0, particle number is not conserved)
3. (states, M)
    states is a collection of non-negative integers
    M is an integer and M <= len(states)
    The available single-particle states: states
    The particle number: M(if M < 0, particle number is not conserved)

The first case is usually used for systems that the particle number is not
conserved.

The second case is used for systems that the particle number is conserved.
For example, a spin-1/2 half-filling N-sites lattice system: (2*N, N).

The third case is usually used for spin-conservation system or single
occupancy system.
For spin conservation system, the Hilbert space is divided
into spin-up subspace and spin-down subspace and the particle number in each
subspace is conserved:
((|0, up>, |1, up>, ..., |n, up>), M0)
((|0, down>, |1, down>, ..., |n, down>), M1)
For single occupancy system, every lattice site has exactly one particle:
((|0, up>, |0, down>), 1), ((|1, up>, |1, down>), 1), ...
"""


__all__ = [
    "SimpleHilbertSpace",
    "HilbertSpace",
    "base_vectors",
]


from itertools import chain, combinations, product

import numpy as np


def is_valid_states_collection(states):
    """
    Determine whether the given `states` parameter is a valid
    single-particle-state collection.

    1. `states` must be non-empty tuple, list or set;
    2. All elements of `states` must be non-negative integer.

    Returns
    -------
    res : bool
    """

    if (
        isinstance(states, (tuple, list, set)) and len(states) > 0 and
        all(isinstance(state, int) and state >= 0 for state in states)
    ):
        return True
    else:
        return False

def is_valid_subspace_specifier(specifier):
    """
    Determine whether the given `specifier` parameter is a valid
    subspace-specifier.

    See the docstring of the `hilbertspace` module for the definition of a
    valid subspace-specifier.

    Returns
    -------
    res: bool
    """

    if isinstance(specifier, int) and specifier > 0:
        return True
    elif isinstance(specifier, (list, tuple)) and len(specifier) == 2:
        states, p_num = specifier
        if isinstance(states, int) and states > 0 and states >= p_num:
            return True
        elif is_valid_states_collection(states) and len(states) >= p_num:
            return True
        else:
            return False
    else:
        return False

def subspace_specifier_preprocess(specifier):
    """
    Preprocess the given `specifier` to standard form.

    See the docstring of the `hilbertspace` module for the definition of the
    standard form of a subspace-specifier.

    Parameters
    ----------
    specifier :
        Subspace specifier.

    Returns
    -------
    specifier : length-2 tuple
        Subspace-specifier in standard form.

    Raises
    ------
    ValueError :
        Raised when the given `specifier` is not a valid subspace-specifier.
    """

    error_msg = "Invalid subspace-specifier: {0}"
    if isinstance(specifier, int) and specifier > 0:
        res = (tuple(range(specifier)), -1)
    elif isinstance(specifier, (tuple, list)) and len(specifier) == 2:
        states, p_num = specifier
        if isinstance(states, int) and states > 0 and states >= p_num:
            res = (tuple(range(states)), p_num)
        elif is_valid_states_collection(states) and len(states) >= p_num:
            res = (tuple(states), p_num)
        else:
            raise ValueError(error_msg.format(specifier))
    else:
        raise ValueError(error_msg.format(specifier))
    return res


class SimpleHilbertSpace:
    """
    A simple Hilbert space

    Simple Hilbert space is defined as a Hilbert space that can not be
    divided into two or more disjoint subspaces.

    Attributes
    ----------
    single_particle_states: tuple
        The available single-particle states.
    state_number : int
        The number of available single-particle state.
    particle_number : int
        The number of particle in the system.

    Examples
    --------
    >>> # 2 single-particle states and no constraint on the number of particle
    >>> SimpleHilbertSpace(states=2).base_vectors()
    array([0, 1, 2, 3], dtype=uint64)
    >>> SimpleHilbertSpace(states=[0, 1]).base_vectors()
    array([0, 1, 2, 3], dtype=uint64)

    >>> # 4 single-particle states and 2 particles
    >>> SimpleHilbertSpace(4, 2).base_vectors()
    array([ 3,  5,  6,  9, 10, 12], dtype=uint64)
    >>> SimpleHilbertSpace([0, 1, 2, 3], 2).base_vectors()
    array([ 3,  5,  6,  9, 10, 12], dtype=uint64)
    """

    def __init__(self, states, number=-1):
        """
        Customize the newly created instance

        Parameters
        ----------
        states : int | tuple | list | set
            If `states` is an integer, it must be positive and the available
            single-particle states are `tuple(range(states))`.
            If `states` is a tuple, list or set, the entries are the
            available single-particle states which must be non-negative integers
        number : int, optional
            The number of particle in the system
            Negative value means the particle number is not conserved
            default: -1
        """

        if isinstance(states, int) and states > 0:
            states = tuple(range(states))
        elif (
            isinstance(states, (tuple, list, set)) and len(states) > 0 and
            all(isinstance(state, int) and state >= 0 for state in states)
        ):
            states = tuple(states)
        else:
            raise ValueError(
                "The `states` parameter must be positive integer or "
                "collection of non-negative integers!"
            )

        state_number = len(states)
        if not isinstance(number, int) or number > state_number:
            raise ValueError(
                "The `number` parameter must be integer and no "
                "larger than the number of single-particle state!"
            )

        self._states = states
        self._state_number = state_number
        self._particle_number = number

    @property
    def single_particle_states(self):
        """
        The `single_particle_states` attribute
        """

        return self._states

    @property
    def state_number(self):
        """
        The `state_number` attribute
        """

        return self._state_number

    @property
    def particle_number(self):
        """
        The `particle_number` attribute
        """

        return self._particle_number

    def __repr__(self):
        """
        The official string representation of the instance
        """

        return "SimpleHilbertSpace({0!r}, {1!r})".format(
            self._states, self._particle_number
        )

    def __str__(self):
        """
        Return a string that describes the content of the instance
        """

        info = "All available single-particle states:\n    {0}\n" \
               "The number of particle: {1}"
        return info.format(self._states, self._particle_number)

    def base_vectors(self, *, container="array", sort=True):
        """
        Return integers that represent the base vectors of the Hilbert space

        Parameters
        ----------
        container : str, optional, keyword-only
            The container of the returned base vectors
            Accepted values are "list", "tuple" and "array", corresponding to
            list, tuple and 1D np.ndarray respectively.
            default: "array"
        sort : boolean, optional, keyword-only
            Sort the result in ascending order
            default: True

        Returns
         -------
         res : tuple, list or 1D np.ndarray
            A collection of base vectors
        """

        assert container in ("list", "tuple", "array")

        if self._particle_number < 0:
            basis_iterator = chain(
                *[
                    combinations(self._states, i)
                    for i in range(self._state_number + 1)
                ]
            )
        else:
            basis_iterator = combinations(self._states, self._particle_number)

        kets = [sum(1<<pos for pos in basis) for basis in basis_iterator]
        if sort:
            kets.sort()

        if container == "array":
            kets = np.array(kets, dtype=np.uint64)
        elif container == "tuple":
            kets = tuple(kets)
        return kets

    __call__ = base_vectors


class HilbertSpace:
    """
    Description of a general Hilbert Space

    Generally, a Hilbert space is composed of several subspaces. Each
    subspace is described by a collection of single-particle states as well
    as the number of particle in the subspace.

    Attributes
    ----------
    subspace_number : int
        The number of subspace
    subspace_specifiers : sequence of tuples
        A collection of subspace specifiers

    Examples
    --------
    >>> # 2 single-particle states and no constraint on the number of particle
    >>> HilbertSpace(2).base_vectors()
    array([0, 1, 2, 3], dtype=uint64)

    >>> # 4 single-particle states and 2 particles
    >>> HilbertSpace([(0, 1, 2, 3), 2]).base_vectors()
    array([ 3,  5,  6,  9, 10, 12], dtype=uint64)

    >>> # 2 sites and every site has spin-up and spin-down state
    >>> # one particle in spin-down state and one particle in spin-up state
    >>> HilbertSpace([(0, 2), 1], [(1, 3), 1]).base_vectors()
    array([ 3,  6,  9, 12], dtype=uint64)

    >>> # 2 sites and every site has spin-up and spin-down state
    >>> # one particle on site0 and one particle on site1
    >>> HilbertSpace([(0, 1), 1], [(2, 3), 1]).base_vectors()
    array([ 5,  6,  9, 10], dtype=uint64)
    """

    def __init__(self, *subspaces):
        """
        Customize the newly created instance

        Parameters
        ----------
        subspaces : Specifiers for different subspaces.
            See also the document for this module
        """

        subspace_specifiers = []
        for subspace in subspaces:
            if isinstance(subspace, int) and subspace > 0:
                specifier = (tuple(range(subspace)), -1)
            elif isinstance(subspace, (tuple, list)) and len(subspace) == 2:
                tmp, particle_number = subspace
                if isinstance(tmp, int) and tmp > 0 and particle_number <= tmp:
                    specifier = (tuple(range(tmp)), particle_number)
                elif (
                    isinstance(tmp, (list, tuple, set)) and len(tmp) > 0 and
                    all(isinstance(state, int) and state >= 0 for state in tmp)
                    and particle_number <= len(tmp)
                ):
                    specifier = (tuple(tmp), particle_number)
                else:
                    raise ValueError(
                        "Invalid subspace specifier: {}".format(subspace)
                    )
            else:
                raise ValueError(
                    "Invalid subspace specifier: {}".format(subspace)
                )
            subspace_specifiers.append(specifier)

        # All the subspaces should be disjoint to each other
        if not all(
            set(s0[0]).isdisjoint(set(s1[0]))
            for s0, s1 in combinations(subspace_specifiers, 2)
        ):
            raise ValueError("All subspaces should be disjoint to each other!")

        self._subspace_number = len(subspace_specifiers)
        self._subspace_specifiers = tuple(subspace_specifiers)

    @property
    def subspace_number(self):
        """
        The `subspace_number` attribute
        """

        return self._subspace_number

    @property
    def subspace_specifiers(self):
        """
        The `subspace_specifiers` attribute
        """

        return self._subspace_specifiers

    def __repr__(self):
        """
        The official string representation of this instance
        """

        parameters = ", ".join(
            "{0!r}".format(tmp) for tmp in self._subspace_specifiers
        )
        return "HilbertSpace({0})".format(parameters)

    def __str__(self):
        """
        Return a string that describes the content of this instance
        """

        info = "subspace_number: {0}\n".format(self._subspace_number)
        info += "subspace_specifiers:\n"
        for specifier in self._subspace_specifiers:
            info += "    {0}\n".format(specifier)
        return info

    def base_vectors(self, container="array", sort=True):
        """
        Return integers that represent the base vectors of the Hilbert space

        Parameters
        ----------
        container : str, optional, keyword-only
            The container of the returned base vectors
            Accepted values are "list", "tuple" and "array", corresponding to
            list, tuple and 1D np.ndarray respectively.
            default: "array"
        sort : boolean, keyword-only, optional
            Sort the result or not
            default: True

        Returns
         -------
         res : tuple, list or 1D np.ndarray
            A collection of base vectors
        """

        assert container in ("list", "tuple", "array")

        subspace_basis_iterators = []
        for states, particle_number in self._subspace_specifiers:
            if particle_number < 0:
                basis_iterator = chain(
                    *[combinations(states, i) for i in range(len(states) + 1)]
                )
            else:
                basis_iterator = combinations(states, particle_number)
            subspace_basis_iterators.append(basis_iterator)

        kets = [
            sum(1 << pos for pos in chain(*basis))
            for basis in product(*subspace_basis_iterators)
        ]

        if sort:
            kets.sort()

        if container == "array":
            kets = np.array(kets, dtype=np.uint64)
        elif container == "tuple":
            kets = tuple(kets)
        return kets

    __call__ = base_vectors


def base_vectors(*subspaces, container="array", sort=True):
    """
    Return integers that represent the base vectors of the Hilbert space

    Parameters
    ----------
    subspaces : Specifiers for different subspaces.
        See also the document for this module
    container : str, optional, keyword-only
        The container of the returned base vectors
        Accepted values are "list", "tuple" and "array", corresponding to
        list, tuple and 1D np.ndarray respectively.
        default: "array"
    sort : boolean, keyword-only, optional
        Sort the result or not
        default: True

    Returns
    -------
    res : tuple, list or 1D np.ndarray
        A collection of base vectors

    Examples
    --------
    >>> # 2 single-particle states and no constraint on the number of particle
    >>> base_vectors(2)
    array([0, 1, 2, 3], dtype=uint64)

    >>> # 4 single-particle states and 2 particles
    >>> base_vectors([(0, 1, 2, 3), 2])
    array([ 3,  5,  6,  9, 10, 12], dtype=uint64)

    >>> # 2 sites and every site has spin-up and spin-down state
    >>> # one particle in spin-down state and one particle in spin-up state
    >>> base_vectors([(0, 2), 1], [(1, 3), 1])
    array([ 3,  6,  9, 12], dtype=uint64)

    >>> # 2 sites and every site has spin-up and spin-down state
    >>> # one particle on site0 and one particle on site1
    >>> base_vectors([(0, 1), 1], [(2, 3), 1])
    array([ 5,  6,  9, 10], dtype=uint64)
    """

    assert container in ("list", "tuple", "array")

    subspace_specifiers = []
    for subspace in subspaces:
        if isinstance(subspace, int) and subspace > 0:
            specifier = (tuple(range(subspace)), -1)
        elif isinstance(subspace, (tuple, list)) and len(subspace) == 2:
            tmp, particle_number = subspace
            if isinstance(tmp, int) and tmp > 0 and particle_number <= tmp:
                specifier = (tuple(range(tmp)), particle_number)
            elif (
                isinstance(tmp, (tuple, list, set)) and len(tmp) > 0 and
                all(isinstance(state, int) and state >= 0 for state in tmp)
                and particle_number <= len(tmp)
            ):
                specifier = (tuple(tmp), particle_number)
            else:
                raise ValueError(
                    "Invalid subspace specifier: {0}".format(subspace)
                )
        else:
            raise ValueError("Invalid subspace specifier: {0}".format(subspace))
        subspace_specifiers.append(specifier)

    if not all(
        set(s0[0]).isdisjoint(set(s1[0]))
        for s0, s1 in combinations(subspace_specifiers, 2)
    ):
        raise ValueError("All subspaces should be disjoint to each other!")

    subspace_basis_iterators = []
    for states, particle_number in subspace_specifiers:
        if particle_number < 0:
            basis_iterator = chain(
                *[combinations(states, i) for i in range(len(states) + 1)]
            )
        else:
            basis_iterator = combinations(states, particle_number)
        subspace_basis_iterators.append(basis_iterator)

    kets = [
        sum(1 << pos for pos in chain(*basis))
        for basis in product(*subspace_basis_iterators)
    ]

    if sort:
        kets.sort()

    if container == "array":
        kets = np.array(kets, dtype=np.uint64)
    elif container == "tuple":
        kets = tuple(kets)
    return kets
