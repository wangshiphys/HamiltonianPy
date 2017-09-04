"""Generate bases of Hilbert space in occupation representation
"""

__all__ = ["basesfactory"]

from collections.abc import Iterable
from itertools import chain, combinations, product


def basesfactory(*dists, totuple=True, sort=True):
    """
    Return integers that represent the base vectors of the Hilbert space

    The positional parameters describe the distribution of particles over
    different sets of single-particle states.

    Generally, this function can be called in two ways:
        basesfactory(N, [totuple[, sort]])
            with a positive integer N and optional totuple and sort argument

        basesfactory((states_set, occupy_num), ..., [totuple[, sort]])
            with at least one 2-tuple and optional totuple and sort argument
            'states_set' is a collection of the single particle states that are
            allowed to be occupied.
            'occupy_num' is the number of particles that might occupy
            these 'states_set'.

        See the examples for detail explanation.

    Notes:
        1) If called in the first form, we assume that only these states:
        [0, 1, ..., N-1] are allowed to be occupied and all other states are
        prohibited.
        2) If called in the second form, we assume that only these states that
        appear in these 'states_set' are allowed to be occupied and all
        others are prohibited.
        3) If called in the second form, for every given 'states_set' there
        should't be duplicate states. Also, there should't be common states
        between these 'states_set'.
        4) If called in the second form, negative 'occupy_num' means there are
        no constraint on the number of particles in the corresponding
        subspace: 'states_set'

    Parameters
    ----------
    dist : int or tuple
        Distribution of particles over different sets of single-particle states.
    totuple : boolean, keyword-only, optional
        Convert the result to a tuple or not.
        default: True
    sort : boolean, keyword-only, optional
        Sort the result or not
        default: True

    Returns
    -------
    res : tuple or list
        A collection of base vectors

    Examples:
    ---------
    >>> # 4 single-particle states and no constraint on the number of particles
    >>> basesfactory(4)
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

    >>> # 4 single-particle states and 2 particles
    >>> states_set = tuple(range(4))
    >>> basesfactory((states_set, 2))
    (3, 5, 6, 9, 10, 12)

    >>> # 2 sites and every site has spin-up and spin-down state
    >>> # one particle in spin-down state and one particle in spin-up state
    >>> spin_down_states = (0, 2)
    >>> spin_up_states = (1, 3)
    >>> basesfactory((spin_down_states, 1), (spin_up_states, 1))
    (3, 6, 9, 12)

    >>> # 2 sites and every site has spin-up and spin-down state
    >>> # one particle on site0 and one particle on site1
    >>> site0_states = (0, 1)
    >>> site1_states = (2, 3)
    >>> basesfactory((site0_states, 1), (site1__states, 1))
    (5, 6, 9, 10)
    """

    if (len(dists) == 1 and isinstance(dists[0], int)):
        states_num = dists[0]
        if states_num > 0:
            bases = list(range(1 << states_num))
        else:
            errmsg = "The given number of single particle states is: {0}\n"
            errmsg = errmsg.format(states_num)
            errmsg += "There should be at least one single particle state."
            raise ValueError(errmsg)
    else:
        subcfgs_collection = []
        for states_set, occupy_num in dists:
            states_num = len(states_set)
            if (isinstance(states_set, Iterable)
                    and isinstance(occupy_num, int)):
                if occupy_num < 0:
                    tmp = [combinations(states_set, i)
                            for i in range(states_num + 1)]
                    subcfgs = chain(*tmp)
                elif occupy_num <= states_num:
                    subcfgs = combinations(states_set, occupy_num)
                else:
                    errmsg = "The states_set being processed: {}"
                    errmsg = errmsg.format(states_set)
                    errmsg += "The give occupy_num: {}".format(occupy_num)
                    errmsg += "The number of occupied state should not be "
                    errmsg += "larger than the number of states."
                    raise ValueError(errmsg)
                subcfgs_collection.append(subcfgs)
            else:
                raise TypeError("The states_set should be iterable and "
                        "occupy_num should be integer!")

        bases = []
        for cfg in product(*subcfgs_collection):
            ket = 0
            for pos in chain(*cfg):
                ket += 1 << pos
            bases.append(ket)

        if sort:
            bases.sort()

    if totuple:
        bases = tuple(bases)
    return bases


if __name__ == "__main__":
    states_num = 4
    bases = basesfactory(states_num)
    assert bases == tuple(range(1<<states_num))
    print("N = {0} single-particles with no constraint.".format(states_num))
    print("The bases: ", bases)
    print()

    state_num = 4
    occupy_num = 2
    bases = basesfactory([range(state_num), occupy_num])
    assert bases == (3, 5, 6, 9, 10, 12)
    print("N = {0} single-particles with M = {1} occupied.".format(states_num, occupy_num))
    print("The bases: ", bases)
    print()

    spin_up = (0, 2)
    spin_down = (1, 3)
    n_up = 1
    n_down = 1
    bases = basesfactory([spin_up, n_up], [spin_down, n_down])
    assert bases == (3, 6, 9, 12)
    print("Spin-up states: ", spin_up, "with {0} state be occupied.".format(n_up))
    print("Spin-down states: ", spin_down, "with {0} state be occupied.".format(n_down))
    print("The bases: ", bases)
    print()

    site0 = (0, 1)
    site1 = (2, 3)
    n0 = 1
    n1 = 1
    bases = basesfactory([site0, n0], [site1, n1])
    assert bases == (5, 6, 9, 10)
    print("Site0 states: ", site0, "with {0} state be occupied.".format(n0))
    print("Site1 states: ", site1, "with {0} state be occupied.".format(n1))
    print("The bases: ", bases)
    print()

    site0 = (0, 1)
    site1 = (2, 3)
    n0 = 1
    n1 = -1
    bases = basesfactory([site0, n0], [site1, n1])
    assert bases == (1, 2, 5, 6, 9, 10, 13, 14)
    print("Site0 states: ", site0, "with {0} state be occupied.".format(n0))
    print("Site1 states: ", site1, "with {0} state be occupied.".format(n1))
    print("The bases: ", bases)
    print()

    site0 = (0, 1)
    site1 = (2, 3)
    n0 = 1
    n1 = 0
    bases = basesfactory([site0, n0], [site1, n1])
    assert bases == (1, 2)
    print("Site0 states: ", site0, "with {0} state be occupied.".format(n0))
    print("Site1 states: ", site1, "with {0} state be occupied.".format(n1))
    print("The bases: ", bases)
