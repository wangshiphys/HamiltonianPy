"""
The function to generate bases of Hilbert space in occupation representation.
"""

from itertools import chain, combinations, product

__all__ = ["basesfactory"]

def basesfactory(*dists):
    """
    Return integers that represent the base vectors of the Hilbert space.

    The positional parameters describe the distribution of particles over
    different sets of single-particle states.
    
    Warning:
        This function assumes that the indices of single-particle states begins
        from zero and are continuous integer. Also this function only takes 
        these states that appear in the positional parameters into 
        consideration, for these state do not appear they are assumed to be 
        unoccupied. For example, if there are 8 states: [0, 1, 2, 3, 4, 5, 6, 7], 
        and we call basesfactory([(0, 1), 1], [(4, 5), 1]),then the states 
        [2, 3, 6, 7] are thought to be unoccupied.

    Parameter:
    ----------
    dist: int or tuple or list
        Descriptor of the distribution of particles over different sets of
        single-particle states.
        Generally, there are two ways to call this function:
        1). basesfactory(integer). Called with only one positive integer parameter.
        2). basesfactory(tuple or list[, ...]). Called with one or more tuples or
        lists. All the tuples or lists should be of length 2 and of form
        [collection, int]. The "collection" should be of type list, tuple, set 
        or range and the "int" should be integer. The "int" should not be larger
        than the length of "collection" and if it is less than zero, the number 
        of occupied state of this "collection" is unconstrained.

        For example:
        1). N single-particle states and no constrains on the number of
        particles: basesfactory(N)
        2). seq is the collection of all states and m is the number of occupied 
        state: basesfactory([seq, m])
        3). seq_up and seq_down is the collection of all spin-up and spin-down 
        states respectively. m_up and m_down is the number of occupied spin-up 
        and spin-down state respectively: 
            basesfactory([seq_up, m_up], [seq_down, m_down])
        4). seq_i is the collection of all states(generally spin-up and 
        spin-down) on site i and m_i is the number of particle on site i: 
            basesfactory([seq_0, m_0], [seq_1, m_1], ..., [seq_n, m_n])

    Return:
    -------
    res: tuple
        The collection of base vectors stored in ascending order.
    """

    if (len(dists) == 1 and isinstance(dists[0], int)):
        states_num = dists[0]
        if states_num > 0:
            return tuple(range(1 << states_num))
        else:
            raise ValueError("There should be at least one state.")
    else:
        subcfgs_collection = []
        for dist in dists:
            if (isinstance(dist, (list, tuple)) and len(dist) == 2):
                states_set, occupy_num = dist
                states_num = len(states_set)
                if (isinstance(states_set, (list, tuple, range, set)) and 
                    isinstance(occupy_num, int)):
                    if occupy_num < 0:
                        subcfgs = []
                        for i in range(states_num + 1):
                            subcfgs.append(combinations(states_set, i))
                        subcfgs_collection.append(chain(*subcfgs))
                    elif occupy_num <= states_num:
                        subcfgs = combinations(states_set, occupy_num)
                        subcfgs_collection.append(subcfgs)
                    else:
                        raise ValueError("The number of occupied state should "
                                "not be more than the number of states.")
                else:
                    raise TypeError("The collection of states should be list, "
                                    "tuple, range or set and the occupied "
                                    "state number should be integer.")
            else:
                raise TypeError("The positional parameters should be "
                                "lists or tuples with two entries.")

        bases = []
        for cfg in product(*subcfgs_collection):
            ket = 0
            for pos in chain(*cfg):
                ket += 1 << pos
            bases.append(ket)
        bases.sort()
        return tuple(bases)


if __name__ == "__main__":
    states_num = 4
    bases = basesfactory(states_num)
    print("N = {0} single-particles with no constrain.".format(states_num))
    print("The bases: ", bases)
    print()
    state_num = 4
    occupy_num = 2
    bases = basesfactory([range(state_num), occupy_num])
    print("N = {0} single-particles with M = {1} occupied.".format(states_num, occupy_num))
    print("The bases: ", bases)
    print()
    spin_up = (0, 2)
    spin_down = (1, 3)
    n_up = 1
    n_down = 1
    bases = basesfactory([spin_up, n_up], [spin_down, n_down])
    print("Spin-up states: ", spin_up, "with {0} state be occupied.".format(n_up))
    print("Spin-down states: ", spin_down, "with {0} state be occupied.".format(n_down))
    print("The bases: ", bases)
    print()
    site0 = (0, 1)
    site1 = (2, 3)
    n0 = 1
    n1 = 1
    bases = basesfactory([site0, n0], [site1, n1])
    print("Site0 states: ", site0, "with {0} state be occupied.".format(n0))
    print("Site1 states: ", site1, "with {0} state be occupied.".format(n1))
    print("The bases: ", bases)
    print()
    site0 = (0, 1)
    site1 = (2, 3)
    n0 = 1
    n1 = -1
    bases = basesfactory([site0, n0], [site1, n1])
    print("Site0 states: ", site0, "with {0} state be occupied.".format(n0))
    print("Site1 states: ", site1, "with {0} state be occupied.".format(n1))
    print("The bases: ", bases)
    print()
    site0 = (0, 1)
    site1 = (2, 3)
    n0 = 1
    n1 = 0
    bases = basesfactory([site0, n0], [site1, n1])
    print("Site0 states: ", site0, "with {0} state be occupied.".format(n0))
    print("Site1 states: ", site1, "with {0} state be occupied.".format(n1))
    print("The bases: ", bases)
