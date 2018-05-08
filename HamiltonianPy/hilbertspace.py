"""
Hilbert space in the occupation number representation
"""


from itertools import chain, combinations, product


class SimpleSpace:
    """
    A simple Hilbert space

    If there is a set of single-particle states and the set can not be
    divided into two or more disjoint sub-sets, we called the corresponding
    Hilbert space a `simple Hilbert space`. For example, a system of N
    single-particle states and M particles.
    """


    def __init__(self, states, pnum=-1):
        """
        Customize the newly created SimpleSpace instance.

        Parameters
        ----------
        states : int or tuple of ints
            If `states` is an integer, it must be positive and the available
            single-particle states are `tuple(range(states))`.
            If `states` is a tuple, the entries represent the available
            single-particle states which must be non-negative.
        pnum : int
            The number of particles in the system.
            If pnum < 0, the particle number is not conserved.
        """

        errmsg = "The `states` parameter must be positive integer or tuple " \
                 "of non-negative integers."
        if isinstance(states, int):
            if states > 0:
                states = tuple(range(states))
            else:
                raise ValueError(errmsg)
        elif isinstance(states, tuple):
            for state in states:
                if not isinstance(state, int) or state < 0:
                    raise ValueError(errmsg)
        else:
            raise ValueError(errmsg)

        if not isinstance(pnum, int) or pnum > len(states):
            raise ValueError(
                "The `pnum` must be integer and no larger than the number of "
                "single-particle states"
            )

        self.states = states
        self.pnum = pnum

    def bases(self, to_int=True):
        states = self.states
        if self.pnum < 0:
            bases_iter = chain.from_iterable(
                [combinations(states, i) for i in range(len(states) + 1)]
            )
        else:
            bases_iter = combinations(states, self.pnum)

        if to_int:
            bases = []
            for cfg in bases_iter:
                ket = 0
                for state in cfg:
                    ket += 1 << state
                bases.append(ket)
            return tuple


class ComposedSpace:
    pass
