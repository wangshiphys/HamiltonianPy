"""
This module provide some self defined exception.
"""

class SelfDefinedError(Exception):# {{{
    """
    Self defined Exception.

    Attributes:
    ----------
    msg : Input message about the exception.
    """

    def __init__(self, msg):
        self.msg = msg
# }}}

class TargetError(SelfDefinedError):# {{{
    """
    Exceptions raised when the input item does not found in the container.
    """
    pass
# }}}

class ConvergenceError(SelfDefinedError):# {{{
    """
    Exceptions raised when the input item does not satisfied specific requirment.
    """
    pass
# }}}

class SwapFermionError(Exception):# {{{
    """
    Exceptions raised when swap creation and annihilation operator that with
    the same single particle state.
    """
    def __init__(self, aoc0, aoc1):
        self.aoc0 = aoc0
        self.aoc1 = aoc1

    def __str__(self):
        info = str(self.aoc0) + "\n" + str(self.aoc1) + "\n"
        info += "Swap these two operators would generate extra "
        info += "identity operator, which can not be processed properly."
        return info
# }}}

#This is a test!
if __name__ == "__main__":
    raise ConvergenceError('This is a test!')
