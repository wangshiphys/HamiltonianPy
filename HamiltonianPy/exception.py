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

class SwapError(SelfDefinedError):# {{{
    """
    Exceptions raised when swap creation and annihilation operator that with
    the same single particle state.
    """
    pass
# }}}

#This is a test!
if __name__ == "__main__":
    raise ConvergenceError('This is a test!')
