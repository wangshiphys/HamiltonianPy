"""
A function to convert a numpy array to a string in specific format.
"""

import numpy as np

__all__ = ["arrayformat"]

def arrayformat(array):
    """
    Convert a numpy array to string in specific format!

    Parameter:
    ----------
    array: ndarray
        The array to be converted!

    Return:
    -------
    res: str
        The resulting string!
    """

    if not isinstance(array, np.ndarray):
        raise TypeError("The input array in not ndarray!")

    form = {'float': lambda x: '%.4f' %x, 'int': lambda x: '%i' %x}
    string = np.array2string(array, suppress_small=True,
                             separator=', ', formatter=form)
    res = string.replace('[', '(').replace(']', ')')
    return res


#This is a test!
if __name__ == "__main__":
    print(arrayformat(np.random.random(4)))
    print(arrayformat(np.random.randint(0, 4, size=4)))
