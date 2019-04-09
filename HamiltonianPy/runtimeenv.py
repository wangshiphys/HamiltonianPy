"""
Temporary run time environment for python program
"""


from time import strftime
from traceback import print_tb
from pathlib import Path

import mkl
import sys


__all__ = [
    "RunTimeEnv",
]


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class RunTimeEnv:
    """
    This class provide a temporary run time environment for a python program

    Currently, this class only provide two functions:
    1. Redirect the printing information of the program to a `txt` file
    2. Set the number of threads used by mkl
    """

    def __init__(self, rd=False, path=".", threads_num=None):
        """
        Customize the newly created instance

        Parameters
        ----------
        rd: Boolean, optional
            Whether to redirect the output information to a `txt` file
            default: False
        path: str, optional
            The destination to save the `txt` file
            default: current working directory
        threads_num: int or None, optional
            The number of threads to used by mkl
            The default value `None` implies the maximum number of the system
            default: None
        """

        if threads_num is None:
            self._threads_num = mkl.get_max_threads()
        elif isinstance(threads_num, int) and threads_num > 0:
            self._threads_num = threads_num
        else:
            raise ValueError("Invalid `threads_num` parameter!")

        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._rd = rd
        self._path = path

    def __enter__(self):
        """
        Construct the run time environment
        """

        mkl.set_num_threads(self._threads_num)

        if self._rd:
            file_name = "Log {0}.txt".format(strftime("%Y-%m-%d %H-%M-%S"))
            fp = open(self._path / file_name, 'w', buffering=1)
            self._stdout = sys.stdout
            sys.stdout = self._fp = fp

            print(
                "Entering run time environment at {0}".format(
                    strftime(TIME_FORMAT)
                )
            )
            print("=" * 80, flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Close the opened file and restore sys.stdout before exit
        """

        mkl.set_num_threads(mkl.get_max_threads())

        if self._rd:
            print("=" * 80)
            if exc_type is None:
                print("Non exception has occurred!")
            else:
                print("Exc_type: {0}".format(exc_type))
                print("Exc_value: {0}".format(exc_value))
                print("Traceback:")
                print_tb(traceback, file=self._fp)
            print("=" * 80)
            print(
                "Exit run time environment at: {0}".format(
                    strftime(TIME_FORMAT)
                ), flush=True
            )
            sys.stdout = self._stdout
            self._fp.close()
        return False


if __name__ == "__main__":
    with RunTimeEnv(rd=True):
        print("This is a test of the RunTimeEnv class!")
        print("This is another line!")
