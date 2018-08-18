from datetime import datetime
from traceback import print_tb

import mkl
import sys


__all__ = [
    "RunTimeEnv",
]


class RunTimeEnv:
    """
    This class provide the run time environment for python program

    Currently, this class only provide two functions:
    1. Redirect the printing information of the program to a txt file
    2. Set the number of threads to use for parallel calculation

    Attribute:
    ----------

    """

    def __init__(self, rd=False, path="./", num_threads=None):
        """
        Customize the newly created instance

        Parameters
        ----------
        rd: Bool, optional
            Determine whether to redirect the output information to a text file
            default: False
        path: str, optional
            The destination to save the text file.
            default: current directory.
        num_threads: int or None, optional
            The number of threads to used for parallel calculation
            default: None, which means the maximum number of the system
        """

        if num_threads is None:
            self._num_threads = mkl.get_max_threads()
        elif isinstance(num_threads, int):
            self._num_threads = num_threads
        else:
            raise ValueError("Invalid `num_threads` parameter!")

        self._rd = rd
        self._path = path

    def __enter__(self):
        """
        Construct the run time environment
        """

        mkl.set_num_threads(self._num_threads)

        if self._rd:
            time_info = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())
            file_type = ".txt"
            file_name = self._path + "Loginfo " + time_info + file_type

            header_info = "Start running at: " + time_info + "\n"
            sep = len(header_info) * "=" + "\n"

            fp = open(file_name, 'w', buffering=1)
            fp.write(header_info + sep * 2)

            self._stdout = sys.stdout
            sys.stdout = self._fp = fp

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Close the opened file and restore sys.stdout before exit
        """

        mkl.set_num_threads(mkl.get_max_threads())

        if self._rd:
            self._fp.write("\n" + "=" * 80)
            if exc_type is None:
                self._fp.write("\nNo exception has occurred!\n")
            else:
                self._fp.write("\nExc_type: {0}\n".format(exc_type))
                self._fp.write("Exc_value: {0}\n".format(exc_value))
                self._fp.write("Traceback:\n")
                print_tb(traceback, file=self._fp)
            self._fp.write("=" * 80)
            info = "\n\nStop running at: "
            info += "{0:%Y-%m-%d %H:%M:%S}\n".format(datetime.now())
            self._fp.write(info)
            sys.stdout = self._stdout
            self._fp.close()
        return False