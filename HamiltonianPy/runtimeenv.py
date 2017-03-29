from datetime import datetime
from traceback import print_tb

import mkl
import sys

class RunTimeEnv:
    """
    This class provide the run time environment for python program.

    Currently, this class only provide two functions: 1) redirect the printing
    information of the program to a txt file; 2) set the number of threads
    should be used for parallel calculation.

    Attribute:
    ----------
    rd: Bool, optional
        Determine whether to redirect the screen output information to a txt
        file.
        default: True
    path: str, optional
        The destination to save the txt file.
        default: current directory.
    num_threads: int or None, optional
        The number of threads should be used for parallel calculation.
        default: None, which means the maximum number of the system.
    """

    def __init__(self, rd=False, path="./", num_threads=None):# {{{
        """
        Initilize the instance of this class.
        """

        if num_threads is None:
            self.num_threads = mkl.get_max_threads()
        elif isinstance(num_threads, int):
            self.num_threads = num_threads
        else:
            raise TypeError("The input num_threads parameter is not integer!")

        self.rd = rd
        self.path = path
    # }}}

    def __enter__(self):# {{{
        """
        Construct the run time environment.
        """

        mkl.set_num_threads(self.num_threads)

        if self.rd:
            timeinfo = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())
            filetype = ".txt"
            filename = self.path + "Loginfo " + timeinfo + filetype

            headerinfo = "Start running at: " + timeinfo + "\n"
            sep = len(headerinfo) * "=" + "\n"

            fp = open(filename, 'w', buffering=1)
            fp.write(headerinfo + sep * 2)

            self._stdout = sys.stdout
            sys.stdout = self.fp = fp

        return self
    # }}}

    def __exit__(self, exc_type, exc_value, traceback):# {{{
        """
        Close the opened file and restore sys.stdout before exit the run time 
        environment if redirected!
        """

        mkl.set_num_threads(mkl.get_max_threads())

        if self.rd:
            self.fp.write("\n" + "=" * 50)
            if exc_type is None:
                self.fp.write("\nNo exception has occurred!\n")
            else:
                self.fp.write("\nExc_type: {0}\n".format(exc_type))
                self.fp.write("Exc_value: {0}\n".format(exc_value))
                self.fp.write("Traceback:\n")
                print_tb(traceback, file=self.fp)
            self.fp.write("=" * 50)
            info = "\n\nStop running at: "
            info += "{0:%Y-%m-%d %H:%M:%S}\n".format(datetime.now())
            self.fp.write(info)
            sys.stdout = self._stdout
            self.fp.close()
        return False
    # }}}
