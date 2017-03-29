#define BITSELECTMOD_DOC "This module provide methods to select specific "\
                         "items from a sequence.\n\n\n"\
                         "Methods defined here:\n"\
                         "bitselect(array, poses, bits)\n"

#define BITSELECT_DOC "Found the positions of items in the given array "\
                      "according to the given poses and bits parameter.\n\n\n"\
                     "Parameter:\n----------\n"\
                     "array: tuple or list\n"\
                     "  A collection of integers!\n"\
                     "poses: tuple or list\n"\
                     "  Specify the positions which should be judged in the "\
                     "  binary representation of integer.\n"\
                     "bits: tuple or list\n"\
                     "  Specify the digit in the positions given in poses "\
                     "  parameter. Entries of this sequence should be only "\
                     "  1 or 0, and the length of this sequence should be "\
                     "  the same as poses parameter.\n\n"\
                     "Return:\n-------\n"\
                     "res: tuple\n"\
                     "  The positions of items in array which satisfy the "\
                     "  condition specified by the poses and bits parameter.\n"
