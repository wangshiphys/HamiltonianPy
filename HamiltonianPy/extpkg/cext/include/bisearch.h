#define BISEARCHMOD_DOC "This module provide methods to search multiple items"\
                        " in an array using binary search algorithm.\n\n\n"\
                        "Methods defined here:\n"\
                        "bisearch(aims, array)\n"

#define BISEARCH_DOC "The C extension of binary search algorithm.\n\n\n"\
                     "Parameter:\n----------\n"\
                     "aims: tuple or list\n"\
                     "  A collection of targets to be searched!\n"\
                     "array: tuple or list\n"\
                     "  A tuple of number sorted in ascending order!\n\n"\
                     "Return:\n-------\n"\
                     "res: tuple\n"\
                     "  The positions of aims in array.\n"

long bisearch(const long aim, const long *list, const long n);
