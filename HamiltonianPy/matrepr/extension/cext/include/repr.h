#define MATREPRCEXT_DOC "This module provide methods to calculate the matrix "\
                   "representation of a specific term in Hamiltonian.\n\n\n"\
                   "Methods defined here:\n"\
                   "hopping(cindex, aindex, base)\n"\
                   "hubbard(index0, index1, base)\n"\
                   "pairing(index0, index1, otype, base)\n"\
                   "aoc(index, otype, lbase[, rbase])\n"

#define HOPPING_DOC "The C extension of calculating a "\
                    "hopping term's matrix representation.\n\n\n"\
                    "Parameter:\n----------\n"\
                    "cindex: int\n"\
                    "   The index of the creation operator.\n"\
                    "aindex: int\n"\
                    "   The index of the annihilation operator.\n"\
                    "base: tuple or list\n"\
                    "   The base of the Hilbert space.\n\n"\
                    "Return:\n-------\n"\
                    "row: list\n"\
                    "   The row indices of these nonzero matrix elements.\n"\
                    "col: list\n"\
                    "   The corresponding column indices "\
                    "of these nonzero matrix elements.\n"\
                    "elmts: list\n"\
                    "   The corresponding nonzero matrix elements.\n"

#define HUBBARD_DOC "The C extension of calculating a "\
                    "hubbard term's matrix representation.\n\n\n"\
                    "Parameter:\n----------\n"\
                    "index0: int\n"\
                    "   The index of the number operator.\n"\
                    "index1: int\n"\
                    "   The index of the number operator.\n"\
                    "base: tuple or list\n"\
                    "   The base of the Hilbert space.\n\n"\
                    "Return:\n-------\n"\
                    "row: list\n"\
                    "   The row indices of these nonzero matrix elements.\n"\
                    "col: list\n"\
                    "   The corresponding column indices "\
                    "of these nonzero matrix elements.\n"\
                    "elmts: list\n"\
                    "   The corresponding nonzero matrix elements.\n"

#define PAIRING_DOC "The C extension of calculating a "\
                    "pairing term's matrix representation.\n\n\n"\
                    "Parameter:\n----------\n"\
                    "index0: int\n"\
                    "   The index of the first pairing operator.\n"\
                    "index1: int\n"\
                    "   The index of the second pairing operator.\n"\
                    "otype: int\n"\
                    "   The type of the pairing operator, "\
                    "0 represents annihilation and 1 creation operator!\n"\
                    "base: tuple or list\n"\
                    "   The base of the Hilbert space.\n\n"\
                    "Return:\n-------\n"\
                    "row: list\n"\
                    "   The row indices of these nonzero matrix elements.\n"\
                    "col: list\n"\
                    "   The corresponding column indices "\
                    "of these nonzero matrix elements.\n"\
                    "elmts: list\n"\
                    "   The corresponding nonzero matrix elements.\n"

#define AOC_DOC "The C extension of calculating a creation "\
                "or nnihilation operator's matrix represenation.\n\n\n"\
                "Parameter:\n----------\n"\
                "index: int\n"\
                "   The index of the operator.\n"\
                "otype: int\n"\
                "   The type of the operator, "\
                "0 represents annihilation and 1 creation operator!\n"\
                "lbase: tuple or list\n"\
                "   The base of the Hilbert space after the operation.\n"\
                "rbase: tuple or list, optional\n"\
                "   The base of the Hilbert space before the operation.\n"\
                "   If not given, the same as lbase!\n\n"\
                "Return:\n-------\n"\
                "row: list\n"\
                "   The row indices of these nonzero matrix elements.\n"\
                "col: list\n"\
                "   The corresponding column indices "\
                "of these nonzero matrix elements.\n"\
                "elmts: list\n"\
                "   The corresponding nonzero matrix elements.\n"


void hopping_C(const int cindex, const int aindex, const long *const base,
             const long dim, long *const row, 
             long *const col, int *const elmts);

void hubbard_C(const int index0, const int index1, const long *const base,
             const long dim, long *const row, 
             long *const col, int *const elmts);

void pairing_C(const int index0, const int index1, const int otype,
             const long *const base, const long dim,
             long *const row, long *const col, int *const elmts);

void aoc_C(const int index, const int otype, const long *const lbase,
         const long *const rbase, const long ldim, const long rdim,
         long *const row, long *const col, int *const elmts);
