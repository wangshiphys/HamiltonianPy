#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#define CREATION 1
#define FALSE 0
#define TRUE 1
#define PREFIX "In function matrixRepr_API, "
#define MODULE_DOC "This module provide method to calculate the matrix "\
                   "representation of a specific Hamiltonian term.\n\n\n"\
                   "Methods defined here:\n---------------------\n"\
                   "matrixRepr(term, rbase[, lbase])\n"
#define METHOD_DOC "The C extension of calculating a term's matrix representation.\n\n\n"\
                   "Parameter:\n----------\n"\
                   "term: list or tuple\n"\
                   "    It is a tuple or list and it's entries must be "\
                   "tuples or lists with two entries.\n"\
                   "    The first entry is the index of the state and the "\
                   "second is the operator type(1 for creation and 0 for "\
                   "annihilation).\n"\
                   "rbase: list or tuple\n"\
                   "    The base of the Hilbert space before the operation.\n"\
                   "lbase: list or tuple, optional\n"\
                   "    The base of the Hilbert space after the operation.\n"\
                   "    default: lbase = rbase\n\n"\
                   "Return:\n-------\n"\
                   "rows: list\n"\
                   "    The row indices of these nonzero matrix entries.\n"\
                   "cols: list\n"\
                   "    The corresponding column indices.\n"\
                   "entries: list\n"\
                   "    The corresponding nonzero matrix entries.\n"

static long bisearch(const long aim, const long *list, const long n)
/*{{{*/
{
    long low, high, mid, buff;
    low = 0;
    high = n - 1;
    while (low <= high) {
        mid = (low + high) / 2;
        buff = *(list + mid);
        if (aim == buff) {
            return mid;
        } else if (aim > buff) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    fprintf(stderr, "Error in bisearch function, "
            "the aim does not contained in the array!\n");
    exit(EXIT_FAILURE);
}
/*}}}*/

static PyObject *matrixRepr_API(PyObject *self, PyObject *args)
/*{{{*/
{
    char *exc_msg;
    int swap, not_zero_flag;
    long i, j, temp;
    long state_index, mask, criterion, ket;
    long aoc_num, rdim, ldim;
    long *aocs = NULL, *rbase_c = NULL, *lbase_c = NULL;
    PyObject *term_py = NULL, *rbase_py = NULL, *lbase_py = NULL;
    PyObject *aoc = NULL, *row_obj = NULL, *col_obj = NULL, *entry_obj = NULL;
    PyObject *rows_py = NULL, *cols_py = NULL, *entries_py = NULL, *res = NULL;


    if (!PyArg_ParseTuple(args, "OO|O", &term_py, &rbase_py, &lbase_py)) {
        return NULL;
    }
    
    //Convert the python representation of term to C array.
    if (PyTuple_CheckExact(term_py) || PyList_CheckExact(term_py)) {
        aoc_num = PySequence_Fast_GET_SIZE(term_py);
        aocs = (long *)PyMem_Calloc(3 * aoc_num, sizeof(long));
        if (aocs != NULL) {
            for (i=0; i<aoc_num; ++i) {
                aoc = PySequence_Fast_GET_ITEM(term_py, i);
                if (PyTuple_CheckExact(aoc) || PyList_CheckExact(aoc)) {
                    state_index = PyLong_AsLong(PySequence_Fast_GET_ITEM(aoc, 0));
                    temp = PyLong_AsLong(PySequence_Fast_GET_ITEM(aoc, 1));
                    mask = 1 << state_index;
                    *(aocs + (3 * i + 0)) = state_index;
                    *(aocs + (3 * i + 1)) = mask;
                    *(aocs + (3 * i + 2)) = (temp == CREATION)? 0: mask;
                } else {
                    exc_msg = PREFIX"the aoc object is not a tuple or list.\n";
                    PyErr_SetString(PyExc_TypeError, exc_msg);
                    return NULL;
                }
            }
        } else {
            exc_msg = PREFIX"unable to allocate the required memory for aocs.\n";
            PyErr_SetString(PyExc_MemoryError, exc_msg);
            return NULL;
        }
    } else {
        exc_msg = PREFIX"the term parameter is not a tuple or list.\n";
        PyErr_SetString(PyExc_TypeError, exc_msg);
        return NULL;
    }

    //Convert the python sequence rbase_py to C array.
    if (PyTuple_CheckExact(rbase_py) || PyList_CheckExact(rbase_py)) {
        rdim = PySequence_Fast_GET_SIZE(rbase_py);
        rbase_c = (long *)PyMem_Calloc(rdim, sizeof(long));
        if (rbase_c != NULL) {
            for(i=0; i<rdim; ++i) {
                *(rbase_c + i) = PyLong_AsLong(PySequence_Fast_GET_ITEM(rbase_py, i));
            }
        } else {
            exc_msg = PREFIX"unable to allocate the required memory for rbase_c.\n";
            PyErr_SetString(PyExc_MemoryError, exc_msg);
            return NULL;
        }
    } else {
        exc_msg = PREFIX"the rbase parameter is not a tuple or list.\n";
        PyErr_SetString(PyExc_TypeError, exc_msg);
        return NULL;
    }

    //Convert the python sequence lbase_py to C array if necessary.
    if (lbase_py == NULL || lbase_py == Py_None) {
        ldim = rdim;
        lbase_c = rbase_c;
    } else { 
        if (PyTuple_CheckExact(lbase_py) || PyList_CheckExact(lbase_py)) {
            ldim = PySequence_Fast_GET_SIZE(lbase_py);
            lbase_c = (long *)PyMem_Calloc(ldim, sizeof(long));
            if (lbase_c != NULL) {
                for(i=0; i<ldim; ++i) {
                    *(lbase_c + i) = PyLong_AsLong(PySequence_Fast_GET_ITEM(lbase_py, i));
                }
            } else {
                exc_msg = PREFIX"unable to allocate the required memory for lbase_c.\n";
                PyErr_SetString(PyExc_MemoryError, exc_msg);
                return NULL;
            }
        } else {
            exc_msg = PREFIX"the lbase parameter is not a tuple or list.\n";
            PyErr_SetString(PyExc_TypeError, exc_msg);
            return NULL;
        }
    }

    //Calculate the matrix representation of this term.
    rows_py = PyList_New(0);
    cols_py = PyList_New(0);
    entries_py = PyList_New(0);
    if (rows_py != NULL && cols_py != NULL && entries_py != NULL) {
        for (i=0; i<rdim; ++i) {
            swap = 0;
            not_zero_flag = TRUE;
            ket = *(rbase_c + i);
            for (j=aoc_num-1; j>=0; --j) {
                state_index = *(aocs + 3*j);
                mask = *(aocs + (3*j+1));
                criterion = *(aocs + (3*j+2));
                if ((ket & mask) == criterion) {
                    for (temp=0; temp<state_index; ++temp) {
                        if (ket & (1 << temp)) swap ^= 1;
                    }
                    ket ^= mask;
                } else {
                    not_zero_flag = FALSE;
                    break;
                }
            }
            if (not_zero_flag) {
                row_obj = PyLong_FromLong(bisearch(ket, lbase_c, ldim));
                PyList_Append(rows_py, row_obj);
                Py_DECREF(row_obj);

                col_obj = PyLong_FromLong(i);
                PyList_Append(cols_py, col_obj);
                Py_DECREF(col_obj);

                entry_obj = PyFloat_FromDouble(swap? -1: 1);
                PyList_Append(entries_py, entry_obj);
                Py_DECREF(entry_obj);
            }
        }
    } else {
        exc_msg = PREFIX"unable to create empty lists to hold rows, cols and entries.\n";
        PyErr_SetString(PyExc_RuntimeError, exc_msg);
        return NULL;
    }

    //Release the allocate memory
    PyMem_Free(aocs);
    if (lbase_c != rbase_c) {
        PyMem_Free(rbase_c);
        PyMem_Free(lbase_c);
    } else {
        PyMem_Free(rbase_c);
    }

    //Construct the return python object
    res = Py_BuildValue("(O(OO))", entries_py, rows_py, cols_py);
    //Decrease the reference count
    Py_DECREF(rows_py);
    Py_DECREF(cols_py);
    Py_DECREF(entries_py);

    return res;
}
/*}}}*/

static PyMethodDef matrixrepr_methods[] = {
    {"matrixRepr", (PyCFunction)matrixRepr_API, METH_VARARGS, METHOD_DOC},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef matrixrepr_module = {
    PyModuleDef_HEAD_INIT,
    "matrixrepr",
    MODULE_DOC,
    -1,
    matrixrepr_methods
};

PyMODINIT_FUNC
PyInit_matrixrepr(void)
{
    return PyModule_Create(&matrixrepr_module);
}
