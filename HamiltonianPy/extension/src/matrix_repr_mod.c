#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

#define ANNIHILATION 0
#define CREATION 1

#define FALSE 0
#define TRUE 1

#define PREFIX "In function matrix_repr_c_api, "

typedef unsigned long long uint64;


static Py_ssize_t binary_search(const uint64 aim, const uint64 *list, const Py_ssize_t n)
{
    uint64 buff;
    Py_ssize_t low, mid, high;
    low = 0;
    high = n - 1;
    while (low <= high) {
        mid = (high - low) / 2 + low;
        buff = *(list + mid);
        if (aim == buff) {
            return mid;
        } else if (aim > buff) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    fprintf(stderr, "Error in binary_search function, the `aim` does not contained in the given list!\n");
    exit(EXIT_FAILURE);
}


//The contents contained in the `tuple` object are non-negative PyLongObject
static uint64 *PyTuple_to_CArray(PyObject *tuple, const Py_ssize_t length)
{
    uint64 *array;
    Py_ssize_t pos;

    array = (uint64 *)PyMem_Calloc(length, sizeof(uint64));
    if (array != NULL) {
        for (pos=0; pos<length; pos++) {
            *(array + pos) = PyLong_AsUnsignedLongLong(PyTuple_GET_ITEM(tuple, pos));
        }
    }
    return array;
}


static PyObject *matrix_repr_c_api(PyObject *self, PyObject *args)
{
    char *exc_msg;
    int swap, not_zero_flag;
    uint64 one, pos, state_index, mask, criterion, ket;

    PyObject *Py_term = NULL, *Py_right_bases = NULL, *Py_left_bases = NULL;
    uint64 *C_term = NULL, *C_right_bases = NULL, *C_left_bases = NULL;
    Py_ssize_t i, j, term_length, right_dim, left_dim;

    PyObject *Py_row_index = NULL, *Py_column_index = NULL, *Py_entry = NULL;
    PyObject *Py_row_indices = NULL, *Py_column_indices = NULL, *Py_entries = NULL, *res = NULL;

    if (!PyArg_ParseTuple(args, "OO|O", &Py_term, &Py_right_bases, &Py_left_bases)) {
        return NULL;
    }

//    Convert the python tuple Py_term and Py_right_bases to C array
    term_length = PyTuple_GET_SIZE(Py_term);
    right_dim = PyTuple_GET_SIZE(Py_right_bases);
    C_term = PyTuple_to_CArray(Py_term, term_length);
    C_right_bases = PyTuple_to_CArray(Py_right_bases, right_dim);
    if (C_term == NULL || C_right_bases == NULL) {
        exc_msg = PREFIX"unable to allocate the required memory.\n";
        PyErr_SetString(PyExc_MemoryError, exc_msg);
        return NULL;
    }

//    Convert the python tuple Py_left_bases to C array if necessary
    if (Py_left_bases == NULL || Py_left_bases == Py_None) {
        left_dim = right_dim;
        C_left_bases = C_right_bases;
    } else {
        left_dim = PyTuple_GET_SIZE(Py_left_bases);
        C_left_bases = PyTuple_to_CArray(Py_left_bases, left_dim);
        if (C_left_bases == NULL) {
            exc_msg = PREFIX"unable to allocate the required memory for C_left_bases.\n";
            PyErr_SetString(PyExc_MemoryError, exc_msg);
            return NULL;
        }
    }

//    Calculate the matrix representation of this term
    one = 1;
    Py_row_indices = PyList_New(0);
    Py_column_indices = PyList_New(0);
    Py_entries = PyList_New(0);
    if (Py_row_indices != NULL && Py_column_indices != NULL && Py_entries != NULL) {
        for (i=0; i<right_dim; i++) {
            swap = 0;
            not_zero_flag = TRUE;
            ket = *(C_right_bases + i);
            for (j=term_length-1; j>=0; ) {
                criterion = C_term[j--];
                mask = C_term[j--];
                state_index = C_term[j--];
                if ((ket & mask) == criterion) {
                    for (pos=0; pos<state_index; pos++) {
                        if (ket & (one << pos)) swap ^= 1;
                    }
                    ket ^= mask;
                } else {
                    not_zero_flag = FALSE;
                    break;
                }
            }
            if (not_zero_flag) {
                Py_row_index = PyLong_FromSsize_t(binary_search(ket, C_left_bases, left_dim));
                PyList_Append(Py_row_indices, Py_row_index);
                Py_DECREF(Py_row_index);

                Py_column_index = PyLong_FromSsize_t(i);
                PyList_Append(Py_column_indices, Py_column_index);
                Py_DECREF(Py_column_index);

                Py_entry = PyFloat_FromDouble(swap? -1.0: 1.0);
                PyList_Append(Py_entries, Py_entry);
                Py_DECREF(Py_entry);
            }
        }
    } else {
        exc_msg = PREFIX"unable to create empty lists to hold row_indices, column_indices and entries.\n";
        PyErr_SetString(PyExc_RuntimeError, exc_msg);
        return NULL;
    }

//    Release the allocate memory
    PyMem_Free(C_term);
    if (C_left_bases != C_right_bases) {
        PyMem_Free(C_right_bases);
        PyMem_Free(C_left_bases);
    } else {
        PyMem_Free(C_right_bases);
    }

//    Construct the return python object
    res = Py_BuildValue("(O(OO))", Py_entries, Py_row_indices, Py_column_indices);
//    Decrease the reference count
    Py_DECREF(Py_entries);
    Py_DECREF(Py_row_indices);
    Py_DECREF(Py_column_indices);

    return res;
}


static PyMethodDef matrix_repr_methods[] = {
    {"matrix_repr_c_api", (PyCFunction)matrix_repr_c_api, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef matrix_repr_c_module = {
    PyModuleDef_HEAD_INIT,
    "matrix_repr_c_mod",
    NULL,
    -1,
    matrix_repr_methods
};

PyMODINIT_FUNC
PyInit_matrix_repr_c_mod(void)
{
    return PyModule_Create(&matrix_repr_c_module);
}
