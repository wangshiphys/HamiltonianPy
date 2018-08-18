#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

#define ANNIHILATION 0
#define CREATION 1

#define FALSE 0
#define TRUE 1

#define PREFIX "In function matrix_repr_c_api, "


static long binary_search(const long aim, const long *list, const long n)
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
    fprintf(stderr, "Error in binary_search function, the `aim` does not contained in the given list!\n");
    exit(EXIT_FAILURE);
}


static PyObject *matrix_repr_c_api(PyObject *self, PyObject *args)
{
    char *exc_msg;
    int swap, not_zero_flag;
    long state_index, mask, criterion, ket;
    Py_ssize_t i, j, pos;

    PyObject *Py_term = NULL, *Py_right_bases = NULL, *Py_left_bases = NULL;
    long *C_term = NULL, *C_right_bases = NULL, *C_left_bases = NULL;
    Py_ssize_t term_length, right_dim, left_dim;

    PyObject *Py_row_index = NULL, *Py_column_index = NULL, *Py_entry = NULL;
    PyObject *Py_row_indices = NULL, *Py_column_indices = NULL, *Py_entries = NULL, *res = NULL;

    if (!PyArg_ParseTuple(args, "OO|O", &Py_term, &Py_right_bases, &Py_left_bases)) {
        return NULL;
    }

//    Convert the python tuple Py_term to C array
    term_length = PyTuple_GET_SIZE(Py_term);
    C_term = (long *)PyMem_Calloc(term_length, sizeof(long));
    if (C_term != NULL) {
        for (i=0; i<term_length; i++) {
            *(C_term + i) = PyLong_AsLong(PyTuple_GET_ITEM(Py_term, i));
        }
    } else {
        exc_msg = PREFIX"unable to allocate the required memory for C_term.\n";
        PyErr_SetString(PyExc_MemoryError, exc_msg);
        return NULL;
    }

//    Convert the python tuple Py_right_bases to C array
    right_dim = PyTuple_GET_SIZE(Py_right_bases);
    C_right_bases = (long *)PyMem_Calloc(right_dim, sizeof(long));
    if (C_right_bases!= NULL) {
        for(i=0; i<right_dim; ++i) {
            *(C_right_bases + i) = PyLong_AsLong(PyTuple_GET_ITEM(Py_right_bases, i));
        }
    } else {
        exc_msg = PREFIX"unable to allocate the required memory for C_right_bases.\n";
        PyErr_SetString(PyExc_MemoryError, exc_msg);
        return NULL;
    }

//    Convert the python tuple Py_left_bases to C array if necessary
    if (Py_left_bases == NULL || Py_left_bases == Py_None) {
        left_dim = right_dim;
        C_left_bases = C_right_bases;
    } else {
        left_dim = PyTuple_GET_SIZE(Py_left_bases);
        C_left_bases = (long *)PyMem_Calloc(left_dim, sizeof(long));
        if (C_left_bases != NULL) {
            for(i=0; i<left_dim; ++i) {
                *(C_left_bases + i) = PyLong_AsLong(PyTuple_GET_ITEM(Py_left_bases, i));
            }
        } else {
            exc_msg = PREFIX"unable to allocate the required memory for C_left_bases.\n";
            PyErr_SetString(PyExc_MemoryError, exc_msg);
            return NULL;
        }
    }

//    Calculate the matrix representation of this term
    Py_row_indices = PyList_New(0);
    Py_column_indices = PyList_New(0);
    Py_entries = PyList_New(0);
    if (Py_row_indices != NULL && Py_column_indices != NULL && Py_entries != NULL) {
        for (i=0; i<right_dim; ++i) {
            swap = 0;
            not_zero_flag = TRUE;
            ket = *(C_right_bases + i);
            for (j=term_length-1; j>=0; ) {
                criterion = C_term[j--];
                mask = C_term[j--];
                state_index = C_term[j--];
                if ((ket & mask) == criterion) {
                    for (pos=0; pos<state_index; pos++) {
                        if (ket & (1 << pos)) swap ^= 1;
                    }
                    ket ^= mask;
                } else {
                    not_zero_flag = FALSE;
                    break;
                }
            }
            if (not_zero_flag) {
                Py_row_index = PyLong_FromLong(binary_search(ket, C_left_bases, (long)left_dim));
                PyList_Append(Py_row_indices, Py_row_index);
                Py_DECREF(Py_row_index);

                Py_column_index = PyLong_FromLong((long)i);
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
