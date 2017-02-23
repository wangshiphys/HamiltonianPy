#include <Python.h>
#include "bitselect.h"
#include "convert.h"

static PyObject *bitselect_api(PyObject *self, PyObject *args)
/*{{{*/
{
    long array_len, poses_len, bits_len, judge, rule, buff, count, i;
    long *array_c, *poses_c, *bits_c, *res_c;
    PyObject *array_py=NULL, *poses_py=NULL, *bits_py=NULL, *res_py=NULL;

    if (!PyArg_ParseTuple(args, "OOO", &array_py, &poses_py, &bits_py)) {
        return NULL;
    }
    else if (!(PyTuple_CheckExact(array_py) || PyList_CheckExact(array_py))) {
        PyErr_SetString(PyExc_TypeError, "In function bitselect_api, "
                        "The input array_py is not of type tuple or list.\n");
        return NULL;
    }
    else if (!(PyTuple_CheckExact(poses_py) || PyList_CheckExact(poses_py))) {
        PyErr_SetString(PyExc_TypeError, "In function bitselect_api, "
                        "The input poses_py is not of type tuple or list.\n");
        return NULL;
    }
    else if (!(PyTuple_CheckExact(bits_py) || PyList_CheckExact(bits_py))) {
        PyErr_SetString(PyExc_TypeError, "In function bitselect_api, "
                        "The input bits_py is not of type tuple or list.\n");
        return NULL;
    }

    array_len = PySequence_Length(array_py);
    poses_len = PySequence_Length(poses_py);
    bits_len = PySequence_Length(bits_py);
    if (bits_len != poses_len) {
        PyErr_SetString(PyExc_ValueError, "In function bitselect_api, "
                "the length of poses and bits does not match.\n");
        return NULL;
    }
    else {
        array_c = (long *)PyMem_Calloc(array_len, sizeof(long));
        poses_c = (long *)PyMem_Calloc(poses_len, sizeof(long));
        bits_c = (long *)PyMem_Calloc(bits_len, sizeof(long));
        res_c = (long *)PyMem_Calloc(array_len, sizeof(long));
    }

    if (array_c==NULL || poses_c==NULL || bits_c==NULL || res_c==NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function bitselect_api, "
                        "Unable to allocate the required memory!\n");
        return NULL;
    }
    /*sequence2array return 0 on success -1 on failure.*/
    else if (sequence2array(array_py, array_c, array_len) || 
             sequence2array(poses_py, poses_c, poses_len) ||
             sequence2array(bits_py, bits_c, bits_len)) {
        PyMem_Free(poses_c);
        PyMem_Free(array_c);
        PyMem_Free(bits_c);
        PyMem_Free(res_c);
        return NULL;
    }

    judge = 0;
    rule = 0;
    for (i=0; i<poses_len; ++i) {
        buff = 1<<(*(poses_c + i));
        judge += buff;
        if (*(bits_c + i) == 1) {
            rule += buff;
        }
        else if (*(bits_c + i) == 0) {
            continue;
        }
        else {
            PyErr_SetString(PyExc_ValueError, "In function bitselect_api, "
                    "the entries of bits array should only be 1 or 0.\n");
            return NULL;
        }
    }
    PyMem_Free(poses_c);
    PyMem_Free(bits_c);

    count = 0;
    for (i=0; i<array_len; ++i) {
        /*Bitwise And*/
        if ((*(array_c + i) & judge) == rule) {
            *(res_c + count) = i;
            ++count;
        }
    }

    res_py = array2tuple(res_c, count);
    PyMem_Free(array_c);
    PyMem_Free(res_c);
    return res_py;
}
/*}}}*/

static PyMethodDef bitselect_methods[] = {
    {"bitselect", (PyCFunction)bitselect_api, METH_VARARGS, BITSELECT_DOC},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef bitselect_module = {
    PyModuleDef_HEAD_INIT,
    "bitselect",
    BITSELECTMOD_DOC,
    -1,
    bitselect_methods
};


PyMODINIT_FUNC
PyInit_bitselect(void)
{
    return PyModule_Create(&bitselect_module);
}
