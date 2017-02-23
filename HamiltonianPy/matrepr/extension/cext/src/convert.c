#include <Python.h>
#include <omp.h>
#include "constant.h"

int sequence2array(PyObject *seq, long *const array, const long length)
/*{{{*/
{
    Py_ssize_t i;
    PyObject *obj=NULL;
    if (!(PyTuple_CheckExact(seq) || PyList_CheckExact(seq))) {
        PyErr_SetString(PyExc_TypeError, "In function sequence2array, the "
                        "input seq parameter is not of tuple or list type.\n");
    }
    else if (PySequence_Length(seq) != length) {
        PyErr_SetString(PyExc_ValueError, "In function sequence2array, the "
        "length of the seq does not match the input length parameter.\n");
    }
    else {
        #pragma omp parallel if(length>NUM_TO_PARALLEL) num_threads(NUM_THREADS)
        {
            //printf("The number of threads: %d\n", omp_get_num_threads());
            #pragma omp for private(i, obj)
            for (i=0; i<length; ++i) {
                obj = PySequence_ITEM(seq, i);
                if (obj == NULL) {
                    Py_XDECREF(obj);
                    continue;
                }
                else {
                    *(array + i) = PyLong_AsLong(obj);
                    Py_DECREF(obj);
                }
            }
        }
    }

    if (PyErr_Occurred() == NULL) {
        return 0;
    }
    else {
        return -1;
    }
}
/*}}}*/


PyObject *array2tuple(const long *const array, const long length)
/*{{{*/
{
    Py_ssize_t i;
    PyObject *res_py=NULL, *obj=NULL;

    res_py = PyTuple_New(length);
    if (res_py == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "In function array2tuple, "
                                            "failed to creat a new tuple.\n");
        Py_XDECREF(res_py);
        return NULL;
    }

    for (i=0; i<length; ++i) {
        obj = PyLong_FromLong(*(array + i));
        if (obj == NULL) {
            Py_XDECREF(obj);
            return NULL;
        }

        /*PyTuple_SetItem return 0 on success*/
        if (PyTuple_SetItem(res_py, i, obj)) {
            Py_DECREF(obj);
            return NULL;
        }
    }
    return res_py;
}
/*}}}*/


PyObject *array2list(const long *const array, const long length)
/*{{{*/
{
    Py_ssize_t i;
    PyObject *res_py=NULL, *obj=NULL;

    res_py = PyList_New(length);
    if (res_py == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "In function array2list, "
                                            "failed to creat a new list.\n");
        Py_XDECREF(res_py);
        return NULL;
    }

    for (i=0; i<length; ++i) {
        obj = PyLong_FromLong(*(array + i));
        if (obj == NULL) {
            Py_XDECREF(obj);
            return NULL;
        }

        /*PyList_SetItem return 0 on success*/
        if (PyList_SetItem(res_py, i, obj)) {
            Py_DECREF(obj);
            return NULL;
        }
    }
    return res_py;
}
/*}}}*/
