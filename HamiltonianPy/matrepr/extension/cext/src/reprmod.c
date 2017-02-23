#include <Python.h>
#include <omp.h>
#include "convert.h"
#include "repr.h"

static PyObject *construct(const long *const row_c, const long *const col_c,
                             const int *const elmts_c, const long dim)
/*{{{*/
{
    int elmt, ok;
    long i;
    PyObject *row_obj=NULL, *col_obj=NULL, *elmt_obj=NULL;
    PyObject *row_py=NULL, *col_py=NULL, *elmts_py=NULL, *res=NULL;

    row_py = PyList_New(0);
    col_py = PyList_New(0);
    elmts_py = PyList_New(0);
    if (row_py == NULL || col_py == NULL || elmts_py == NULL) {
        Py_XDECREF(row_py);
        Py_XDECREF(col_py);
        Py_XDECREF(elmts_py);
        return NULL;
    }

    for (i=0; i<dim; ++i) {
        elmt = *(elmts_c + i);
        if (elmt) {
            row_obj = PyLong_FromLong(*(row_c + i));
            col_obj = PyLong_FromLong(*(col_c + i));
            elmt_obj = PyLong_FromLong(elmt);
            if (row_obj == NULL || col_obj == NULL || elmt_obj == NULL) {
                Py_XDECREF(row_obj);
                Py_XDECREF(col_obj);
                Py_XDECREF(elmt_obj);
                return NULL;
            }

            ok = PyList_Append(row_py, row_obj);
            Py_DECREF(row_obj);
            if (ok == -1) {
                return NULL;
            }

            ok = PyList_Append(col_py, col_obj);
            Py_DECREF(col_obj);
            if (ok == -1) {
                return NULL;
            }

            ok = PyList_Append(elmts_py, elmt_obj);
            Py_DECREF(elmt_obj);
            if (ok == -1) {
                return NULL;
            }
        }
    }
    res = Py_BuildValue("OOO", row_py, col_py, elmts_py);
    Py_DECREF(row_py);
    Py_DECREF(col_py);
    Py_DECREF(elmts_py);
    if (res == NULL) {
        Py_XDECREF(res);
        return NULL;
    }
    
    return res;
}
/*}}}*/

static PyObject *hopping_api(PyObject *self, PyObject *args)
/*{{{*/
{
    int cindex, aindex, *elmts_c;
    long dim, *base_c, *row_c, *col_c;
    PyObject *base_py = NULL, *res_py = NULL;

    if (!PyArg_ParseTuple(args, "iiO", &cindex, &aindex, &base_py)) {
        return NULL;
    }
    else if (!(PyTuple_CheckExact(base_py) || PyList_CheckExact(base_py))) {
        PyErr_SetString(PyExc_TypeError, "In function hopping_api, "
                        "The third input parameter is not a tuple or list!\n");
        return NULL;
    }

    dim = (long)PySequence_Length(base_py);
    base_c = (long *)PyMem_Calloc(dim, sizeof(long));
    if (base_c == NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function hopping_api, "
                        "Unable to allocate the required memory for base_c!\n");
        return NULL;
    }
    /*sequence2array return 0 on success -1 on failure.*/
    else if (sequence2array(base_py, base_c, dim)) {
        PyMem_Free(base_c);
        return NULL;
    }
    
    row_c = (long *)PyMem_Calloc(dim , sizeof(long));
    col_c = (long *)PyMem_Calloc(dim , sizeof(long));
    elmts_c = (int *)PyMem_Calloc(dim , sizeof(int));
    if (row_c == NULL || col_c == NULL || elmts_c == NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function hopping_api, "
                        "Unable to allocate the required memory!\n");
        return NULL;
    }

    hopping_C(cindex, aindex, base_c, dim, row_c, col_c, elmts_c);
    res_py = construct(row_c, col_c, elmts_c, dim);
    PyMem_Free(base_c);
    PyMem_Free(row_c);
    PyMem_Free(col_c);
    PyMem_Free(elmts_c);
    return res_py;
}
/*}}}*/

static PyObject *hubbard_api(PyObject *self, PyObject *args)
/*{{{*/
{
    int index0, index1, *elmts_c;
    long dim, *base_c, *row_c, *col_c;
    PyObject *base_py = NULL, *res_py = NULL;

    if (!PyArg_ParseTuple(args, "iiO", &index0, &index1, &base_py)) {
        return NULL;
    }
    else if (!(PyTuple_CheckExact(base_py) || PyList_CheckExact(base_py))) {
        PyErr_SetString(PyExc_TypeError, "In function hubbard_api, "
                        "The third input parameter is not a tuple or list!\n");
        return NULL;
    }

    dim = (long)PySequence_Length(base_py);
    base_c = (long *)PyMem_Calloc(dim, sizeof(long));
    if (base_c == NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function hubbard_api, "
                        "Unable to allocate the required memory for base_c!\n");
        return NULL;
    }
    /*sequence2array return 0 on success -1 on failure.*/
    else if (sequence2array(base_py, base_c, dim)) {
        PyMem_Free(base_c);
        return NULL;
    }
    
    row_c = (long *)PyMem_Calloc(dim , sizeof(long));
    col_c = (long *)PyMem_Calloc(dim , sizeof(long));
    elmts_c = (int *)PyMem_Calloc(dim , sizeof(int));
    if (row_c == NULL || col_c == NULL || elmts_c == NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function hubbard_api, "
                        "Unable to allocate the required memory!\n");
        return NULL;
    }

    hubbard_C(index0, index1, base_c, dim, row_c, col_c, elmts_c);
    res_py = construct(row_c, col_c, elmts_c, dim);
    PyMem_Free(base_c);
    PyMem_Free(row_c);
    PyMem_Free(col_c);
    PyMem_Free(elmts_c);
    return res_py;
}
/*}}}*/

static PyObject *pairing_api(PyObject *self, PyObject *args)
/*{{{*/
{
    int index0, index1, otype, *elmts_c;
    long dim, *base_c, *row_c, *col_c;
    PyObject *base_py = NULL, *res_py = NULL;

    if (!PyArg_ParseTuple(args, "iiiO", &index0, &index1, &otype, &base_py)) {
        return NULL;
    }
    else if (!(PyTuple_CheckExact(base_py) || PyList_CheckExact(base_py))) {
        PyErr_SetString(PyExc_TypeError, "In function pairing_api, "
                        "The fourth input parameter is not a tuple or list!\n");
        return NULL;
    }

    dim = (long)PySequence_Length(base_py);
    base_c = (long *)PyMem_Calloc(dim, sizeof(long));
    if (base_c == NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function pairing_api, "
                        "Unable to allocate the required memory for base_c!\n");
        return NULL;
    }
    else if (sequence2array(base_py, base_c, dim)) {
        PyMem_Free(base_c);
        return NULL;
    }
    
    row_c = (long *)PyMem_Calloc(dim , sizeof(long));
    col_c = (long *)PyMem_Calloc(dim , sizeof(long));
    elmts_c = (int *)PyMem_Calloc(dim , sizeof(int));
    if (row_c == NULL || col_c == NULL || elmts_c == NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function pairing_api, "
                        "Unable to allocate the required memory!\n");
        return NULL;
    }

    pairing_C(index0, index1, otype, base_c, dim, row_c, col_c, elmts_c);
    res_py = construct(row_c, col_c, elmts_c, dim);
    PyMem_Free(base_c);
    PyMem_Free(row_c);
    PyMem_Free(col_c);
    PyMem_Free(elmts_c);
    return res_py;
}
/*}}}*/

static PyObject *aoc_api(PyObject *self, PyObject *args)
/*{{{*/
{
    int index, otype, *elmts_c;
    long ldim, rdim, *lbase_c, *rbase_c, *row_c, *col_c;
    PyObject *lbase_py = NULL, *rbase_py=NULL, *res_py = NULL;

    if (!PyArg_ParseTuple(args, "iiO|O", &index, &otype, &lbase_py, &rbase_py))
    {
        return NULL;
    }
    else if (!(PyTuple_CheckExact(lbase_py) || PyList_CheckExact(lbase_py))) {
        PyErr_SetString(PyExc_TypeError, "In function aoc_api, "
                        "The third input parameter is not a tuple or list!\n");
        return NULL;
    }

    ldim = (long)PySequence_Length(lbase_py);
    lbase_c = (long *)PyMem_Calloc(ldim, sizeof(long));
    if (lbase_c == NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function aoc_api, "
                        "Unable to allocate the required memory for base_c!\n");
        return NULL;
    }
    else if (sequence2array(lbase_py, lbase_c, ldim)) {
        PyMem_Free(lbase_c);
        return NULL;
    }
    
    if (rbase_py == NULL) {
        rdim = ldim;
        rbase_c = lbase_c;
    }
    else {
        if (!(PyTuple_CheckExact(rbase_py) || PyList_CheckExact(rbase_py))) {
            PyErr_SetString(PyExc_TypeError, "In function aoc_api, the "
                            "fourth input parameter is not a tuple or list!\n");
            return NULL;
        }

        rdim = (long)PySequence_Length(rbase_py);
        rbase_c = (long *)PyMem_Calloc(rdim, sizeof(long));
        if (rbase_c == NULL) {
            PyErr_SetString(PyExc_MemoryError, "In function aoc_api, "
                            "Unable to allocate the required memory for base_c!\n");
            return NULL;
        }
        else if (sequence2array(rbase_py, rbase_c, rdim)) {
            PyMem_Free(rbase_c);
            return NULL;
        }
    }
    
    row_c = (long *)PyMem_Calloc(rdim , sizeof(long));
    col_c = (long *)PyMem_Calloc(rdim , sizeof(long));
    elmts_c = (int *)PyMem_Calloc(rdim , sizeof(int));
    if (row_c == NULL || col_c == NULL || elmts_c == NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function aoc_api, "
                        "Unable to allocate the required memory!\n");
        return NULL;
    }

    aoc_C(index, otype, lbase_c, rbase_c, ldim, rdim, row_c, col_c, elmts_c);
    res_py = construct(row_c, col_c, elmts_c, rdim);
    PyMem_Free(lbase_c);
    PyMem_Free(row_c);
    PyMem_Free(col_c);
    PyMem_Free(elmts_c);
    if (rbase_c != lbase_c) {
        PyMem_Free(rbase_c);
    }
    return res_py;
}
/*}}}*/

static PyMethodDef matreprcext_methods[] = {
    {"hopping", (PyCFunction)hopping_api, METH_VARARGS, HOPPING_DOC},
    {"hubbard", (PyCFunction)hubbard_api, METH_VARARGS, HUBBARD_DOC},
    {"pairing", (PyCFunction)pairing_api, METH_VARARGS, PAIRING_DOC},
    {"aoc", (PyCFunction)aoc_api, METH_VARARGS, AOC_DOC},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef matreprcext_module = {
    PyModuleDef_HEAD_INIT,
    "matreprcext",
    MATREPRCEXT_DOC,
    -1,
    matreprcext_methods
};


PyMODINIT_FUNC
PyInit_matreprcext(void)
{
    return PyModule_Create(&matreprcext_module);
}



