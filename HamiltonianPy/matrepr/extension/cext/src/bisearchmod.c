#include <Python.h>
#include "bisearch.h"
#include "convert.h"

static PyObject *bisearch_api(PyObject *self, PyObject *args)
/*{{{*/
{
    PyObject *aims_py = NULL, *seq_py = NULL, *res_py=NULL;
    long *aims_c, *array_c, *res_c, aims_len, array_len, i;

    if (!PyArg_ParseTuple(args, "OO", &aims_py, &seq_py)) {
        return NULL;
    }
    else if (!(PyTuple_CheckExact(aims_py) || PyList_CheckExact(aims_py))) {
        PyErr_SetString(PyExc_TypeError, "In function bisearch_api, "
                        "The input aims_py is not of type tuple or list.\n");
        return NULL;
    }
    else if (!(PyTuple_CheckExact(seq_py) || PyList_CheckExact(seq_py))) {
        PyErr_SetString(PyExc_TypeError, "In function bisearch_api, "
                        "The input seq_py is not of type tuple or list.\n");
        return NULL;
    }

    aims_len = (long)PySequence_Length(aims_py);
    array_len = (long)PySequence_Length(seq_py);
    aims_c = (long *)PyMem_Calloc(aims_len, sizeof(long));
    res_c = (long *)PyMem_Calloc(aims_len, sizeof(long));
    array_c = (long *)PyMem_Calloc(array_len, sizeof(long));
   
    if (aims_c==NULL || res_c==NULL || array_c==NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function bisearch_api, "
                        "Unable to allocate the required memory!\n");
        return NULL;
    }
    /*sequence2array return 0 on success and -1 on failure.*/
    else if (sequence2array(aims_py, aims_c, aims_len) || 
             sequence2array(seq_py, array_c, array_len)) {
        PyMem_Free(aims_c);
        PyMem_Free(array_c);
        PyMem_Free(res_c);
        return NULL;
    }
    
    for(i=0; i<aims_len; ++i) {
        *(res_c + i) = bisearch(*(aims_c + i), array_c, array_len);
    }

    res_py = array2tuple(res_c, aims_len);
    PyMem_Free(aims_c);
    PyMem_Free(res_c);
    PyMem_Free(array_c);
    return res_py;
}
/*}}}*/


static PyMethodDef bisearch_methods[] = {
    {"bisearch", (PyCFunction)bisearch_api, METH_VARARGS, BISEARCH_DOC},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef bisearch_module = {
    PyModuleDef_HEAD_INIT,
    "bisearch",
    BISEARCHMOD_DOC,
    -1,
    bisearch_methods
};


PyMODINIT_FUNC
PyInit_bisearch(void)
{
    return PyModule_Create(&bisearch_module);
}
