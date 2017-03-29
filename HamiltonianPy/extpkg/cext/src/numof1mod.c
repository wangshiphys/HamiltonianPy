#include <Python.h>
#include "convert.h"
#include "numof1.h"

static PyObject *numof1_api(PyObject *self, PyObject *args)
/*{{{*/
{
    int p0, p1;
    long *array_c, *res_c, length, i, buff;
    PyObject *array_py = NULL, *res_py=NULL;

    if (!PyArg_ParseTuple(args, "iiO", &p0, &p1, &array_py)) {
        return NULL;
    }
    else if (!(PyTuple_CheckExact(array_py) || PyList_CheckExact(array_py))) {
        PyErr_SetString(PyExc_TypeError, "In function numof1_api, "
                        "The input array is not of type tupleor list!\n");
        return NULL;
    }

    length = (long)PySequence_Length(array_py);
    array_c = (long *)PyMem_Calloc(length, sizeof(long));
    res_c = (long *)PyMem_Calloc(length, sizeof(long));
   
    if (res_c==NULL || array_c==NULL) {
        PyErr_SetString(PyExc_MemoryError, "In function numof1_api, "
                        "Unable to allocate the required memory!\n");
        return NULL;
    }
    /*sequence return 0 on success -1 on failure.*/
    else if (sequence2array(array_py, array_c, length)) {
        PyMem_Free(array_c);
        PyMem_Free(res_c);
        return NULL;
    }
    
    for(i=0; i<length; ++i) {
        buff = *(array_c + i);
        *(res_c + i) = numof1(buff, p0, p1);
    }

    res_py = array2tuple(res_c, length);
    PyMem_Free(res_c);
    PyMem_Free(array_c);
    return res_py;
}
/*}}}*/


static PyMethodDef numof1_methods[] = {
    {"numof1", (PyCFunction)numof1_api, METH_VARARGS, NUMOF1_DOC},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef numof1_module = {
    PyModuleDef_HEAD_INIT,
    "numof1",
    NUMOF1MOD_DOC,
    -1,
    numof1_methods
};


PyMODINIT_FUNC
PyInit_numof1(void)
{
    return PyModule_Create(&numof1_module);
}



