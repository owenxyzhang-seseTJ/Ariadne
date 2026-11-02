// 依原贡献列顺序累加共享强度设计，保留重复 equivalence ID 的贡献。
PyObject* scaling_design(PyObject*,PyObject* args) {
  PyObject *design_object,*gain_object,*index_object;Py_ssize_t columns;
  if(!PyArg_ParseTuple(args,"OOOn",&design_object,&gain_object,&index_object,&columns))return nullptr;
  PyArrayObject* design=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(design_object,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  PyArrayObject* gain=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(gain_object,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  PyArrayObject* index=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(index_object,NPY_INTP,NPY_ARRAY_IN_ARRAY));
  if(!design || !gain || !index){Py_XDECREF(design);Py_XDECREF(gain);Py_XDECREF(index);return nullptr;}
  bool valid=PyArray_NDIM(design)==2 && PyArray_NDIM(gain)==1 && PyArray_NDIM(index)==1 && columns>=0;
  npy_intp rows=0,contributions=0;
  if(valid){rows=PyArray_DIM(design,0);contributions=PyArray_DIM(design,1);
    valid=PyArray_SIZE(gain)==contributions && PyArray_SIZE(index)==contributions;}
  const npy_intp* ids=static_cast<npy_intp*>(PyArray_DATA(index));
  if(valid)for(npy_intp j=0;j<contributions;++j)if(ids[j]<0 || ids[j]>=columns){valid=false;break;}
  if(!valid){Py_DECREF(design);Py_DECREF(gain);Py_DECREF(index);PyErr_SetString(PyExc_ValueError,"Invalid scaling design arrays or column indices");return nullptr;}
  npy_intp shape[2]={rows,columns};
  PyArrayObject* result=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(2,shape,NPY_DOUBLE,0));
  if(!result){Py_DECREF(design);Py_DECREF(gain);Py_DECREF(index);return nullptr;}
  const double *d=static_cast<double*>(PyArray_DATA(design)),*g=static_cast<double*>(PyArray_DATA(gain));
  double* out=static_cast<double*>(PyArray_DATA(result));
  Py_BEGIN_ALLOW_THREADS
  for(npy_intp row=0;row<rows;++row)for(npy_intp j=0;j<contributions;++j) {
    // 不融合乘加，避免与 NumPy 原有逐列累加产生消去误差差异。
    volatile double term=d[row*contributions+j]*g[j];
    out[row*columns+ids[j]]+=term;
  }
  Py_END_ALLOW_THREADS
  Py_DECREF(design);Py_DECREF(gain);Py_DECREF(index);
  return reinterpret_cast<PyObject*>(result);
}
