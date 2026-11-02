// 完整候选集合的三维 Gaussian 责任度；不做近邻截断或后验裁剪。
PyObject* centroid_expectation(PyObject*,PyObject* args) {
  PyObject *objects[7];double fraction;int block;
  if(!PyArg_ParseTuple(args,"OOOOOOOdi",&objects[0],&objects[1],&objects[2],&objects[3],&objects[4],&objects[5],&objects[6],&fraction,&block))return nullptr;
  PyArrayObject* arrays[7]={};
  for(int j=0;j<7;++j) {
    arrays[j]=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(objects[j],j==4?NPY_BOOL:NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
    if(!arrays[j]){for(auto* a:arrays)Py_XDECREF(a);return nullptr;}
  }
  auto dimension=[&](int array,int axis){return PyArray_NDIM(arrays[array])>axis?PyArray_DIM(arrays[array],axis):npy_intp(-1);};
  const npy_intp n=dimension(0,0),m=dimension(1,0);
  bool valid=PyArray_NDIM(arrays[0])==2 && dimension(0,1)==3
    && PyArray_NDIM(arrays[1])==2 && dimension(1,1)==3
    && PyArray_NDIM(arrays[2])==3 && dimension(2,0)==n && dimension(2,1)==3 && dimension(2,2)==3
    && PyArray_NDIM(arrays[4])==2 && dimension(4,0)==n && dimension(4,1)==m
    && fraction>0 && fraction<1 && block>0;
  for(int j:{3,5,6})valid&=PyArray_NDIM(arrays[j])==1 && PyArray_SIZE(arrays[j])==n;
  if(!valid){for(auto* a:arrays)Py_DECREF(a);PyErr_SetString(PyExc_ValueError,"Invalid centroid expectation shapes or settings");return nullptr;}
  const double *y=static_cast<double*>(PyArray_DATA(arrays[0])),*means=static_cast<double*>(PyArray_DATA(arrays[1]));
  const double *white=static_cast<double*>(PyArray_DATA(arrays[2])),*norm=static_cast<double*>(PyArray_DATA(arrays[3]));
  const npy_bool* allowed=static_cast<npy_bool*>(PyArray_DATA(arrays[4]));
  const double *counts=static_cast<double*>(PyArray_DATA(arrays[5])),*density=static_cast<double*>(PyArray_DATA(arrays[6]));
  for(npy_intp i=0;i<n;++i)if(!(counts[i]>0) || !(density[i]>0))valid=false;
  if(!valid){for(auto* a:arrays)Py_DECREF(a);PyErr_SetString(PyExc_ValueError,"Need positive candidate counts and outlier density");return nullptr;}
  npy_intp shape[2]={n,m},vector_shape[1]={n};
  PyArrayObject* responsibilities=reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2,shape,NPY_DOUBLE));
  PyArrayObject* outlier=reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1,vector_shape,NPY_DOUBLE));
  PyArrayObject* logs=reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1,vector_shape,NPY_DOUBLE));
  if(!responsibilities || !outlier || !logs){Py_XDECREF(responsibilities);Py_XDECREF(outlier);Py_XDECREF(logs);for(auto* a:arrays)Py_DECREF(a);return nullptr;}
  double *r=static_cast<double*>(PyArray_DATA(responsibilities)),*o=static_cast<double*>(PyArray_DATA(outlier)),*ll=static_cast<double*>(PyArray_DATA(logs));
  Py_BEGIN_ALLOW_THREADS
  const npy_intp columns=std::min(m,npy_intp(std::min(2048,block)));
  for(npy_intp i=0;i<n;++i) {
    const double normalizer=norm[i]+std::log1p(-fraction)-std::log(counts[i]);
    double total=-std::numeric_limits<double>::infinity();
    for(npy_intp first=0;first<m;first+=columns) {
      const npy_intp last=std::min(m,first+columns);double maximum=-std::numeric_limits<double>::infinity();
      for(npy_intp j=first;j<last;++j) {
        double value=-std::numeric_limits<double>::infinity();
        if(allowed[i*m+j]) {
          const double d[3]={y[3*i]-means[3*j],y[3*i+1]-means[3*j+1],y[3*i+2]-means[3*j+2]};
          double residual[3];
          for(int axis=0;axis<3;++axis) {
            const double* w=white+9*i+3*axis;
            residual[axis]=(w[0]*d[0]+w[1]*d[1])+w[2]*d[2];
          }
          const double square=(residual[0]*residual[0]+residual[1]*residual[1])+residual[2]*residual[2];
          value=normalizer-.5*square;
        }
        r[i*m+j]=value;maximum=std::max(maximum,value);
      }
      if(maximum!=-std::numeric_limits<double>::infinity()) {
        double sum=0.;for(npy_intp j=first;j<last;++j)sum+=std::exp(r[i*m+j]-maximum);
        total=pg_logadd(total,maximum+std::log(sum));
      }
    }
    const double log_outlier=std::log(fraction)+std::log(density[i]);
    ll[i]=pg_logadd(total,log_outlier);o[i]=std::exp(log_outlier-ll[i]);
    for(npy_intp j=0;j<m;++j)r[i*m+j]=std::exp(r[i*m+j]-ll[i]);
  }
  Py_END_ALLOW_THREADS
  for(auto* a:arrays)Py_DECREF(a);
  return Py_BuildValue("NNN",reinterpret_cast<PyObject*>(responsibilities),reinterpret_cast<PyObject*>(outlier),reinterpret_cast<PyObject*>(logs));
}
