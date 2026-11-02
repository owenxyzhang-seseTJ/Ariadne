// 放入 _native.cpp 的匿名 namespace，复用已有头文件。
// 独立输入副本保证释放 GIL 后不受调用方修改或对象生命周期影响。
static npy_intp grouping_root(std::vector<npy_intp>& parent, npy_intp i) {
  while(parent[i]!=i) {parent[i]=parent[parent[i]];i=parent[i];}
  return i;
}

PyObject* connected_boxes_int64(PyObject*, PyObject* args) {
  PyObject* object;
  if(!PyArg_ParseTuple(args,"O",&object))return nullptr;
  if(!PyArray_Check(object) || PyArray_TYPE(reinterpret_cast<PyArrayObject*>(object))!=NPY_INT64
      || PyArray_NDIM(reinterpret_cast<PyArrayObject*>(object))!=2
      || PyArray_DIM(reinterpret_cast<PyArrayObject*>(object),1)!=6) {
    PyErr_SetString(PyExc_ValueError,"boxes must be an int64 (n, 6) ndarray");return nullptr;
  }
  auto* array=reinterpret_cast<PyArrayObject*>(PyArray_FromAny(object,PyArray_DescrFromType(NPY_INT64),2,2,
      NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY,nullptr));
  if(!array)return nullptr;
  const npy_intp n=PyArray_DIM(array,0);
  const auto* boxes=static_cast<const npy_int64*>(PyArray_DATA(array));
  std::vector<npy_intp> parent,order,active,roots;
  std::exception_ptr failure;
  Py_BEGIN_ALLOW_THREADS
  try {
    parent.resize(n);order.resize(n);roots.resize(n);active.reserve(n);
    for(npy_intp i=0;i<n;++i)parent[i]=order[i]=i;
    std::stable_sort(order.begin(),order.end(),[&](npy_intp a,npy_intp b){return boxes[6*a+4]<boxes[6*b+4];});
    for(npy_intp i:order) {
      const auto* b=boxes+6*i;
      size_t kept=0;
      // 稳定压缩 active，保留 Python 的遍历顺序及根归并方向。
      for(npy_intp j:active)if(boxes[6*j+5]>b[4])active[kept++]=j;
      active.resize(kept);
      for(npy_intp j:active) {
        const auto* c=boxes+6*j;
        if(b[0]<c[1] && c[0]<b[1] && b[2]<c[3] && c[2]<b[3])
          parent[grouping_root(parent,i)]=grouping_root(parent,j);
      }
      active.push_back(i);
    }
    for(npy_intp i=0;i<n;++i)roots[i]=grouping_root(parent,i);
    // 复用 parent 保存根到首次出现的分组下标，保持输出顺序。
    std::fill(parent.begin(),parent.end(),npy_intp(-1));
  } catch(...) {failure=std::current_exception();}
  Py_END_ALLOW_THREADS
  Py_DECREF(array);
  if(failure) {
    try{std::rethrow_exception(failure);}
    catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());return nullptr;}
    catch(...){PyErr_SetString(PyExc_RuntimeError,"connected-box kernel failed");return nullptr;}
  }
  PyObject* groups=PyList_New(0);
  if(!groups)return nullptr;
  for(npy_intp i=0;i<n;++i) {
    const npy_intp root=roots[i];
    if(parent[root]<0) {
      PyObject* group=PyList_New(0);
      if(!group){Py_DECREF(groups);return nullptr;}
      parent[root]=PyList_GET_SIZE(groups);
      const int status=PyList_Append(groups,group);Py_DECREF(group);
      if(status<0){Py_DECREF(groups);return nullptr;}
    }
    PyObject* index=PyLong_FromSsize_t(i);
    if(!index){Py_DECREF(groups);return nullptr;}
    const int status=PyList_Append(PyList_GET_ITEM(groups,parent[root]),index);Py_DECREF(index);
    if(status<0){Py_DECREF(groups);return nullptr;}
  }
  return groups;
}

// right 为包含式端点。圆周 2*n 展开时 period_size 为原始 n；线性窗口为零。
// 排序、取模、中位数由调用方保留。
PyObject* densest_window(PyObject*, PyObject* args) {
  PyObject* object;double width;Py_ssize_t period;
  if(!PyArg_ParseTuple(args,"Odn",&object,&width,&period))return nullptr;
  if(!PyArray_Check(object) || PyArray_TYPE(reinterpret_cast<PyArrayObject*>(object))!=NPY_DOUBLE
      || PyArray_NDIM(reinterpret_cast<PyArrayObject*>(object))!=1
      || !std::isfinite(width) || width<0 || period<0) {
    PyErr_SetString(PyExc_ValueError,"need a sorted float64 vector, finite nonnegative width and period_size");return nullptr;
  }
  auto* array=reinterpret_cast<PyArrayObject*>(PyArray_FromAny(object,PyArray_DescrFromType(NPY_DOUBLE),1,1,
      NPY_ARRAY_CARRAY | NPY_ARRAY_ENSURECOPY,nullptr));
  if(!array)return nullptr;
  const npy_intp size=PyArray_SIZE(array);
  if(period>0 && (size%2 || period!=size/2)) {
    Py_DECREF(array);PyErr_SetString(PyExc_ValueError,"circular expansion must have 2*period_size entries");return nullptr;
  }
  const double* values=static_cast<const double*>(PyArray_DATA(array));
  npy_intp left=0,best_left=0,best_right=0;
  bool valid=true;
  Py_BEGIN_ALLOW_THREADS
  for(npy_intp i=0;i<size;++i)
    if(!std::isfinite(values[i]) || (i>0 && values[i]<values[i-1])) {valid=false;break;}
  if(valid)for(npy_intp right=0;right<size;++right) {
    while(values[right]-values[left]>width || (period>0 && right-left+1>period))++left;
    if((period==0 || left<period) && right-left>best_right-best_left) {
      best_left=left;best_right=right;
    }
  }
  Py_END_ALLOW_THREADS
  Py_DECREF(array);
  if(!valid){PyErr_SetString(PyExc_ValueError,"window values must be finite and sorted");return nullptr;}
  return Py_BuildValue("nn",static_cast<Py_ssize_t>(best_left),static_cast<Py_ssize_t>(best_right));
}
