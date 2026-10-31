// Ariadne native count-excess proposals. The statistical rules match spots.py.
// No thresholds, noise calibration or measured pixel values are fitted here.
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

namespace {
using Vec=std::vector<double>;

Vec integral(const Vec& values, size_t height, size_t width) {
  Vec result((height+1)*(width+1),0.0);
  const size_t stride=width+1;
  for(size_t y=0;y<height;++y) {
    double row=0;
    for(size_t x=0;x<width;++x) {
      row+=values[y*width+x];
      result[(y+1)*stride+x+1]=result[y*stride+x+1]+row;
    }
  }
  return result;
}

inline double box(const Vec& table, size_t y, size_t x, size_t height, size_t width, int window) {
  const size_t r=static_cast<size_t>(window/2), stride=width+1;
  const size_t y0=y>r?y-r:0, x0=x>r?x-r:0;
  const size_t y1=std::min(height,y+r+1), x1=std::min(width,x+r+1);
  return (table[y1*stride+x1]-table[y0*stride+x1])-(table[y1*stride+x0]-table[y0*stride+x0]);
}

void proposals(const double* input, const npy_bool* mask, npy_bool* output,
               size_t height,size_t width,double readout,int poisson,int bw,int ew,
               const std::vector<int>& scales,double threshold,double support,double fraction,int iterations) {
  const size_t count=height*width;
  Vec data(count),weights(count),retained(count),mean(count),counts(count),variance(count),best(count,0.0);
  std::vector<unsigned char> usable(count),supported(count);
  for(size_t i=0;i<count;++i) {
    weights[i]=mask[i]?1.:0.;retained[i]=weights[i];data[i]=mask[i]?input[i]:0.;
  }
  const double area=double(bw)*bw-double(ew)*ew;
  for(int iteration=0;iteration<=iterations;++iteration) {
    Vec selected(count);
    for(size_t i=0;i<count;++i)selected[i]=retained[i]?data[i]:0.;
    const Vec valid_sum=integral(retained,height,width), data_sum=integral(selected,height,width);
    for(size_t y=0;y<height;++y)for(size_t x=0;x<width;++x) {
      const size_t i=y*width+x;
      counts[i]=box(valid_sum,y,x,height,width,bw)-box(valid_sum,y,x,height,width,ew);
      const double sum=box(data_sum,y,x,height,width,bw)-box(data_sum,y,x,height,width,ew);
      usable[i]=counts[i]>=fraction*area;
      mean[i]=counts[i]>0?std::max(sum/counts[i],0.):0.;
      if(iteration<iterations) {
        const double noise=mean[i]+readout;
        if(usable[i] && data[i]>mean[i]+4.*std::sqrt(noise+noise/std::max(counts[i],1.)))retained[i]=0.;
      }
    }
  }
  for(size_t i=0;i<count;++i) {
    supported[i]=input[i]>mean[i]+support*std::sqrt(mean[i]+readout);
    variance[i]=mask[i]?(mean[i]+readout)/std::max(counts[i],1.):0.;
    if(!mask[i])mean[i]=0.;
  }
  const Vec data_sum=integral(data,height,width), mean_sum=integral(mean,height,width),
            valid_sum=integral(weights,height,width);
  const Vec variance_sum=poisson?Vec():integral(variance,height,width);
  for(int window:scales) {
    for(size_t y=0;y<height;++y)for(size_t x=0;x<width;++x) {
      const size_t i=y*width+x;
      if(!mask[i] || !usable[i] || !supported[i])continue;
      const double valid=box(valid_sum,y,x,height,width,window);
      if(valid<double(window)*window*fraction)continue;
      const double observed=box(data_sum,y,x,height,width,window);
      double expected=box(mean_sum,y,x,height,width,window),score;
      if(poisson) {
        expected=std::max(expected,.5);
        const double excess=std::max(observed,expected);
        score=std::sqrt(std::max(2.*(excess*std::log(excess/expected)-excess+expected),0.));
      } else {
        const double v=expected+readout*valid+double(window)*window*box(variance_sum,y,x,height,width,window);
        score=std::max(observed-expected,0.)/std::sqrt(std::max(v,.5));
      }
      best[i]=std::max(best[i],score);
    }
  }
  for(size_t i=0;i<count;++i)output[i]=mask[i] && usable[i] && supported[i] && best[i]>=threshold;
}

PyObject* spot_proposals(PyObject*,PyObject* args) {
  PyObject *data_obj,*mask_obj,*scales_obj;
  double readout,threshold,support,fraction;
  int poisson,bw,ew,iterations;
  if(!PyArg_ParseTuple(args,"OOdpiiOdddi",&data_obj,&mask_obj,&readout,&poisson,&bw,&ew,&scales_obj,
                       &threshold,&support,&fraction,&iterations))return nullptr;
  PyArrayObject* data=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(data_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  if(!data)return nullptr;
  PyArrayObject* mask=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(mask_obj,NPY_BOOL,NPY_ARRAY_IN_ARRAY));
  if(!mask){Py_DECREF(data);return nullptr;}
  if(PyArray_NDIM(data)!=2 || PyArray_NDIM(mask)!=2 || PyArray_DIM(data,0)!=PyArray_DIM(mask,0)
      || PyArray_DIM(data,1)!=PyArray_DIM(mask,1) || bw<1 || ew<1 || bw<=ew || bw%2!=1 || ew%2!=1
      || iterations<0 || !std::isfinite(readout) || readout<0 || !std::isfinite(threshold) || threshold<=0
      || !std::isfinite(support) || support<0 || !std::isfinite(fraction) || fraction<=0 || fraction>1) {
    Py_DECREF(data);Py_DECREF(mask);PyErr_SetString(PyExc_ValueError,"Invalid native proposal arrays/settings");return nullptr;
  }
  PyObject* sequence=PySequence_Fast(scales_obj,"scales must be a sequence");
  if(!sequence){Py_DECREF(data);Py_DECREF(mask);return nullptr;}
  std::vector<int> scales;
  for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(sequence);++i) {
    const long value=PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence,i));
    if(PyErr_Occurred() || value<1 || value>ew || value%2!=1) {
      Py_DECREF(data);Py_DECREF(mask);Py_DECREF(sequence);
      if(!PyErr_Occurred())PyErr_SetString(PyExc_ValueError,"Invalid native proposal scale");
      return nullptr;
    }
    scales.push_back(static_cast<int>(value));
  }
  Py_DECREF(sequence);
  if(scales.empty()){Py_DECREF(data);Py_DECREF(mask);PyErr_SetString(PyExc_ValueError,"Empty proposal scales");return nullptr;}
  npy_intp shape[2]={PyArray_DIM(data,0),PyArray_DIM(data,1)};
  PyArrayObject* output=reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2,shape,NPY_BOOL));
  if(!output){Py_DECREF(data);Py_DECREF(mask);return nullptr;}
  std::exception_ptr failure;
  Py_BEGIN_ALLOW_THREADS
  try {
    proposals(static_cast<const double*>(PyArray_DATA(data)),static_cast<const npy_bool*>(PyArray_DATA(mask)),
              static_cast<npy_bool*>(PyArray_DATA(output)),shape[0],shape[1],readout,poisson,bw,ew,
              scales,threshold,support,fraction,iterations);
  } catch(...) {failure=std::current_exception();}
  Py_END_ALLOW_THREADS
  Py_DECREF(data);Py_DECREF(mask);
  if(failure) {
    Py_DECREF(output);
    try{std::rethrow_exception(failure);}catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& error){PyErr_SetString(PyExc_RuntimeError,error.what());return nullptr;}
  }
  return reinterpret_cast<PyObject*>(output);
}

PyMethodDef methods[]={{"spot_proposals",spot_proposals,METH_VARARGS,"Count-excess proposals using fused summed-area statistics."},
                       {nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_native",nullptr,-1,methods};
}

PyMODINIT_FUNC PyInit__native(){import_array();return PyModule_Create(&module);}
