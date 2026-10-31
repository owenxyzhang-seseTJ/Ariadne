// Ariadne native standardized/count-excess proposals. Rules match spots.py.
// No thresholds, noise calibration or measured pixel values are fitted here.
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <vector>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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

int component_root(std::vector<int>& parent, int value) {
  while(parent[value]!=value) {
    parent[value]=parent[parent[value]];
    value=parent[value];
  }
  return value;
}

void discard_small_components(npy_bool* output,size_t height,size_t width,int minimum) {
  if(minimum<=1)return;
  const size_t count=height*width;
  std::vector<int> parent(count,-1),sizes(count,0);
  for(size_t i=0;i<count;++i)if(output[i])parent[i]=static_cast<int>(i);
  for(size_t y=0;y<height;++y)for(size_t x=0;x<width;++x) {
    const size_t index=y*width+x;
    if(!output[index])continue;
    if(x && output[index-1]) {
      const int a=component_root(parent,static_cast<int>(index));
      const int b=component_root(parent,static_cast<int>(index-1));
      if(a!=b)parent[a]=b;
    }
    if(y && output[index-width]) {
      const int a=component_root(parent,static_cast<int>(index));
      const int b=component_root(parent,static_cast<int>(index-width));
      if(a!=b)parent[a]=b;
    }
  }
  for(size_t i=0;i<count;++i)if(output[i])++sizes[component_root(parent,static_cast<int>(i))];
  for(size_t i=0;i<count;++i)if(output[i]
      && sizes[component_root(parent,static_cast<int>(i))]<minimum)output[i]=0;
}

void proposals(const double* input, const npy_bool* mask, npy_bool* output,
               size_t height,size_t width,double readout,int poisson,int bw,int ew,
               const std::vector<int>& scales,double threshold,double support,double fraction,
               int iterations,int standardized,int component_minimum) {
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
  const Vec variance_sum=
      (standardized || !poisson)?integral(variance,height,width):Vec();
  std::vector<size_t> candidates;
  candidates.reserve(count/100);
  for(size_t i=0;i<count;++i)
    if(mask[i] && usable[i] && supported[i])candidates.push_back(i);
  for(int window:scales) {
    for(size_t i:candidates) {
      const size_t y=i/width,x=i%width;
      const double valid=box(valid_sum,y,x,height,width,window);
      if(valid<double(window)*window*fraction)continue;
      const double observed=box(data_sum,y,x,height,width,window);
      double expected=box(mean_sum,y,x,height,width,window),score;
      if(standardized) {
        const double v=expected+readout*valid
            +double(window)*window*box(variance_sum,y,x,height,width,window);
        score=std::max(observed-expected,0.)/std::sqrt(std::max(v,.5));
      } else if(poisson) {
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
  std::fill(output,output+count,0);
  for(size_t i:candidates)output[i]=best[i]>=threshold;
  discard_small_components(output,height,width,component_minimum);
}

PyObject* spot_proposals(PyObject*,PyObject* args) {
  PyObject *data_obj,*mask_obj,*scales_obj;
  double readout,threshold,support,fraction;
  int poisson,bw,ew,iterations,standardized,component_minimum;
  if(!PyArg_ParseTuple(args,"OOdpiiOdddiii",&data_obj,&mask_obj,&readout,&poisson,&bw,&ew,&scales_obj,
                       &threshold,&support,&fraction,&iterations,&standardized,&component_minimum))return nullptr;
  PyArrayObject* data=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(data_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  if(!data)return nullptr;
  PyArrayObject* mask=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(mask_obj,NPY_BOOL,NPY_ARRAY_IN_ARRAY));
  if(!mask){Py_DECREF(data);return nullptr;}
  if(PyArray_NDIM(data)!=2 || PyArray_NDIM(mask)!=2 || PyArray_DIM(data,0)!=PyArray_DIM(mask,0)
      || PyArray_DIM(data,1)!=PyArray_DIM(mask,1) || bw<1 || ew<1 || bw<=ew || bw%2!=1 || ew%2!=1
      || iterations<0 || !std::isfinite(readout) || readout<0 || !std::isfinite(threshold) || threshold<=0
      || !std::isfinite(support) || support<0 || !std::isfinite(fraction) || fraction<=0 || fraction>1
      || (standardized!=0 && standardized!=1) || component_minimum<1) {
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
              scales,threshold,support,fraction,iterations,standardized,component_minimum);
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

inline void solve_lower(const std::vector<double>& lower, const std::vector<double>& rhs,
                        int size, std::vector<double>& result) {
  for(int i=0;i<size;++i) {
    double value=rhs[i];
    for(int j=0;j<i;++j)value-=lower[i*size+j]*result[j];
    result[i]=value/lower[i*size+i];
  }
}

inline void solve_upper(const std::vector<double>& upper, const std::vector<double>& rhs,
                        int size, std::vector<double>& result) {
  for(int i=size-1;i>=0;--i) {
    double value=rhs[i];
    for(int j=i+1;j<size;++j)value-=upper[i*size+j]*result[j];
    result[i]=value/upper[i*size+i];
  }
}

void symmetric_eigenvalues(std::vector<double> matrix, int size, std::vector<double>& values) {
  for(int iteration=0;iteration<64*size*size;++iteration) {
    int p=0,q=1;double largest=0;
    for(int i=0;i<size;++i)for(int j=i+1;j<size;++j) {
      const double value=std::abs(matrix[i*size+j]);
      if(value>largest){largest=value;p=i;q=j;}
    }
    double scale=0;
    for(int i=0;i<size;++i)scale=std::max(scale,std::abs(matrix[i*size+i]));
    if(largest<=1e-14*std::max(scale,1e-300))break;
    const double app=matrix[p*size+p],aqq=matrix[q*size+q],apq=matrix[p*size+q];
    const double angle=.5*std::atan2(2*apq,aqq-app);
    const double c=std::cos(angle),s=std::sin(angle);
    for(int i=0;i<size;++i) {
      if(i==p || i==q)continue;
      const double aip=matrix[i*size+p],aiq=matrix[i*size+q];
      const double nip=c*aip-s*aiq,niq=s*aip+c*aiq;
      matrix[i*size+p]=matrix[p*size+i]=nip;
      matrix[i*size+q]=matrix[q*size+i]=niq;
    }
    matrix[p*size+p]=c*c*app-2*s*c*apq+s*s*aqq;
    matrix[q*size+q]=s*s*app+2*s*c*apq+c*c*aqq;
    matrix[p*size+q]=matrix[q*size+p]=0.;
  }
  values.resize(size);
  for(int i=0;i<size;++i)values[i]=matrix[i*size+i];
  std::sort(values.begin(),values.end(),std::greater<double>());
}

PyObject* array_from_vector(const std::vector<double>& values,npy_intp rows,npy_intp columns) {
  npy_intp shape[2]={rows,columns};
  PyObject* result=columns==0?PyArray_SimpleNew(1,shape,NPY_DOUBLE)
                              :PyArray_SimpleNew(2,shape,NPY_DOUBLE);
  if(!result)return nullptr;
  if(!values.empty())std::copy(values.begin(),values.end(),
      static_cast<double*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(result))));
  return result;
}

PyObject* known_background_wls(PyObject*,PyObject* args) {
  PyObject *data_obj,*variance_obj,*profiles_obj;
  double rcond;
  if(!PyArg_ParseTuple(args,"OOOd",&data_obj,&variance_obj,&profiles_obj,&rcond))return nullptr;
  PyArrayObject* data=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
      data_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  PyArrayObject* variance=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
      variance_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  PyArrayObject* profiles=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
      profiles_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  if(!data || !variance || !profiles) {
    Py_XDECREF(data);Py_XDECREF(variance);Py_XDECREF(profiles);return nullptr;
  }
  const int rows=PyArray_DIM(profiles,0),components=PyArray_DIM(profiles,1);
  if(PyArray_NDIM(data)!=1 || PyArray_NDIM(variance)!=1 || PyArray_NDIM(profiles)!=2
      || PyArray_DIM(data,0)!=rows || PyArray_DIM(variance,0)!=rows
      || components<1 || components>32 || rows<=components+2
      || !std::isfinite(rcond) || rcond<=0 || rcond>=1) {
    Py_DECREF(data);Py_DECREF(variance);Py_DECREF(profiles);
    PyErr_SetString(PyExc_ValueError,"Invalid known-background fit dimensions");return nullptr;
  }
  const double* d=static_cast<const double*>(PyArray_DATA(data));
  const double* v=static_cast<const double*>(PyArray_DATA(variance));
  const double* p=static_cast<const double*>(PyArray_DATA(profiles));
  std::vector<double> scales(components),gram(components*components,0.),
      rhs(components,0.),target_norm=std::vector<double>(1,0.);
  std::vector<double> eigenvalues,lower(components*components,0.),
      upper(components*components,0.),modes_y(components),coefficients(components),
      intensities(components),covariance(components*components),singular(components);
  bool failure=false,rank_fallback=false;
  Py_BEGIN_ALLOW_THREADS
  for(int row=0;row<rows;++row) {
    if(!std::isfinite(d[row]) || !std::isfinite(v[row]) || v[row]<=0) {
      failure=true;break;
    }
    for(int column=0;column<components;++column) {
      const double value=p[row*components+column];
      if(!std::isfinite(value)){failure=true;break;}
      const double weighted=value/std::sqrt(v[row]);
      scales[column]+=weighted*weighted;
    }
    if(failure)break;
    target_norm[0]+=d[row]*d[row]/v[row];
  }
  if(!failure)for(int column=0;column<components;++column) {
    scales[column]=std::sqrt(scales[column]);
    if(scales[column]<=1e-15)failure=true;
  }
  if(!failure)for(int row=0;row<rows;++row) {
    const double weighted_target=d[row]/std::sqrt(v[row]);
    for(int i=0;i<components;++i) {
      const double xi=p[row*components+i]/(std::sqrt(v[row])*scales[i]);
      rhs[i]+=xi*weighted_target;
      for(int j=0;j<=i;++j)gram[i*components+j]+=xi*
          (p[row*components+j]/(std::sqrt(v[row])*scales[j]));
    }
  }
  if(!failure)for(int i=0;i<components;++i)for(int j=0;j<i;++j)
    gram[j*components+i]=gram[i*components+j];
  if(!failure)symmetric_eigenvalues(gram,components,eigenvalues);
  if(failure || eigenvalues.empty() || eigenvalues[0]<=0
      || eigenvalues.back()<=eigenvalues[0]*rcond*rcond) {
    rank_fallback=true;
    failure=false;
  }
  if(!rank_fallback)for(int i=0;i<components;++i) {
    for(int j=0;j<=i;++j) {
      double value=gram[i*components+j];
      for(int k=0;k<j;++k)value-=lower[i*components+k]*lower[j*components+k];
      if(i==j) {
        if(value<=1e-15)failure=true;
        else lower[i*components+i]=std::sqrt(value);
      } else lower[i*components+j]=value/lower[j*components+j];
    }
  }
  if(!rank_fallback && !failure) {
    for(int i=0;i<components;++i)for(int j=0;j<components;++j)
      upper[i*components+j]=lower[j*components+i];
    solve_lower(lower,rhs,components,modes_y);
    solve_upper(upper,modes_y,components,coefficients);
    for(int column=0;column<components;++column) {
      singular[column]=std::sqrt(std::max(eigenvalues[column],0.));
      intensities[column]=coefficients[column]/scales[column];
    }
    for(int column=0;column<components;++column) {
      std::vector<double> unit(components,0.),forward(components),solved(components);
      unit[column]=1.;
      solve_lower(lower,unit,components,forward);
      solve_upper(upper,forward,components,solved);
      for(int row=0;row<components;++row)
        covariance[row*components+column]=solved[row]/scales[row]/scales[column];
    }
  }
  Py_END_ALLOW_THREADS
  Py_DECREF(data);Py_DECREF(variance);Py_DECREF(profiles);
  if(rank_fallback)Py_RETURN_NONE;
  if(failure)return PyErr_NoMemory();
  std::vector<double> fitted(rows),data_sum(rows,0.);
  for(int row=0;row<rows;++row)for(int column=0;column<components;++column)
    fitted[row]+=p[row*components+column]*intensities[column];
  double chi2=0;
  for(int row=0;row<rows;++row)chi2+=(d[row]-fitted[row])*(d[row]-fitted[row])/v[row];
  double modes_norm=0;for(double value:modes_y)modes_norm+=value*value;
  const double constant=std::max(target_norm[0]-modes_norm,0.);
  PyObject* result=PyTuple_New(9);
  if(!result)return nullptr;
  PyTuple_SET_ITEM(result,0,array_from_vector(intensities,components,0));
  PyTuple_SET_ITEM(result,1,array_from_vector(covariance,components,components));
  PyTuple_SET_ITEM(result,2,array_from_vector(fitted,rows,0));
  PyTuple_SET_ITEM(result,3,PyFloat_FromDouble(chi2));
  PyTuple_SET_ITEM(result,4,array_from_vector(singular,components,0));
  PyTuple_SET_ITEM(result,5,array_from_vector(upper,components,components));
  PyTuple_SET_ITEM(result,6,array_from_vector(modes_y,components,0));
  PyTuple_SET_ITEM(result,7,array_from_vector(scales,components,0));
  PyTuple_SET_ITEM(result,8,PyFloat_FromDouble(constant));
  return result;
}

PyObject* known_background_irls(PyObject*,PyObject* args) {
  PyObject *data_obj,*background_obj,*profiles_obj,*valid_obj;
  double rcond,tolerance,minimum_weight;
  int max_iterations;
  if(!PyArg_ParseTuple(args,"OOOOddid",&data_obj,&background_obj,&profiles_obj,&valid_obj,
                       &rcond,&tolerance,&max_iterations,&minimum_weight))return nullptr;
  PyArrayObject* data=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
      data_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  PyArrayObject* background=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
      background_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  PyArrayObject* profiles=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
      profiles_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  PyArrayObject* valid=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
      valid_obj,NPY_BOOL,NPY_ARRAY_IN_ARRAY));
  if(!data || !background || !profiles || !valid) {
    Py_XDECREF(data);Py_XDECREF(background);Py_XDECREF(profiles);Py_XDECREF(valid);
    return nullptr;
  }
  const int rows=PyArray_DIM(profiles,0),components=PyArray_DIM(profiles,1);
  if(PyArray_NDIM(data)!=1 || PyArray_NDIM(background)!=1 || PyArray_NDIM(valid)!=1
      || PyArray_NDIM(profiles)!=2 || PyArray_DIM(data,0)!=rows
      || PyArray_DIM(background,0)!=rows || PyArray_DIM(valid,0)!=rows
      || components<1 || components>32 || !std::isfinite(rcond) || rcond<=0 || rcond>=1
      || !std::isfinite(tolerance) || tolerance<=0 || max_iterations<1
      || !std::isfinite(minimum_weight) || minimum_weight<=0) {
    Py_DECREF(data);Py_DECREF(background);Py_DECREF(profiles);Py_DECREF(valid);
    PyErr_SetString(PyExc_ValueError,"Invalid known-background IRLS dimensions/settings");
    return nullptr;
  }
  const double* d=static_cast<const double*>(PyArray_DATA(data));
  const double* bg=static_cast<const double*>(PyArray_DATA(background));
  const double* p=static_cast<const double*>(PyArray_DATA(profiles));
  const npy_bool* mask=static_cast<const npy_bool*>(PyArray_DATA(valid));
  std::vector<int> indices;
  for(int row=0;row<rows;++row)if(mask[row])indices.push_back(row);
  const int pixels=indices.size();
  if(pixels<=components+2) {
    Py_DECREF(data);Py_DECREF(background);Py_DECREF(profiles);Py_DECREF(valid);
    Py_RETURN_NONE;
  }
  std::vector<double> current(components,1.),next(components),variance(pixels),
      scales(components),gram(components*components),rhs(components),
      modes_y(components),coefficients(components),lower(components*components),
      upper(components*components),covariance(components*components),
      singular(components),eigenvalues;
  bool failure=false,rank_fallback=false,converged=false;
  int iterations=0;
  double constant=0.;
  Py_BEGIN_ALLOW_THREADS
  for(iterations=0;iterations<max_iterations;++iterations) {
    std::fill(scales.begin(),scales.end(),0.);
    std::fill(gram.begin(),gram.end(),0.);
    std::fill(rhs.begin(),rhs.end(),0.);
    for(int row=0;row<pixels;++row) {
      const int source=indices[row];
      if(!std::isfinite(d[source]) || !std::isfinite(bg[source]) || bg[source]<0
          || !std::isfinite(p[source*components])) {
        failure=true;break;
      }
      double predicted_background=bg[source];
      for(int column=0;column<components;++column)
        predicted_background+=p[source*components+column]
            *std::max(minimum_weight,std::abs(current[column]));
      if(!std::isfinite(predicted_background) || predicted_background<=0) {
        failure=true;break;
      }
      variance[row]=predicted_background;
      const double inverse_sqrt=1./std::sqrt(predicted_background);
      for(int column=0;column<components;++column) {
        const double value=p[source*components+column];
        if(!std::isfinite(value)) {failure=true;break;}
        const double weighted=value*inverse_sqrt;
        scales[column]+=weighted*weighted;
      }
      if(failure)break;
    }
    if(failure)break;
    for(int column=0;column<components;++column) {
      scales[column]=std::sqrt(scales[column]);
      if(scales[column]<=1e-15)failure=true;
    }
    if(failure)break;
    for(int row=0;row<pixels;++row) {
      const int source=indices[row];
      const double inverse_sqrt=1./std::sqrt(variance[row]);
      const double signal=(d[source]-bg[source])*inverse_sqrt;
      for(int i=0;i<components;++i) {
        const double xi=p[source*components+i]*inverse_sqrt/scales[i];
        rhs[i]+=xi*signal;
        for(int j=0;j<=i;++j)
          gram[i*components+j]+=xi*
              (p[source*components+j]*inverse_sqrt/scales[j]);
      }
    }
    for(int i=0;i<components;++i)for(int j=0;j<i;++j)
      gram[j*components+i]=gram[i*components+j];
    symmetric_eigenvalues(gram,components,eigenvalues);
    if(eigenvalues.empty() || eigenvalues[0]<=0
        || eigenvalues.back()<=eigenvalues[0]*rcond*rcond) {
      rank_fallback=true;break;
    }
    std::fill(lower.begin(),lower.end(),0.);
    for(int i=0;i<components;++i)for(int j=0;j<=i;++j) {
      double value=gram[i*components+j];
      for(int k=0;k<j;++k)value-=lower[i*components+k]*lower[j*components+k];
      if(i==j)lower[i*components+i]=std::sqrt(value);
      else lower[i*components+j]=value/lower[j*components+j];
    }
    for(int i=0;i<components;++i)for(int j=0;j<components;++j)
      upper[i*components+j]=lower[j*components+i];
    solve_lower(lower,rhs,components,modes_y);
    solve_upper(upper,modes_y,components,coefficients);
    double change=0.;
    for(int column=0;column<components;++column) {
      next[column]=coefficients[column]/scales[column];
      const double difference=next[column]-current[column];
      change+=difference*difference;
    }
    current.swap(next);
    if(change<tolerance) {converged=true;break;}
  }
  if(!failure && !rank_fallback && converged) {
    for(int column=0;column<components;++column)
      singular[column]=std::sqrt(std::max(eigenvalues[column],0.));
    for(int column=0;column<components;++column) {
      std::vector<double> unit(components,0.),forward(components),solved(components);
      unit[column]=1.;
      solve_lower(lower,unit,components,forward);
      solve_upper(upper,forward,components,solved);
      for(int row=0;row<components;++row)
        covariance[row*components+column]=solved[row]/scales[row]/scales[column];
    }
  }
  double modes_norm=0;for(double value:modes_y)modes_norm+=value*value;
  double final_target_norm=0.;
  for(int row=0;row<pixels;++row) {
    const int source=indices[row];
    const double residual=(d[source]-bg[source])/std::sqrt(variance[row]);
    final_target_norm+=residual*residual;
  }
  constant=std::max(final_target_norm-modes_norm,0.);
  Py_END_ALLOW_THREADS
  Py_DECREF(data);Py_DECREF(background);Py_DECREF(profiles);Py_DECREF(valid);
  if(rank_fallback)Py_RETURN_NONE;
  if(failure || !converged) {
    PyErr_SetString(PyExc_RuntimeError,"Native known-background IRLS did not converge");
    return nullptr;
  }
  PyObject* result=PyTuple_New(9);
  if(!result)return nullptr;
  PyTuple_SET_ITEM(result,0,array_from_vector(current,components,0));
  PyTuple_SET_ITEM(result,1,array_from_vector(covariance,components,components));
  PyTuple_SET_ITEM(result,2,array_from_vector(singular,components,0));
  PyTuple_SET_ITEM(result,3,array_from_vector(upper,components,components));
  PyTuple_SET_ITEM(result,4,array_from_vector(modes_y,components,0));
  PyTuple_SET_ITEM(result,5,array_from_vector(scales,components,0));
  PyTuple_SET_ITEM(result,6,PyFloat_FromDouble(constant));
  PyTuple_SET_ITEM(result,7,PyLong_FromLong(iterations+1));
  PyTuple_SET_ITEM(result,8,PyLong_FromLong(pixels));
  return result;
}

// ---- 自校准 matched-filter 内核，与 spots.propose_matched 一致 ----
}
#include <chrono>
#include <cstdlib>
#include <atomic>
#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif
namespace {
// 每个调用最多四个工作单元；并发调用共享三个附加工作单元预算。
std::atomic<int> matched_extra_workers{0};
thread_local int matched_threads=1;
struct MatchedThreadBudget {
  int extra=0,previous=matched_threads;
  explicit MatchedThreadBudget(size_t pixels) {
    int requested=4;
    if(const char* value=std::getenv("ARIADNE_MATCHED_THREADS"))requested=std::max(1,std::min(4,std::atoi(value)));
    if(pixels<262144)requested=1;
#if !defined(__APPLE__)
    requested=1;
#endif
    int used=matched_extra_workers.load();
    while(requested>1 && used<3) {
      extra=std::min(requested-1,3-used);
      if(matched_extra_workers.compare_exchange_weak(used,used+extra))break;
      extra=0;
    }
    matched_threads=1+extra;
  }
  ~MatchedThreadBudget(){matched_threads=previous;if(extra)matched_extra_workers.fetch_sub(extra);}
};
template<class F> void matched_parallel(size_t count,F&& fn) {
#if defined(__APPLE__)
  const size_t parts=std::min(count,size_t(matched_threads));
  if(parts>1) {
    struct Context {F* fn;size_t count,parts;};Context ctx{&fn,count,parts};
    dispatch_apply_f(parts,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&ctx,[](void* raw,size_t j){
      auto& c=*static_cast<Context*>(raw);(*c.fn)(c.count*j/c.parts,c.count*(j+1)/c.parts);
    });return;
  }
#endif
  fn(0,count);
}
struct MatchedSettings {
  double gain,offset,readout,clip,false_alarms,t_peak,t_low,contrast,zinger_fraction,zinger_sigma;
  int tile,rounds,min_bin,calibrate,local_cells,component_minimum;
};
constexpr int kScales=2;
const double kH1[5]={1./16,4./16,6./16,4./16,1./16};
const double kH2[9]={4./256,16./256,32./256,48./256,56./256,48./256,32./256,16./256,4./256};
const int noise_lags[12][2]={{0,1},{1,0},{1,1},{1,-1},{0,2},{2,0},{1,2},{2,1},{1,-2},{2,-1},{2,2},{2,-2}};
Vec matched_kernel(int scale) {return scale?Vec(kH2,kH2+9):Vec(kH1,kH1+5);}

struct MatchedWorkspace {
  size_t height=0,width=0;
  std::vector<unsigned char> cached_valid,valid,keep,boundary,needed,good;
  std::vector<float> area[kScales],square[kScales];
  Vec x,b,inv_neff,phi,residual,z0,z1,bg_xs,bg_ns;
  std::vector<int> bg_bins;
  Vec local_xs,local_ns,local_es,local_c,local_phi,local_keep,local_sx,local_se,local_u,local_kn,local_nk,local_su,window_scratch;
  std::vector<int> label,parent;
};
thread_local MatchedWorkspace matched_ws;

// A/Q 的核系数都是二进制精确数；稀疏变化可直接局部加减，不累计舍入漂移。
void matched_mask_terms(MatchedWorkspace& w,size_t H,size_t W) {
  const size_t n=H*W;
  std::vector<size_t> changed;
  const bool same=w.height==H && w.width==W && w.cached_valid.size()==n;
  if(same)for(size_t i=0;i<n;++i)if(w.valid[i]!=w.cached_valid[i])changed.push_back(i);
  if(same && changed.empty())return;
  if(same && changed.size()<n/128) {
    for(int k=0;k<2;++k) {
      const Vec h=matched_kernel(k);const long r=h.size()/2;
      for(size_t p:changed) {
        const long y=p/W,x=p%W;const double d=double(w.valid[p])-w.cached_valid[p];
        for(long dy=-r;dy<=r;++dy)for(long dx=-r;dx<=r;++dx) {
          const long yy=y+dy,xx=x+dx;if(yy<0 || xx<0 || yy>=long(H) || xx>=long(W))continue;
          const size_t q=yy*W+xx;const double a=h[dy+r]*h[dx+r];
          w.area[k][q]+=d*a;w.square[k][q]+=d*a*a;
        }
      }
    }
  } else {
    // 先按列再按行，与 NumPy reference 的分离滤波顺序一致。
    Vec scratch(n);
    for(int k=0;k<2;++k) {
      const Vec h=matched_kernel(k);const long r=h.size()/2;
      w.area[k].resize(n);w.square[k].resize(n);
      for(int power=1;power<=2;++power) {
        for(size_t y=0;y<H;++y)for(size_t x=0;x<W;++x) {
          double a=0;
          for(long d=-r;d<=r;++d)if(long(y)+d>=0 && long(y)+d<long(H)) {
            const double v=h[d+r];a+=(power==1?v:v*v)*w.valid[(long(y)+d)*W+x];
          }
          scratch[y*W+x]=a;
        }
        auto& out=power==1?w.area[k]:w.square[k];
        for(size_t y=0;y<H;++y)for(size_t x=0;x<W;++x) {
          double a=0;
          for(long d=-r;d<=r;++d)if(long(x)+d>=0 && long(x)+d<long(W)) {
            const double v=h[d+r];a+=(power==1?v:v*v)*scratch[y*W+long(x)+d];
          }
          out[y*W+x]=a;
        }
      }
    }
  }
  w.height=H;w.width=W;w.cached_valid=w.valid;
}

void fill_bins(Vec& value,const std::vector<unsigned char>& good) {
  const int n=value.size();int previous=-1;
  for(int i=0;i<n;++i)if(good[i]) {
    if(previous<0)for(int j=0;j<i;++j)value[j]=value[i];
    else for(int j=previous+1;j<i;++j)
      value[j]=value[previous]+(value[i]-value[previous])*double(j-previous)/double(i-previous);
    previous=i;
  }
  for(int j=previous+1;j<n;++j)value[j]=value[previous];
}
bool fill_tiles(const Vec& sn,const Vec& sd,const Vec& cnt,size_t ny,size_t nx,double need,Vec& m,Vec& c0) {
  const size_t nt=ny*nx;m.assign(nt,0.);c0.assign(nt,0.);
  std::vector<unsigned char> filled(nt,0);size_t remaining=nt;
  for(size_t t=0;t<nt;++t)if(cnt[t]>=need && sd[t]>0) {
    m[t]=sn[t]/sd[t];c0[t]=cnt[t];filled[t]=1;--remaining;
  }
  if(remaining==nt)return false;
  while(remaining) {
    auto next=filled;Vec nm=m,nc=c0;
    for(size_t y=0;y<ny;++y)for(size_t x=0;x<nx;++x) {
      const size_t p=y*nx+x;if(filled[p])continue;
      double a=0,b=0,c=0;
      for(long dy=-1;dy<=1;++dy)for(long dx=-1;dx<=1;++dx) {
        const long yy=long(y)+dy,xx=long(x)+dx;
        if(yy<0 || xx<0 || yy>=long(ny) || xx>=long(nx))continue;
        const size_t q=yy*nx+xx;if(filled[q]){a+=m[q];b+=1;c+=c0[q];}
      }
      if(b>0){nm[p]=a/b;nc[p]=c/b;next[p]=1;--remaining;}
    }
    filled.swap(next);m.swap(nm);c0.swap(nc);
  }
  for(double& v:c0)v=std::max(v,1.);
  return true;
}
struct AxisWeight {size_t i0,i1;double f;};
AxisWeight axis_weight(double u,int T,size_t n) {
  double c=std::min(std::max((u+.5)/T-.5,0.),double(n-1));
  const size_t a=std::min(size_t(std::floor(c)),n>=2?n-2:0);
  return {a,std::min(a+1,n-1),c-double(a)};
}
void interp_rows(const Vec& grid,size_t ny,size_t nx,const std::vector<AxisWeight>& wx,Vec& rows) {
  const size_t n=wx.size();rows.resize(ny*n);
  for(size_t y=0;y<ny;++y)for(size_t j=0;j<n;++j) {
    const auto& a=wx[j];rows[y*n+j]=grid[y*nx+a.i0]*(1-a.f)+grid[y*nx+a.i1]*a.f;
  }
}
struct MatchedBackground {Vec S,e,P,cnt,m,c0;std::vector<unsigned char> keep;size_t ny,nx;};
bool matched_background(MatchedWorkspace& w,const int* bins,int nb,size_t H,size_t W,
                        const MatchedSettings& s,double r,MatchedBackground& state,bool warm) {
  const size_t ch=(H+1)/2,cw=(W+1)/2,n=ch*cw,tc=s.tile/2;
  const size_t ny=(ch+tc-1)/tc,nx=(cw+tc-1)/tc,nt=ny*nx;
  Vec& xs=w.bg_xs;Vec& ns=w.bg_ns;std::vector<int>& cb=w.bg_bins;
  xs.resize(n);if(!warm){ns.resize(n);cb.resize(n);}
  matched_parallel(ch,[&](size_t first,size_t last){
    for(size_t cy=first;cy<last;++cy)for(size_t cx=0;cx<cw;++cx) {
      const size_t y=2*cy,x=2*cx,i=y*W+x,j=cy*cw+cx;
      double sum=w.x[i]+(x+1<W?w.x[i+1]:0.);
      if(y+1<H)sum+=w.x[i+W]+(x+1<W?w.x[i+W+1]:0.);xs[j]=sum;
      if(!warm) {
        ns[j]=w.valid[i]+(x+1<W?w.valid[i+1]:0)+(y+1<H?w.valid[i+W]:0)+(y+1<H && x+1<W?w.valid[i+W+1]:0);
        cb[j]=bins?bins[i]:0;
      }
    }
  });
  Vec P(nb),cnt(nb),num(nb),den(nb),m,c0,partial(12*size_t(nb));
  std::vector<unsigned char> keep(n),good(nb);
  auto profile=[&]() {
    std::fill(num.begin(),num.end(),0.);std::fill(den.begin(),den.end(),0.);std::fill(cnt.begin(),cnt.end(),0.);
    std::fill(partial.begin(),partial.end(),0.);
    for(size_t i=0;i<n;++i)if(keep[i]) {
      double* h=partial.data()+3*((i%4)*size_t(nb)+cb[i]);h[0]+=xs[i];h[1]+=ns[i]*state.S[i];h[2]+=ns[i];
    }
    for(int bin=0;bin<nb;++bin)for(int lane=0;lane<4;++lane) {
      const double* h=partial.data()+3*(size_t(lane)*nb+bin);num[bin]+=h[0];den[bin]+=h[1];cnt[bin]+=h[2];
    }
    const double need=std::min(double(s.min_bin),std::max(1.,*std::max_element(cnt.begin(),cnt.end())));
    bool any=false;
    for(int k=0;k<nb;++k){good[k]=cnt[k]>=need && den[k]>0;P[k]=good[k]?num[k]/den[k]:0.;any|=good[k];}
    if(!any)return false;fill_bins(P,good);fill_bins(cnt,good);return true;
  };
  auto clipped=[&](size_t i,double e) {
    e=std::max(e,0.);const double d=xs[i]-e-1.;
    return ns[i]>0 && (d<=0 || d*d<=s.clip*s.clip*(e+ns[i]*r));
  };
  std::vector<AxisWeight> wy(ch),wx(cw);
  for(size_t y=0;y<ch;++y)wy[y]=axis_weight(2.*y+.5,s.tile,ny);
  for(size_t x=0;x<cw;++x)wx[x]=axis_weight(2.*x+.5,s.tile,nx);
  auto interpolate_S=[&](bool clip_after) {
    Vec rows;interp_rows(m,ny,nx,wx,rows);
    matched_parallel(ch,[&](size_t first,size_t last){
      for(size_t y=first;y<last;++y) {
        const auto& a=wy[y];double* __restrict dst=state.S.data()+y*cw;
        const double* __restrict u=rows.data()+a.i0*cw;const double* __restrict v=rows.data()+a.i1*cw;
        for(size_t x=0;x<cw;++x) {
          dst[x]=u[x]*(1-a.f)+v[x]*a.f;
          if(clip_after){const size_t i=y*cw+x;keep[i]=clipped(i,ns[i]*P[cb[i]]*dst[x]);}
        }
      }
    });
  };
  if(!warm) {
    state.S.assign(n,1.);for(size_t i=0;i<n;++i)keep[i]=ns[i]>0;
    if(!profile())return false;
    Vec med(nt,0.),ones(nt,1.),enough(nt,0.);
    matched_parallel(ny,[&](size_t first,size_t last){
      Vec values;values.reserve(tc*tc);
      for(size_t ty=first;ty<last;++ty)for(size_t tx=0;tx<nx;++tx) {
        values.clear();
        for(size_t y=ty*tc;y<std::min(ch,(ty+1)*tc);++y)for(size_t x=tx*tc;x<std::min(cw,(tx+1)*tc);++x) {
          const size_t i=y*cw+x;if(ns[i]>0 && P[cb[i]]>0) {
            const double v=xs[i]/(ns[i]*P[cb[i]]);if(std::isfinite(v))values.push_back(v);
          }
        }
        if(values.size()>=.25*tc*tc) {
          const size_t mid=values.size()/2,t=ty*nx+tx;std::nth_element(values.begin(),values.begin()+mid,values.end());
          med[t]=values[mid];if(values.size()%2==0)med[t]=.5*(med[t]+*std::max_element(values.begin(),values.begin()+mid));enough[t]=1.;
        }
      }
    });
    if(fill_tiles(med,ones,enough,ny,nx,1.,m,c0))interpolate_S(true);
    else matched_parallel(n,[&](size_t first,size_t last){for(size_t i=first;i<last;++i)keep[i]=clipped(i,ns[i]*P[cb[i]]*state.S[i]);});
  } else {
    matched_parallel(n,[&](size_t first,size_t last){for(size_t i=first;i<last;++i)keep[i]=clipped(i,ns[i]*state.e[i]);});
  }
  Vec sn(nt),sd(nt),sc(nt);
  for(int round=0;round<s.rounds;++round) {
    if(!profile())return false;
    std::fill(sn.begin(),sn.end(),0.);std::fill(sd.begin(),sd.end(),0.);std::fill(sc.begin(),sc.end(),0.);
    matched_parallel(ny,[&](size_t first,size_t last){
      for(size_t ty=first;ty<last;++ty)for(size_t tx=0;tx<nx;++tx) {
        double sx=0,se=0,count=0;
        for(size_t y=ty*tc;y<std::min(ch,(ty+1)*tc);++y)for(size_t x=tx*tc;x<std::min(cw,(tx+1)*tc);++x) {
          const size_t i=y*cw+x;keep[i]=clipped(i,ns[i]*P[cb[i]]*state.S[i]);
          if(keep[i]){sx+=xs[i];se+=ns[i]*P[cb[i]];count+=ns[i];}
        }
        const size_t t=ty*nx+tx;sn[t]=sx;sd[t]=se;sc[t]=count;
      }
    });
    if(!fill_tiles(sn,sd,sc,ny,nx,.25*s.tile*s.tile,m,c0))return false;
    interpolate_S(true);
  }
  state.e.resize(n);for(size_t i=0;i<n;++i)state.e[i]=P[cb[i]]*state.S[i];
  state.P.swap(P);state.cnt.swap(cnt);state.m.swap(m);state.c0.swap(c0);state.keep.swap(keep);state.ny=ny;state.nx=nx;
  return true;
}
// 拟合只读取固定抽样；最终背景和有效计数推迟到校准完成后一次物化。
void background_pixels(MatchedWorkspace& w,const MatchedBackground& bg,const int* bins,size_t H,size_t W,
                       int tile,bool samples,double o=0.,double alpha=1.) {
  const size_t ny=bg.ny,nx=bg.nx,cw=(W+1)/2;
  std::vector<AxisWeight> wx(W);for(size_t x=0;x<W;++x)wx[x]=axis_weight(x,tile,nx);
  Vec rows,icrows,ic(bg.c0.size());
  interp_rows(bg.m,ny,nx,wx,rows);
  if(!samples){for(size_t t=0;t<ic.size();++t)ic[t]=1./bg.c0[t];interp_rows(ic,ny,nx,wx,icrows);}
  w.b.resize(H*W);w.keep.resize(H*W);if(!samples)w.inv_neff.resize(H*W);
  const size_t ystep=samples?8:1,xstep=samples?2:1;
  matched_parallel((H+ystep-1)/ystep,[&](size_t first,size_t last){
  for(size_t row=first;row<last;++row) {
    const size_t y=row*ystep;const auto a=axis_weight(y,tile,ny);
    for(size_t x=0;x<W;x+=xstep) {
      const size_t i=y*W+x;const int bin=bins?bins[i]:0;
      w.b[i]=(bg.P[bin]*(rows[a.i0*W+x]*(1-a.f)+rows[a.i1*W+x]*a.f)-o)/alpha;
      if(!samples)w.inv_neff[i]=1./std::max(bg.cnt[bin],1.)+icrows[a.i0*W+x]*(1-a.f)+icrows[a.i1*W+x]*a.f;
      w.keep[i]=bg.keep[(y/2)*cw+x/2] && w.valid[i];
    }
  }});
}

bool noise_fit(const MatchedWorkspace& w,size_t H,size_t W,double r,double& alpha,double& beta) {
  Vec bs,rr;double lo=std::numeric_limits<double>::infinity(),hi=0;
  for(size_t y=0;y<H;y+=8)for(size_t x=0;x<W;x+=2) {
    const size_t i=y*W+x;if(!w.keep[i] || !(w.b[i]>0))continue;
    bs.push_back(w.b[i]);const double d=w.x[i]-w.b[i];rr.push_back(d*d);lo=std::min(lo,w.b[i]);hi=std::max(hi,w.b[i]);
  }
  if(bs.size()<400)return false;
  const double q=std::pow(hi/lo,1./24)*(1+1e-12);double edges[23],edge=lo;
  for(int i=0;i<23;++i){edge*=q;edges[i]=edge;}
  double cnt[24]={},sb[24]={},s2[24];std::vector<int> group(bs.size());
  for(size_t i=0;i<bs.size();++i){int g=std::upper_bound(edges,edges+23,bs[i])-edges;group[i]=g;cnt[g]+=1;sb[g]+=bs[i];}
  for(int g=0;g<24;++g)s2[g]=std::max(sb[g]/std::max(cnt[g],1.)+r,1e-6);
  // 分箱后各箱独立执行完整四轮裁剪，避免逐像素散写的累加依赖。
  size_t offsets[25]={};for(int g=0;g<24;++g)offsets[g+1]=offsets[g]+size_t(cnt[g]);
  size_t next[24];std::copy(offsets,offsets+24,next);Vec grouped(rr.size());
  for(size_t i=0;i<rr.size();++i)grouped[next[group[i]]++]=rr[i];
  matched_parallel(24,[&](size_t first,size_t last){
    for(size_t g=first;g<last;++g)for(int it=0;it<4;++it) {
      double ss=0.,count=0.;size_t i=offsets[g];const size_t end=offsets[g+1];const double limit=16*s2[g];
#if defined(__ARM_NEON)
      float64x2_t sum=vdupq_n_f64(0.),nn=vdupq_n_f64(0.);const float64x2_t threshold=vdupq_n_f64(limit),zero=vdupq_n_f64(0.),one=vdupq_n_f64(1.);
      for(;i+2<=end;i+=2) {
        const float64x2_t v=vld1q_f64(grouped.data()+i);const uint64x2_t ok=vcltq_f64(v,threshold);
        sum=vaddq_f64(sum,vbslq_f64(ok,v,zero));nn=vaddq_f64(nn,vbslq_f64(ok,one,zero));
      }
      ss=vaddvq_f64(sum);count=vaddvq_f64(nn);
#endif
      for(;i<end;++i)if(grouped[i]<limit){ss+=grouped[i];count+=1.;}
      s2[g]=std::max(ss/std::max(count,1.),1e-12);
    }
  });
  int used=0;double s00=0,s01=0,s11=0,t0=0,t1=0;
  for(int g=0;g<24;++g)if(cnt[g]>=200) {
    ++used;const double b=sb[g]/cnt[g],v=s2[g],w2=cnt[g]/(v*v);
    s00+=w2*b*b;s01+=w2*b;s11+=w2;t0+=w2*b*v;t1+=w2*v;
  }
  const double det=s00*s11-s01*s01;if(used<2 || !(det>1e-2*s00*s11))return false;
  alpha=(t0*s11-t1*s01)/det;beta=(s00*t1-s01*t0)/det;return alpha>0;
}
Vec noise_correlation(const MatchedWorkspace& w,size_t H,size_t W,double r) {
  Vec rho(12,0.);const size_t rows=(H+15)/16;
  // 只缓存实际采样的三行，十二个 lag 共享标准残差和裁剪判断。
  Vec u(rows*3*W,0.);std::vector<unsigned char> keep(rows*3*W,0);
  for(size_t row=0;row<rows;++row)for(size_t dy=0;dy<3;++dy) {
    const size_t y=16*row+dy;if(y>=H)continue;
    for(size_t x=0;x<W;++x) {
      const size_t i=y*W+x,j=(row*3+dy)*W+x;
      const double v=w.valid[i]?(w.x[i]-w.b[i])/std::sqrt(std::max(std::max(w.b[i],0.)+r,1e-6)):0.;
      u[j]=v;keep[j]=w.keep[i] && std::fabs(v)<4;
    }
  }
  for(int j=0;j<12;++j) {
    const int dy=noise_lags[j][0],dx=noise_lags[j][1];double aa=0,cc=0,ac=0;size_t count=0;
    if(H<=size_t(dy))continue;
    for(size_t row=0;16*row<H-dy;++row) {
      const double* a=u.data()+row*3*W;const double* c=a+dy*W;
      const unsigned char* m=keep.data()+row*3*W;const unsigned char* n=m+dy*W;
      for(long x=std::max(0,-dx);x<long(W)-std::max(0,dx);++x)if(m[x] && n[x+dx]) {
        ++count;aa+=a[x]*a[x];cc+=c[x+dx]*c[x+dx];ac+=a[x]*c[x+dx];
      }
    }
    if(count>=1000 && aa>0 && cc>0)rho[j]=ac/std::sqrt(aa*cc);
  }
  return rho;
}
struct KernelNoise {double kappa,ly,lx;};
KernelNoise kernel_noise(const Vec& h,const Vec& rho) {
  const int n=h.size();Vec a(n,0.);
  for(int d=0;d<n;++d)for(int j=0;j<n-d;++j)a[d]+=h[j]*h[j+d];
  auto ac=[&](int d){return std::abs(d)<n?a[std::abs(d)]:0.;};
  auto cov=[&](int dy,int dx) {
    double v=ac(dy)*ac(dx);
    for(int j=0;j<12;++j) {
      const int y=noise_lags[j][0],x=noise_lags[j][1];
      v+=rho[j]*ac(dy-y)*ac(dx-x);v+=rho[j]*ac(dy+y)*ac(dx+x);
    }
    return v;
  };
  const double v=cov(0,0);return {v/(a[0]*a[0]),2*(1-cov(1,0)/v),2*(1-cov(0,1)/v)};
}
double rft_threshold(double area,const KernelNoise& k,double target) {
  const double pi=std::acos(-1.),rate=area*std::sqrt(std::max(k.ly*k.lx,0.))/std::pow(2*pi,1.5);
  double lo=1,hi=12;for(int j=0;j<60;++j){const double u=.5*(lo+hi);if(rate*u*std::exp(-u*u/2)>target)lo=u;else hi=u;}
  return .5*(lo+hi);
}
// 双通道窗口共享遍历和边界判断，double SIMD 保留完整精度。
void window_pair(const Vec& a,const Vec& b,Vec& oa,Vec& ob,size_t H,size_t W,int k,Vec& scratch) {
  oa.resize(H*W);ob.resize(H*W);scratch.assign(size_t(matched_threads)*2*W,0.);
  const int threads=matched_threads;
  matched_parallel(size_t(threads),[&](size_t first,size_t last){
    for(size_t part=first;part<last;++part) {
      const size_t y0=H*part/threads,y1=H*(part+1)/threads;
      double* vertical=scratch.data()+part*2*W;
      for(size_t y=y0>size_t(k)?y0-k:0;y<std::min(H,y0+k+1);++y)for(size_t x=0;x<W;++x) {
        vertical[2*x]+=a[y*W+x];vertical[2*x+1]+=b[y*W+x];
      }
      for(size_t y=y0;y<y1;++y) {
        if(y>y0) {
          const double* add_a=y+size_t(k)<H?a.data()+(y+k)*W:nullptr;
          const double* add_b=add_a?b.data()+(y+k)*W:nullptr;
          const double* sub_a=y>size_t(k)?a.data()+(y-k-1)*W:nullptr;
          const double* sub_b=sub_a?b.data()+(y-k-1)*W:nullptr;
          for(size_t x=0;x<W;++x) {
            if(add_a){vertical[2*x]+=add_a[x];vertical[2*x+1]+=add_b[x];}
            if(sub_a){vertical[2*x]-=sub_a[x];vertical[2*x+1]-=sub_b[x];}
          }
        }
#if defined(__ARM_NEON)
        float64x2_t sum=vdupq_n_f64(0.);
        for(size_t x=0;x<std::min(W,size_t(k)+1);++x)sum=vaddq_f64(sum,vld1q_f64(vertical+2*x));
        for(size_t x=0;x<W;++x) {
          if(x){if(x+size_t(k)<W)sum=vaddq_f64(sum,vld1q_f64(vertical+2*(x+k)));if(x>size_t(k))sum=vsubq_f64(sum,vld1q_f64(vertical+2*(x-k-1)));}
          oa[y*W+x]=vgetq_lane_f64(sum,0);ob[y*W+x]=vgetq_lane_f64(sum,1);
        }
#else
        double sa=0,sb=0;for(size_t x=0;x<std::min(W,size_t(k)+1);++x){sa+=vertical[2*x];sb+=vertical[2*x+1];}
        for(size_t x=0;x<W;++x) {
          if(x){if(x+size_t(k)<W){sa+=vertical[2*(x+k)];sb+=vertical[2*(x+k)+1];}if(x>size_t(k)){sa-=vertical[2*(x-k-1)];sb-=vertical[2*(x-k-1)+1];}}
          oa[y*W+x]=sa;ob[y*W+x]=sb;
        }
#endif
      }
    }
  });
}
void local_scale(MatchedWorkspace& w,size_t H,size_t W,const Vec& rho,double r,int k) {
  const size_t B=2,ch=(H+B-1)/B,cw=(W+B-1)/B,n=ch*cw;
  Vec& xs=w.local_xs;Vec& ns=w.local_ns;Vec& es=w.local_es;xs.assign(n,0.);ns.assign(n,0.);es.assign(n,0.);
  matched_parallel(ch,[&](size_t first,size_t last){
    for(size_t cy=first;cy<last;++cy)for(size_t cx=0;cx<cw;++cx) {
      const size_t y=2*cy,x=2*cx,i=y*W+x,j=cy*cw+cx;
      xs[j]=w.x[i]+(x+1<W?w.x[i+1]:0.);if(y+1<H)xs[j]+=w.x[i+W]+(x+1<W?w.x[i+W+1]:0.);ns[j]=w.bg_ns[j];
      double sum=(w.valid[i]?std::max(w.b[i],0.):0.)+(x+1<W && w.valid[i+1]?std::max(w.b[i+1],0.):0.);
      if(y+1<H)sum+=(w.valid[i+W]?std::max(w.b[i+W],0.):0.)+(x+1<W && w.valid[i+W+1]?std::max(w.b[i+W+1],0.):0.);
      es[j]=sum;
    }
  });
  double kc=1;for(int j=0;j<12;++j)kc+=2*rho[j]*std::max(int(B)-std::abs(noise_lags[j][0]),0)*std::max(int(B)-std::abs(noise_lags[j][1]),0)/double(B*B);
  Vec& c=w.local_c;Vec& phi=w.local_phi;Vec& kept=w.local_keep;Vec& sx=w.local_sx;Vec& se=w.local_se;Vec& u=w.local_u;Vec& kn=w.local_kn;
  c.assign(n,1.);phi.assign(n,1.);kept.resize(n);sx.resize(n);se.resize(n);u.resize(n);kn.resize(n);
  const size_t window_side=2*size_t(k)+1;
  const size_t max_count=std::min(ch,window_side)*std::min(cw,window_side);
  Vec phi_bias(max_count+1,0.);for(size_t j=1;j<=max_count;++j)phi_bias[j]=std::sqrt(2./j);
  for(int it=0;it<3;++it) {
    matched_parallel(n,[&](size_t first,size_t last){for(size_t i=first;i<last;++i) {
      const double e=c[i]*es[i],d=xs[i]-e;
      kept[i]=ns[i]>0 && d*d<9*phi[i]*std::max(e+ns[i]*r,1e-6)*kc;
      sx[i]=kept[i]?xs[i]:0.;se[i]=kept[i]?es[i]:0.;
    }});
    window_pair(sx,se,w.local_nk,w.local_su,ch,cw,k,w.window_scratch);
    sx.swap(w.local_nk);se.swap(w.local_su);
    matched_parallel(n,[&](size_t first,size_t last){for(size_t i=first;i<last;++i) {
      if(se[i]>1e-9)c[i]=sx[i]/se[i];const double e=c[i]*es[i],d=xs[i]-e;
      u[i]=kept[i]?d*d/(std::max(e+ns[i]*r,1e-6)*kc):0.;
    }});
    window_pair(kept,u,w.local_nk,w.local_su,ch,cw,k,w.window_scratch);
    const Vec& nk=w.local_nk;const Vec& su=w.local_su;
    matched_parallel(n,[&](size_t first,size_t last){for(size_t i=first;i<last;++i)if(nk[i]>=8)phi[i]=std::max(1.,(su[i]/nk[i]-phi_bias[size_t(nk[i])])/.9733);});
  }
  for(size_t i=0;i<n;++i)kn[i]=kept[i]?ns[i]:0.;
  window_pair(kn,kn,w.local_nk,w.local_su,ch,cw,k,w.window_scratch);kn.swap(w.local_nk);
  w.phi.resize(H*W);
  matched_parallel(H,[&](size_t first,size_t last){for(size_t y=first;y<last;++y)for(size_t x=0;x<W;++x) {
    const size_t i=y*W+x,j=(y/B)*cw+x/B;
    w.b[i]=c[j]*std::max(w.b[i],0.);w.phi[i]=phi[j];w.inv_neff[i]+=1./std::max(kn[j],1.);
  }});
}
struct ScoreConstants {double r,kinf,kappa[2],threshold[2],C3[2],C4[2],c1[2],c2[2],c3[2];};
inline double calibrated(double N,double b,double inv_neff,double phi,double A,double Q,
                         const ScoreConstants& c,int k,bool valid) {
  if(!valid || !(A>.5))return -std::numeric_limits<double>::infinity();
  const double bp=std::max(b,0.),V=std::max(std::max(bp+c.r,1e-6)*phi*(c.kappa[k]*Q+c.kinf*A*A*inv_neff),1e-12);
  const double i=1./std::sqrt(V),i2=i*i,g1=bp*c.C3[k]*i2*i,g2=bp*c.C4[k]*i2*i2;
  return N*i-(g1*c.c1[k]+g2*c.c2[k]-g1*g1*c.c3[k]);
}
// fine 核流式处理只需五行缓冲；完整残差保留给少量 coarse 候选。
void matched_scores(MatchedWorkspace& w,size_t H,size_t W,const ScoreConstants& sc) {
  const size_t n=H*W;w.residual.resize(n);w.z0.resize(n);Vec ring(5*W),row(W+4,0.),zero(W,0.);
  for(size_t y=0;y<H+2;++y) {
    if(y<H) {
      double* rp=row.data()+2;
      for(size_t x=0;x<W;++x){const size_t i=y*W+x;rp[x]=w.residual[i]=w.valid[i]?w.x[i]-w.b[i]:0.;}
      double* out=ring.data()+(y%5)*W;
      for(size_t x=0;x<W;++x)out[x]=kH1[0]*(rp[long(x)-2]+rp[x+2])+kH1[1]*(rp[long(x)-1]+rp[x+1])+kH1[2]*rp[x];
    }
    if(y<2)continue;const size_t yy=y-2;const double* rows[5];
    for(int j=-2;j<=2;++j){const long q=long(yy)+j;rows[j+2]=(q<0 || q>=long(H))?zero.data():ring.data()+(q%5)*W;}
    size_t x=0;const size_t base=yy*W;
#if defined(__ARM_NEON)
    const float64x2_t h0=vdupq_n_f64(kH1[0]),h1=vdupq_n_f64(kH1[1]),h2=vdupq_n_f64(kH1[2]);
    const float64x2_t zero2=vdupq_n_f64(0.),one=vdupq_n_f64(1.),rr=vdupq_n_f64(sc.r),kk=vdupq_n_f64(sc.kappa[0]),ki=vdupq_n_f64(sc.kinf);
    for(;x+2<=W;x+=2) {
      const size_t i=base+x;
      const float64x2_t N=vaddq_f64(vaddq_f64(vmulq_f64(h0,vaddq_f64(vld1q_f64(rows[0]+x),vld1q_f64(rows[4]+x))),vmulq_f64(h1,vaddq_f64(vld1q_f64(rows[1]+x),vld1q_f64(rows[3]+x)))),vmulq_f64(h2,vld1q_f64(rows[2]+x)));
      const float64x2_t bp=vmaxq_f64(vld1q_f64(w.b.data()+i),zero2);
      const double aa[2]={w.area[0][i],w.area[0][i+1]},qq[2]={w.square[0][i],w.square[0][i+1]};
      const float64x2_t A=vld1q_f64(aa),Q=vld1q_f64(qq),ph=w.phi.empty()?one:vld1q_f64(w.phi.data()+i);
      const float64x2_t term=vaddq_f64(vmulq_f64(kk,Q),vmulq_f64(vmulq_f64(vmulq_f64(ki,A),A),vld1q_f64(w.inv_neff.data()+i)));
      const float64x2_t V=vmaxq_f64(vmulq_f64(vmulq_f64(vmaxq_f64(vaddq_f64(bp,rr),vdupq_n_f64(1e-6)),ph),term),vdupq_n_f64(1e-12));
      const float64x2_t inv=vdivq_f64(one,vsqrtq_f64(V)),i2=vmulq_f64(inv,inv);
      const float64x2_t g1=vmulq_f64(vmulq_f64(vmulq_f64(bp,vdupq_n_f64(sc.C3[0])),i2),inv),g2=vmulq_f64(vmulq_f64(vmulq_f64(bp,vdupq_n_f64(sc.C4[0])),i2),i2);
      const float64x2_t shift=vsubq_f64(vaddq_f64(vmulq_f64(g1,vdupq_n_f64(sc.c1[0])),vmulq_f64(g2,vdupq_n_f64(sc.c2[0]))),vmulq_f64(vmulq_f64(g1,g1),vdupq_n_f64(sc.c3[0])));
      const uint64_t bits[2]={w.valid[i]?~uint64_t(0):0,w.valid[i+1]?~uint64_t(0):0};
      const uint64x2_t valid=vandq_u64(vld1q_u64(bits),vcgtq_f64(A,vdupq_n_f64(.5)));
      vst1q_f64(w.z0.data()+i,vbslq_f64(valid,vsubq_f64(vmulq_f64(N,inv),shift),vdupq_n_f64(-std::numeric_limits<double>::infinity())));
    }
#endif
    for(;x<W;++x) {
      const size_t i=base+x;const double N=kH1[0]*(rows[0][x]+rows[4][x])+kH1[1]*(rows[1][x]+rows[3][x])+kH1[2]*rows[2][x];
      w.z0[i]=calibrated(N,w.b[i],w.inv_neff[i],w.phi.empty()?1.:w.phi[i],w.area[0][i],w.square[0][i],sc,0,w.valid[i]);
    }
  }
}
double coarse_response(const Vec& R,long H,long W,long y,long x) {
  double total=0;for(int j=0;j<9;++j) {
    const long xx=x+j-4;if(xx<0 || xx>=W)continue;double a=0;
    for(int i=0;i<9;++i){const long yy=y+i-4;if(yy>=0 && yy<H)a+=kH2[i]*R[yy*W+xx];}
    total+=kH2[j]*a;
  }
  return total;
}
bool matched_sharp(const MatchedWorkspace& w,size_t H,size_t W,size_t root,double r,double fraction,double sigma) {
  const long y=root/W,x=root%W;
  auto residual=[&](long yy,long xx){const size_t i=yy*W+xx;return w.valid[i]?w.x[i]-w.b[i]:0.;};
  auto variance=[&](long yy,long xx){const size_t i=yy*W+xx;return w.valid[i]?std::max(w.x[i],w.b[i])+r:0.;};
  double best=-std::numeric_limits<double>::infinity();long py=y,px=x;
  for(long yy=std::max(0L,y-1);yy<std::min(long(H),y+2);++yy)for(long xx=std::max(0L,x-1);xx<std::min(long(W),x+2);++xx) {
    const double v=residual(yy,xx);if(v>best){best=v;py=yy;px=xx;}
  }
  const long y0=std::max(0L,py-1),y1=std::min(long(H),py+2),x0=std::max(0L,px-1),x1=std::min(long(W),px+2);
  double total=0,vt=0,second=-std::numeric_limits<double>::infinity(),vs=0;
  for(long yy=y0;yy<y1;++yy)for(long xx=x0;xx<x1;++xx){total+=residual(yy,xx);vt+=variance(yy,xx);}
  const int offsets[4][2]={{0,1},{0,-1},{1,0},{-1,0}};
  for(const auto& d:offsets){const long yy=py+d[0],xx=px+d[1];if(yy<y0 || yy>=y1 || xx<x0 || xx>=x1)continue;
    const double v=residual(yy,xx);if(v>second){second=v;vs=variance(yy,xx);}}
  const double core=residual(py,px)+second,vc=variance(py,px)+vs;
  return fraction*core-(total-core)>sigma*std::sqrt(fraction*fraction*vc+vt-vc);
}
void matched_basins(MatchedWorkspace& w,npy_bool* output,size_t H,size_t W,const MatchedSettings& s,const ScoreConstants& sc) {
  const size_t count=H*W;
  std::vector<std::pair<double,int>> ranked;
  for(size_t i=0;i<count;++i)if(w.valid[i] && w.z0[i]>=s.t_low)ranked.emplace_back(-w.z0[i],int(i));
  std::sort(ranked.begin(),ranked.end());std::vector<int> order;order.reserve(ranked.size());for(const auto& p:ranked)order.push_back(p.second);
  w.label.assign(count,-1);w.parent.resize(count);w.boundary.assign(count,0);w.needed.assign(count,0);w.good.assign(count,0);w.z1.resize(count);
  const double* region=w.z0.data();int* label=w.label.data();
  auto find=[&](int a){while(w.parent[a]!=a){w.parent[a]=w.parent[w.parent[a]];a=w.parent[a];}return a;};
  int roots[8];const long LH=H,LW=W;
  for(int p:order) {
    const long y=p/LW,x=p%LW;int nr=0;
    for(long dy=-1;dy<=1;++dy)for(long dx=-1;dx<=1;++dx) {
      if(!dy && !dx)continue;const long yy=y+dy,xx=x+dx;if(yy<0 || xx<0 || yy>=LH || xx>=LW)continue;
      const int l=label[yy*LW+xx];if(l<0)continue;const int q=find(l);bool seen=false;
      for(int j=0;j<nr;++j)seen|=roots[j]==q;if(!seen)roots[nr++]=q;
    }
    if(!nr){w.parent[p]=p;label[p]=p;continue;}
    int main=roots[0];for(int j=1;j<nr;++j){const int q=roots[j];if(region[q]>region[main] || (region[q]==region[main] && q<main))main=q;}
    for(int j=0;j<nr;++j){const int q=roots[j];if(q==main)continue;if(std::min(region[main],region[q])-region[p]<s.contrast)w.parent[q]=main;else w.boundary[p]=1;}
    label[p]=main;
  }
  auto z1_at=[&](long y,long x) {
    const size_t q=y*W+x;if(!w.needed[q]) {
      w.needed[q]=1;w.z1[q]=calibrated(coarse_response(w.residual,LH,LW,y,x),w.b[q],w.inv_neff[q],w.phi.empty()?1.:w.phi[q],w.area[1][q],w.square[1][q],sc,1,w.valid[q]);
    }
    return w.z1[q];
  };
  std::vector<int> touched;for(int p:order){const int root=find(label[p]);if(root==p)touched.push_back(root);}
  for(int pass=0;pass<2;++pass)for(int p:order) {
    const int root=find(label[p]);if(w.good[root])continue;const long y=p/LW,x=p%LW;
    const double z=pass?z1_at(y,x):region[p];if(!(z>=sc.threshold[pass]))continue;bool local=true;
    for(long yy=std::max(0L,y-1);yy<std::min(LH,y+2) && local;++yy)for(long xx=std::max(0L,x-1);xx<std::min(LW,x+2);++xx) {
      if((pass?z1_at(yy,xx):region[yy*LW+xx])>z){local=false;break;}
    }
    if(local)w.good[root]=1;
  }
  if(s.zinger_sigma>0)for(int root:touched)if(w.good[root] && matched_sharp(w,H,W,root,sc.r,s.zinger_fraction,s.zinger_sigma*std::sqrt(sc.kappa[0])))w.good[root]=0;
  std::fill(output,output+count,0);for(int p:order)output[p]=!w.boundary[p] && w.good[find(label[p])];
  discard_small_components(output,H,W,s.component_minimum);
}
void matched_proposals(const double* input,const npy_bool* mask,const int* bins,int nb,npy_bool* output,
                       size_t H,size_t W,const MatchedSettings& s,double* stats) {
  using Clock=std::chrono::steady_clock;auto time=Clock::now();int stage=0;
  auto mark=[&](){auto next=Clock::now();stats[20+stage++]=std::chrono::duration<double,std::milli>(next-time).count();time=next;};
  MatchedThreadBudget budget(H*W);
  MatchedWorkspace& w=matched_ws;const size_t n=H*W;w.x.resize(n);w.valid.resize(n);w.phi.clear();size_t area=0;
  for(size_t i=0;i<n;++i){const double v=(input[i]-s.offset)/s.gain;w.valid[i]=mask[i] && std::isfinite(v);w.x[i]=w.valid[i]?v:0.;area+=w.valid[i];}
  mark();MatchedBackground bg;double r=s.readout,G=1.,O=0.;
  stats[0]=1.;stats[1]=0.;stats[2]=r;stats[3]=stats[4]=stats[5]=1.;stats[6]=stats[7]=std::numeric_limits<double>::infinity();
  if(!matched_background(w,bins,nb,H,W,s,r,bg,false)){mark();std::fill(output,output+n,0);return;}
  mark();
  double last_o=0.,last_alpha=1.;
  if(s.calibrate) {
    for(int it=0;it<2;++it) {
      if(it && !matched_background(w,bins,nb,H,W,s,r,bg,true)){mark();std::fill(output,output+n,0);return;}
      background_pixels(w,bg,bins,H,W,s.tile,true);
      double alpha=1.,beta=r;if(!noise_fit(w,H,W,r,alpha,beta)){alpha=1.;beta=r;}
      const double o=std::max(-beta/alpha,0.);r=std::max(beta,0.)/(alpha*alpha);O+=G*o;G*=alpha;
      matched_parallel(n,[&](size_t first,size_t last){
        double* __restrict dst=w.x.data();const unsigned char* __restrict valid=w.valid.data();size_t i=first;
#if defined(__ARM_NEON)
        const float64x2_t off=vdupq_n_f64(s.offset),gain=vdupq_n_f64(s.gain),oo=vdupq_n_f64(O),gg=vdupq_n_f64(G);
        for(;i+2<=last;i+=2) {
          const float64x2_t x=vdivq_f64(vsubq_f64(vdivq_f64(vsubq_f64(vld1q_f64(input+i),off),gain),oo),gg);
          const uint64_t bits[2]={valid[i]?~uint64_t(0):0,valid[i+1]?~uint64_t(0):0};
          vst1q_f64(dst+i,vbslq_f64(vld1q_u64(bits),x,vdupq_n_f64(0.)));
        }
#endif
        for(;i<last;++i)dst[i]=valid[i]?(((input[i]-s.offset)/s.gain)-O)/G:0.;
      });
      for(double& e:bg.e)e=(e-o)/alpha;last_o=o;last_alpha=alpha;
    }
  }
  background_pixels(w,bg,bins,H,W,s.tile,false,last_o,last_alpha);
  stats[0]=G;stats[1]=O;stats[2]=r;mark();
  const Vec rho=s.calibrate?noise_correlation(w,H,W,r):Vec(12,0.);
  ScoreConstants sc;sc.r=r;sc.kinf=1.;for(int j=0;j<12;++j){stats[8+j]=rho[j];sc.kinf+=2*rho[j];}sc.kinf=std::max(sc.kinf,.1);stats[3]=sc.kinf;
  for(int k=0;k<2;++k) {
    const Vec h=matched_kernel(k);const auto noise=kernel_noise(h,rho);sc.kappa[k]=std::max(noise.kappa,.1);
    const double t=s.t_peak>0?s.t_peak:rft_threshold(area,noise,s.false_alarms/2);sc.threshold[k]=t;
    double a=0,b=0;for(double v:h){a+=v*v*v;b+=v*v*v*v;}sc.C3[k]=a*a;sc.C4[k]=b*b;
    sc.c1[k]=(t*t-1)/6;sc.c2[k]=(t*t*t-3*t)/24;sc.c3[k]=(2*t*t*t-5*t)/36;
    stats[4+k]=sc.kappa[k];stats[6+k]=t;
  }
  if(s.local_cells)local_scale(w,H,W,rho,r,s.local_cells);
  mark();matched_mask_terms(w,H,W);matched_scores(w,H,W,sc);mark();
  matched_basins(w,output,H,W,s,sc);mark();
}

PyObject* matched_spot_proposals(PyObject*,PyObject* args) {
  PyObject *data_obj,*mask_obj,*bins_obj;MatchedSettings s;
  if(!PyArg_ParseTuple(args,"OOOdddiidiiddddddii",&data_obj,&mask_obj,&bins_obj,&s.gain,&s.offset,&s.readout,
       &s.tile,&s.rounds,&s.clip,&s.min_bin,&s.calibrate,&s.false_alarms,&s.t_peak,&s.t_low,&s.contrast,
       &s.zinger_fraction,&s.zinger_sigma,&s.local_cells,&s.component_minimum))return nullptr;
  const double parameters[]={s.gain,s.offset,s.readout,s.clip,s.false_alarms,s.t_peak,s.t_low,s.contrast,s.zinger_fraction,s.zinger_sigma};
  bool finite=true;for(double p:parameters)finite&=std::isfinite(p);
  if(!finite || !(s.gain>0) || s.readout<0 || s.tile<4 || s.tile%2 || s.rounds<1 || s.clip<=0 || s.min_bin<1
     || s.false_alarms<=0 || s.t_peak<0 || s.contrast<0 || s.zinger_fraction<=0 || s.zinger_sigma<0 || s.local_cells<0 || s.component_minimum<1) {
    PyErr_SetString(PyExc_ValueError,"Invalid matched proposal settings");return nullptr;
  }
  PyArrayObject* data=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(data_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));if(!data)return nullptr;
  PyArrayObject* mask=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(mask_obj,NPY_BOOL,NPY_ARRAY_IN_ARRAY));if(!mask){Py_DECREF(data);return nullptr;}
  PyArrayObject* bins=nullptr;
  if(bins_obj!=Py_None){bins=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(bins_obj,NPY_INT32,NPY_ARRAY_IN_ARRAY));if(!bins){Py_DECREF(data);Py_DECREF(mask);return nullptr;}}
  const bool shape_ok=PyArray_NDIM(data)==2 && PyArray_NDIM(mask)==2 && PyArray_DIM(data,0)==PyArray_DIM(mask,0)
    && PyArray_DIM(data,1)==PyArray_DIM(mask,1) && PyArray_DIM(data,0)>=3 && PyArray_DIM(data,1)>=3
    && (!bins || (PyArray_NDIM(bins)==2 && PyArray_DIM(bins,0)==PyArray_DIM(data,0) && PyArray_DIM(bins,1)==PyArray_DIM(data,1)));
  int nb=1;bool bins_ok=true;
  if(shape_ok && bins){const int* b=static_cast<const int*>(PyArray_DATA(bins));for(npy_intp i=0;i<PyArray_SIZE(bins);++i){if(b[i]<0 || b[i]==std::numeric_limits<int>::max()){bins_ok=false;break;}nb=std::max(nb,b[i]+1);}}
  if(!shape_ok || !bins_ok || (shape_ok && PyArray_SIZE(data)>std::numeric_limits<int>::max())) {
    Py_DECREF(data);Py_DECREF(mask);Py_XDECREF(bins);PyErr_SetString(PyExc_ValueError,"Invalid matched proposal arrays");return nullptr;
  }
  npy_intp shape[2]={PyArray_DIM(data,0),PyArray_DIM(data,1)},stat_shape[1]={26};
  PyArrayObject* output=reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2,shape,NPY_BOOL));
  PyArrayObject* stats=reinterpret_cast<PyArrayObject*>(PyArray_ZEROS(1,stat_shape,NPY_DOUBLE,0));
  if(!output || !stats){Py_XDECREF(output);Py_XDECREF(stats);Py_DECREF(data);Py_DECREF(mask);Py_XDECREF(bins);return nullptr;}
  std::exception_ptr failure;
  Py_BEGIN_ALLOW_THREADS
  try{matched_proposals(static_cast<const double*>(PyArray_DATA(data)),static_cast<const npy_bool*>(PyArray_DATA(mask)),
       bins?static_cast<const int*>(PyArray_DATA(bins)):nullptr,nb,static_cast<npy_bool*>(PyArray_DATA(output)),shape[0],shape[1],s,static_cast<double*>(PyArray_DATA(stats)));}
  catch(...){failure=std::current_exception();}
  Py_END_ALLOW_THREADS
  Py_DECREF(data);Py_DECREF(mask);Py_XDECREF(bins);
  if(failure){Py_DECREF(output);Py_DECREF(stats);try{std::rethrow_exception(failure);}catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_RuntimeError,e.what());return nullptr;}}
  return Py_BuildValue("NN",reinterpret_cast<PyObject*>(output),reinterpret_cast<PyObject*>(stats));
}

PyMethodDef methods[]={{"spot_proposals",spot_proposals,METH_VARARGS,"Standardized or count-excess proposals using fused summed-area statistics."},
                       {"matched_spot_proposals",matched_spot_proposals,METH_VARARGS,
                        "Matched-filter proposals: radial x tile background, watershed, zinger test."},
                       {"known_background_wls",known_background_wls,METH_VARARGS,
                        "Fixed known-background signed WLS with Cholesky covariance."},
                       {"known_background_irls",known_background_irls,METH_VARARGS,
                        "Fused DIALS-compatible fixed-background IRLS."},
                       {nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_native",nullptr,-1,methods};
}

PyMODINIT_FUNC PyInit__native(){import_array();return PyModule_Create(&module);}
