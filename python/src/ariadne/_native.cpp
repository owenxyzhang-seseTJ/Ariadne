// Ariadne native standardized/count-excess proposals. Rules match spots.py.
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

PyMethodDef methods[]={{"spot_proposals",spot_proposals,METH_VARARGS,"Standardized or count-excess proposals using fused summed-area statistics."},
                       {"known_background_wls",known_background_wls,METH_VARARGS,
                        "Fixed known-background signed WLS with Cholesky covariance."},
                       {"known_background_irls",known_background_irls,METH_VARARGS,
                        "Fused DIALS-compatible fixed-background IRLS."},
                       {nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_native",nullptr,-1,methods};
}

PyMODINIT_FUNC PyInit__native(){import_array();return PyModule_Create(&module);}
