// 离散 Poisson–Gaussian 卷积；保留自适应求和窗口、尾界与计算预算。

inline double pg_log_probability(double n,double mu) {
  if(n==0)return -mu;
  const double delta=n-mu;
  double deviance;
  if(std::fabs(delta)<=.125*mu) {
    const double d=delta/mu;double p=1./(24*23);
    for(int power=23;power>1;--power)p=(power%2?-1.:1.)/(power*(power-1))+d*p;
    deviance=(mu*d)*d*p;
  } else deviance=n*(std::log(n)-std::log(mu))-delta;
  const double log2pi=std::log(2*std::acos(-1.));double remainder;
  if(n<16)remainder=std::lgamma(n+1)-(n+.5)*std::log(n)+n-.5*log2pi;
  else {
    const double inv=1/n,inv2=inv*inv;
    remainder=inv*(1./12+inv2*(-1./360+inv2*(1./1260+inv2*(-1./1680+inv2*(1./1188+inv2*(-691./360360+inv2/156))))));
  }
  return -deviance-(.5*(log2pi+std::log(n))+remainder);
}
inline double pg_residual(double y,double gain,double n) {
  // 与 NumPy 的乘法后减法一致，阻止编译器在此处融合为 FMA。
  volatile double product=gain*n;
  return y-product;
}
inline double pg_logadd(double a,double b) {
  if(a==-std::numeric_limits<double>::infinity())return b;
  if(b==-std::numeric_limits<double>::infinity())return a;
  const double hi=std::max(a,b),lo=std::min(a,b);
  return hi+std::log1p(std::exp(lo-hi));
}
PyObject* poisson_gaussian_sums(PyObject*,PyObject* args) {
  PyObject *y_obj,*mu_obj;double gain,variance,tolerance;Py_ssize_t pixel_budget,remaining;
  if(!PyArg_ParseTuple(args,"OOdddnn",&y_obj,&mu_obj,&gain,&variance,&tolerance,&pixel_budget,&remaining))return nullptr;
  PyArrayObject* y=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(y_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));if(!y)return nullptr;
  PyArrayObject* mu=reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(mu_obj,NPY_DOUBLE,NPY_ARRAY_IN_ARRAY));
  if(!mu){Py_DECREF(y);return nullptr;}
  const npy_intp count=PyArray_SIZE(y);
  bool valid=PyArray_NDIM(y)==1 && PyArray_NDIM(mu)==1 && count==PyArray_SIZE(mu)
    && std::isfinite(gain) && gain>0 && std::isfinite(variance) && variance>0
    && std::isfinite(tolerance) && tolerance>0 && tolerance<1 && pixel_budget>0 && remaining>=0;
  const double* observed=static_cast<const double*>(PyArray_DATA(y));
  const double* means=static_cast<const double*>(PyArray_DATA(mu));
  if(valid)for(npy_intp i=0;i<count;++i)valid&=std::isfinite(observed[i]) && std::isfinite(means[i]) && means[i]>=0;
  if(!valid){Py_DECREF(y);Py_DECREF(mu);PyErr_SetString(PyExc_ValueError,"Invalid Poisson-Gaussian arrays or settings");return nullptr;}
  npy_intp shape[1]={count};
  PyArrayObject* logs=reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1,shape,NPY_DOUBLE));
  PyArrayObject* errors=reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1,shape,NPY_DOUBLE));
  if(!logs || !errors){Py_XDECREF(logs);Py_XDECREF(errors);Py_DECREF(y);Py_DECREF(mu);return nullptr;}
  double* out=static_cast<double*>(PyArray_DATA(logs));double* tails=static_cast<double*>(PyArray_DATA(errors));
  Py_ssize_t used=0;bool exhausted=false;std::exception_ptr failure;
  Py_BEGIN_ALLOW_THREADS
  try {
    const double sd=std::sqrt(variance),normalizer=-.5*(std::log(2*std::acos(-1.))+std::log(variance));
    const double log_tolerance=std::log(tolerance),minus_inf=-std::numeric_limits<double>::infinity();
    // 零均值项与参考路径一样预先计入全局预算。
    for(npy_intp i=0;i<count;++i)if(means[i]==0)++used;
    if(used>remaining)exhausted=true;
    Vec terms;
    for(npy_intp i=0;i<count && !exhausted;++i) {
      const double yy=observed[i],mean=means[i];tails[i]=0.;
      if(mean==0){const double z=yy/sd;out[i]=normalizer-.5*z*z;continue;}
      const double extent=std::max(mean,yy/gain);
      if(!std::isfinite(extent) || extent>=4503599627370496.)throw std::invalid_argument("Poisson-Gaussian photon index exceeds reliable float64 integer range");
      const double log_mu=std::log(mean);
      auto ratio=[&](long long n){return log_mu-std::log(double(n)+1.)+gain*pg_residual(yy,gain,double(n)+.5)/variance;};
      long long left=0,right=std::max(1LL,static_cast<long long>(std::ceil(extent)));
      while(left<right){const long long middle=left+(right-left)/2;if(ratio(middle)>0)left=middle+1;else right=middle;}
      const long long mode=left;Py_ssize_t spent=0,width=8;
      while(true) {
        const long long lo=std::max(0LL,mode-width),hi=mode+width;
        const auto size=static_cast<Py_ssize_t>(hi-lo+1);
        if(size>pixel_budget-spent || size>remaining-used){exhausted=true;break;}
        terms.resize(size);double maximum=minus_inf;
        for(Py_ssize_t j=0;j<size;++j) {
          const double n=double(lo+j),z=pg_residual(yy,gain,n)/sd;
          terms[j]=pg_log_probability(n,mean)+normalizer-.5*z*z;
          maximum=std::max(maximum,terms[j]);
        }
        double sum=0.;for(double v:terms)sum+=std::exp(v-maximum);
        const double total=maximum+std::log(sum);
        double lower=minus_inf;
        if(lo>0){const double lr=-ratio(lo-1);lower=terms.front()+lr-std::log(-std::expm1(lr));}
        const double lr=ratio(hi),upper=terms.back()+lr-std::log(-std::expm1(lr));
        const double tail=pg_logadd(lower,upper)-total;
        spent+=size;used+=size;
        if(tail<=log_tolerance){out[i]=total;tails[i]=std::exp(tail);break;}
        // 下一轮必定超预算时立即退出，避免整数翻倍溢出。
        if(width>pixel_budget/2){exhausted=true;break;}
        width*=2;
      }
    }
  } catch(...){failure=std::current_exception();}
  Py_END_ALLOW_THREADS
  Py_DECREF(y);Py_DECREF(mu);
  if(failure || exhausted) {
    Py_DECREF(logs);Py_DECREF(errors);
    if(exhausted)Py_RETURN_NONE;
    try{std::rethrow_exception(failure);}catch(const std::bad_alloc&){return PyErr_NoMemory();}
    catch(const std::exception& e){PyErr_SetString(PyExc_ValueError,e.what());return nullptr;}
  }
  return Py_BuildValue("NNn",reinterpret_cast<PyObject*>(logs),reinterpret_cast<PyObject*>(errors),used);
}
