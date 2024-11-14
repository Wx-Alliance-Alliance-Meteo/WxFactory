#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>

#ifdef __CUDACC__
  #define DEVICE_SPACE __device__
  #include <cuda/std/complex>
  #include <cuda/std/ccomplex>
  using complex_t = cuda::std::complex<double>;
#else 
  #define DEVICE_SPACE
  using namespace std;
  using complex_t = std::complex<double>;
  #include <pybind11/numpy.h>
#endif

// Declarations
DEVICE_SPACE const double p0 = 100000.;
DEVICE_SPACE const double Rd = 287.05;
DEVICE_SPACE const double cpd = 1005.46;
DEVICE_SPACE const double cvd = (cpd - Rd);
DEVICE_SPACE const double kappa = Rd / cpd;
DEVICE_SPACE const double heat_capacity_ratio = cpd / cvd;
DEVICE_SPACE const double inp0 = 1.0 / p0;
DEVICE_SPACE const double Rdinp0 = Rd * inp0;

const int BLOCK_SIZE = 256;

// Define non-standard operations for complex numbers
DEVICE_SPACE inline complex_t fabs(const complex_t &z)
{
  return sqrt(z.real()*z.real() + z.imag()*z.imag());
}

DEVICE_SPACE inline complex_t fmax(const complex_t &z1, const complex_t &z2)
{   
  if (abs(z1)>abs(z2)) return z1;
  else return z2;
}

DEVICE_SPACE inline complex_t fmin(const complex_t &z1, const complex_t &z2)
{   
  if (abs(z1)<abs(z2)) return z1;
  else return z2;
}

#endif // DEFINITIONS_H