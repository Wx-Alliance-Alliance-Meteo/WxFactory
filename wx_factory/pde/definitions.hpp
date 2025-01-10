#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <cmath>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifdef __CUDACC__
#define DEVICE_SPACE __device__
#include <cuda/std/ccomplex>
#include <cuda/std/complex>
using complex_t = cuda::std::complex<double>;
#else
#define DEVICE_SPACE
using namespace std;
using complex_t = std::complex<double>;
#endif

// Declarations
DEVICE_SPACE const double p0                  = 100000.;
DEVICE_SPACE const double Rd                  = 287.05;
DEVICE_SPACE const double cpd                 = 1005.46;
DEVICE_SPACE const double cvd                 = (cpd - Rd);
DEVICE_SPACE const double kappa               = Rd / cpd;
DEVICE_SPACE const double heat_capacity_ratio = cpd / cvd;
DEVICE_SPACE const double inp0                = 1.0 / p0;
DEVICE_SPACE const double Rdinp0              = Rd * inp0;

const int BLOCK_SIZE = 256;

//!{ \name Define non-standard operations for complex numbers
DEVICE_SPACE inline complex_t fabs(const complex_t& z) {
  return sqrt(z.real() * z.real() + z.imag() * z.imag());
}

DEVICE_SPACE inline complex_t fmax(const complex_t& z1, const complex_t& z2) {
  if (z1.real() > z2.real())
    return z1;
  else if (z1.real() < z2.real())
    return z2;
  else if (z1.imag() > z2.imag())
    return z1;
  else
    return z2;
}

DEVICE_SPACE inline complex_t fmin(const complex_t& z1, const complex_t& z2) {
  if (z1.real() < z2.real())
    return z1;
  else if (z1.real() > z2.real())
    return z2;
  else if (z1.imag() < z2.imag())
    return z1;
  else
    return z2;
}
//!}

// For debugging
DEVICE_SPACE inline double to_real(const complex_t& x) {
  return x.real();
}

// For debugging
DEVICE_SPACE inline double to_real(const double& x) {
  return x;
}

//!{ \name Metric array indices
const int h11 = 0;
const int h12 = 1;
const int h13 = 2;
const int h22 = 4;
const int h23 = 5;
const int h33 = 8;

const int c101 = 0;
const int c102 = 1;
const int c103 = 2;
const int c111 = 3;
const int c112 = 4;
const int c113 = 5;
const int c122 = 6;
const int c123 = 7;
const int c133 = 8;
const int c201 = 9;
const int c202 = 10;
const int c203 = 11;
const int c211 = 12;
const int c212 = 13;
const int c213 = 14;
const int c222 = 15;
const int c223 = 16;
const int c233 = 17;
const int c301 = 18;
const int c302 = 19;
const int c303 = 20;
const int c311 = 21;
const int c312 = 22;
const int c313 = 23;
const int c322 = 24;
const int c323 = 25;
const int c333 = 26;
//!}

#endif // DEFINITIONS_H
