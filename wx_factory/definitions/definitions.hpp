#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <cmath>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>

#ifdef __CUDACC__

#define HOST_DEVICE_SPACE __host__ __device__
#define DEVICE_SPACE      __device__
#include <cuda/std/array>
#include <cuda/std/ccomplex>
#include <cuda/std/complex>
using complex_t = cuda::std::complex<double>;
template <class T, std::size_t N>
using array = cuda::std::array<T, N>;

#define cudaCheck(cmd) cudaApiAssert((cmd), __FILE__, __LINE__)

//! Test the given return code and abort the program with an error message if it is not
//! "cudaSuccess"
inline void cudaApiAssert(const cudaError_t code, const char* filename, const int line) {
  if (code != cudaSuccess)
  {
    std::cerr << "CUDA API call failed: " << cudaGetErrorString(code) << ", at "
              << filename << ":" << line << "\n";
    exit(-1);
  }
}

#else

#define DEVICE_SPACE
#define HOST_DEVICE_SPACE
#include <array>
using complex_t = std::complex<double>;
template <class T, std::size_t N>
using array = std::array<T, N>;

#endif

// Declarations
// DEVICE_SPACE const double p0 = 100000.0;
// DEVICE_SPACE const double Rd = 287.05;
// DEVICE_SPACE const double cpd                 = 1005.46;
// DEVICE_SPACE const double cvd                 = (cpd - Rd);
// DEVICE_SPACE const double kappa               = Rd / cpd;
// DEVICE_SPACE const double heat_capacity_ratio = cpd / cvd;
// DEVICE_SPACE const double inp0                = 1.0 / p0;
// DEVICE_SPACE const double Rdinp0              = Rd * inp0;
#define p0                  100000.0
#define Rd                  287.05
#define cpd                 1005.46
#define cvd                 (cpd - Rd)
#define kappa               (Rd / cpd)
#define heat_capacity_ratio (cpd / cvd)
#define inp0                (1.0 / p0)
#define Rdinp0              (Rd * inp0)

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
const int h21 = 3;
const int h22 = 4;
const int h23 = 5;
const int h31 = 6;
const int h32 = 7;
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

// Returns the index in a flattened array from a 4d index group
DEVICE_SPACE inline int
get_c_index(const int i, const int j, const int k, const int l, const int shape[4]) {
  return i * shape[1] * shape[2] * shape[3] + j * shape[2] * shape[3] + k * shape[3] + l;
}

DEVICE_SPACE inline int get_c_index(
    const int i,
    const int j,
    const int k,
    const int l,
    const int m,
    const int shape[5]) {
  return i * shape[1] * shape[2] * shape[3] * shape[4] +
         j * shape[2] * shape[3] * shape[4] + k * shape[3] * shape[4] + l * shape[4] + m;
}

// Return the cupy pointer
template <typename num_t>
num_t* get_cupy_pointer(pybind11::object obj) {
  uintptr_t cp_ptr = obj.attr("data").attr("ptr").cast<uintptr_t>();
  return reinterpret_cast<num_t*>(cp_ptr);
  // return cp_ptr;
}

//! Extract raw pointer to given array's data and cast it to the requested type
//! \tparam num_t The type we wish to get from the input array
template <typename num_t>
const num_t* get_raw_ptr(const pybind11::array_t<num_t>& a) {
  return static_cast<num_t*>(a.request().ptr);
}
template <typename num_t>
num_t* get_raw_ptr(pybind11::array_t<num_t>& a) {
  return static_cast<num_t*>(a.request().ptr);
}

template <typename num_t>
const num_t* get_raw_ptr(const pybind11::object& obj) {
  uintptr_t cp_ptr = obj.attr("data").attr("ptr").cast<uintptr_t>();
  return reinterpret_cast<num_t*>(cp_ptr);
}
template <typename num_t>
num_t* get_raw_ptr(pybind11::object& obj) {
  uintptr_t cp_ptr = obj.attr("data").attr("ptr").cast<uintptr_t>();
  return reinterpret_cast<num_t*>(cp_ptr);
}

template <typename num_t>
struct var
{
  num_t* value = nullptr;

  HOST_DEVICE_SPACE var() {}
  HOST_DEVICE_SPACE var(num_t* field, const size_t index) : value(field + index) {}
  HOST_DEVICE_SPACE var(num_t* field) : value(field) {}

  HOST_DEVICE_SPACE operator num_t() const { return *value; }

  HOST_DEVICE_SPACE num_t  operator*() const { return *value; }
  HOST_DEVICE_SPACE num_t& operator*() { return *value; }

  HOST_DEVICE_SPACE void move_index(const int64_t index_change) { value += index_change; }
};

template <typename num_t, int size>
DEVICE_SPACE array<var<num_t>, size>
             make_var_sequence(const num_t* offset, const size_t stride) {
  array<var<num_t>, size> result;
  for (int i = 0; i < size; i++)
  {
    result[i] = i * stride + offset;
  }
  return result;
}

template <typename num_t, int num_var>
struct var_multi
{
  var<num_t> val[num_var];

  HOST_DEVICE_SPACE num_t  operator[](int i) const { return *val[i]; }
  HOST_DEVICE_SPACE num_t& operator[](int i) { return *val[i]; }

  HOST_DEVICE_SPACE var_multi(num_t* field, const size_t index, const size_t stride) {
    for (int i = 0; i < num_var; ++i)
    {
      val[i] = field + index + i * stride;
    }
  }

  HOST_DEVICE_SPACE void move_index(const int64_t index_change) {
    for (int i = 0; i < num_var; ++i)
    {
      val[i].move_index(index_change);
    }
  }
};
#endif // DEFINITIONS_H
