#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

// #include <cmath>

// #include <pybind11/complex.h>
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
//! Numpy (const) version
template <typename num_t>
const num_t* get_raw_ptr(const pybind11::array_t<num_t>& a) {
  return static_cast<num_t*>(a.request().ptr);
}

//! Extract raw pointer to given array's data and cast it to the requested type
//! \tparam num_t The type we wish to get from the input array
//! Numpy version
template <typename num_t>
num_t* get_raw_ptr(pybind11::array_t<num_t>& a) {
  return static_cast<num_t*>(a.request().ptr);
}

//! Extract raw pointer to given array's data and cast it to the requested type
//! \tparam num_t The type we wish to get from the input array
//! CuPy (const) version
template <typename num_t>
const num_t* get_raw_ptr(const pybind11::object& obj) {
  uintptr_t cp_ptr = obj.attr("data").attr("ptr").cast<uintptr_t>();
  return reinterpret_cast<num_t*>(cp_ptr);
}

//! Extract raw pointer to given array's data and cast it to the requested type
//! \tparam num_t The type we wish to get from the input array
//! CuPy version
template <typename num_t>
num_t* get_raw_ptr(pybind11::object& obj) {
  uintptr_t cp_ptr = obj.attr("data").attr("ptr").cast<uintptr_t>();
  return reinterpret_cast<num_t*>(cp_ptr);
}

#endif // COMMON_FUNCTIONS_H
