#ifndef EXCHANGES_HPP_
#define EXCHANGES_HPP_

#include "common/parameters.hpp"

#include <vector>
#include <numeric>
#include <functional>
#include <complex>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <cuda_runtime.h>

namespace py = pybind11;

template <typename T, typename U>
void start_exchange_euler_3d(
    py::array_t<T, py::array::c_style> send_buffer,
    py::array_t<T, py::array::c_style> south,
    py::array_t<T, py::array::c_style> north,
    py::array_t<T, py::array::c_style> west,
    py::array_t<T, py::array::c_style> east,
    py::array_t<U, py::array::c_style> boundary_sn,
    py::array_t<U, py::array::c_style> boundary_we,

    // reference slice shape (n_variables, n_vert, n_hori, n*n nodal pts)
    const std::vector<int>& shape,
    const std::vector<int>& flip_dims,
    const std::vector<bool>& flip_flags,
    const int panel
);

struct TransformRule {
    int s11, s12, s13, s14;
    int s21, s22, s23, s24;
};

// Cuda kernels
template <typename T>
__global__ void memcpy_faces_kernel(
    T* send_buffer,
    const T* south,
    const T* north,
    const T* west,
    const T* east,
    int block_size);

template <typename T, typename U>
__global__ void convert_pair_kernel(
    const T* __restrict__ a1,
    const T* __restrict__ a2,
    const U* __restrict__ coord,
    T* __restrict__ o1,
    T* __restrict__ o2,
    int panel,
    int neighbour,
    int n_coord,
    int var_size);

template <typename T>
__global__ void flip_axis_kernel(
    T* arr,
    const int* shape,
    const int* stride,
    int ndim,
    int axis,
    size_t total_size);


// Host wrappers
template <typename T>
void memcpy_faces_wrapper(
    T* send_buffer,
    const T* south,
    const T* north,
    const T* west,
    const T* east,
    int block_size);

template <typename T, typename U>
void convert_pair_wrapper(
    const T* a1,
    const T* a2,
    const U* coord,
    T* o1,
    T* o2,
    int panel,
    int neighbour,
    int n_coord,
    int var_size);

template <typename T>
void flip_axis_wrapper(
    T* arr,
    const std::vector<int>& shape,
    const std::vector<int>& axes);

template <typename T, typename U>
void start_exchange_euler_3d(
    py::array_t<T, py::array::c_style> send_buffer,
    py::array_t<T, py::array::c_style> south,
    py::array_t<T, py::array::c_style> north,
    py::array_t<T, py::array::c_style> west,
    py::array_t<T, py::array::c_style> east,
    py::array_t<U, py::array::c_style> boundary_sn,
    py::array_t<U, py::array::c_style> boundary_we,
    const std::vector<int>& shape,
    const std::vector<int>& flip_dims,
    const std::vector<bool>& flip_flags,
    int panel);

// Conversion table
inline constexpr TransformRule rules[6][4] = {
    // Panel 0
    {
        { 1, 0, 0, 1,   0, 1, 0, 0 },  // South
        { 1, 0, 0, -1,  0, 1, 0, 0 },  // North
        { 1, 0, 0, 0,   0, 1, 1, 0 },  // West
        { 1, 0, 0, 0,   0, 1, -1, 0 }, // East
    },
    // Panel 1
    {
        { 0, 1, 0, 0,  -1, 0, 0, -1 }, // South
        { 0,-1, 0, 0,   1, 0, 0, -1 }, // North
        { 1, 0, 0, 0,   0, 1, 1, 0 },  // West
        { 1, 0, 0, 0,   0, 1,-1, 0 },  // East
    },
    // Panel 2
    {
        {-1, 0, 0, -1,  0,-1, 0, 0 },  // South
        {-1, 0, 0, 1,   0,-1, 0, 0 },  // North
        { 1, 0, 0, 0,   0, 1, 1, 0 },  // West
        { 1, 0, 0, 0,   0, 1,-1, 0 },  // East
    },
    // Panel 3
    {
        { 0,-1, 0, 0,   1, 0, 0, 1 },  // South
        { 0, 1, 0, 0,  -1, 0, 0, 1 },  // North
        { 1, 0, 0, 0,   0, 1, 1, 0 },  // West
        { 1, 0, 0, 0,   0, 1,-1, 0 },  // East
    },
    // Panel 4
    {
        { 1, 0, 0, 1,   0, 1, 0, 0 },  // South
        {-1, 0, 0, 1,   0,-1, 0, 0 },  // North
        { 0,-1, -1, 0,   1, 0, 0, 0 }, // West
        { 0, 1, -1, 0,  -1, 0, 0, 0 }, // East
    },
    // Panel 5
    {
        {-1, 0, 0,-1,   0,-1, 0, 0 },  // South
        { 1, 0, 0,-1,   0, 1, 0, 0 },  // North
        { 0, 1, 1, 0,  -1, 0, 0, 0 },  // West
        { 0,-1, 1, 0,   1, 0, 0, 0 },  // East
    }
};

extern __constant__ TransformRule rules_c[6][4];



#endif //EXCHANGES_HPP_