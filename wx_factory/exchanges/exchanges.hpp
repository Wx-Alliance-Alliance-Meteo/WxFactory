#ifndef EXCHANGES_HPP_
#define EXCHANGES_HPP_

#include "common/parameters.hpp"
#include "common/transform_rule.hpp"

#include <vector>
#include <numeric>
#include <functional>
#include <complex>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;

template <typename T>
void memcpy_faces_wrapper(
    T* send_buffer,
    const T* south,
    const T* north,
    const T* west,
    const T* east,
    size_t face_size
);


#ifdef __CUDACC__

struct Flags { unsigned char f[4]; };

template <typename T>
__global__ void memcpy_faces_kernel(
    T* send_buffer,
    const T* south,
    const T* north,
    const T* west,
    const T* east,
    size_t face_size
);


template <typename T>
void flip_axis_wrapper_gpu(T* send_buffer, const std::vector<int>& slice_shape, const std::vector<int>& flip_axes, size_t face_size, const std::vector<bool>& flip_flags);

template <typename T>
__global__ void flip_axis_kernel(
    T* send_buffer,
    int dim,
    const size_t stride_axis,
    const size_t outer_rows,
    const size_t face_size,
    Flags flags
);

template <typename T, typename U>
struct PairParams {
    const T* data[4];
    const U* boundary[4];
};

template <typename T, typename U>
void convert_pair_wrapper_gpu(
    const T* face_data[4],
    const U* face_boundary[4],
    T* send_buffer,
    const size_t face_size,
    const int panel,
    const size_t coord_size,
    const size_t var_size
);

template <typename T, typename U>
__global__ void convert_pair_kernel(
    PairParams<T, U> params, T* send_buffer, const int panel, const size_t coord_size, const size_t var_size, const size_t face_size);

#endif // __CUDACC__

#endif //EXCHANGES_HPP_