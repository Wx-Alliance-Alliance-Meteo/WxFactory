#ifndef EXCHANGES_HPP_
#define EXCHANGES_HPP_

#include "common/parameters.hpp"
#include "transform_rule.hpp"

#include <vector>
#include <numeric>
#include <functional>
#include <complex>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

namespace py = pybind11;

template <typename num_t>
void memcpy_faces_wrapper(
    num_t* send_buffer,
    const num_t* south,
    const num_t* north,
    const num_t* west,
    const num_t* east,
    size_t face_size
);


#ifdef __CUDACC__

struct Flags { unsigned char f[4]; };

template <typename num_t>
__global__ void memcpy_faces_kernel(
    num_t* send_buffer,
    const num_t* south,
    const num_t* north,
    const num_t* west,
    const num_t* east,
    size_t face_size
);


template <typename num_t>
void flip_axis_wrapper_gpu(num_t* send_buffer, const std::vector<int>& slice_shape, const std::vector<int>& flip_axes, size_t face_size, const std::vector<bool>& flip_flags);

template <typename num_t>
__global__ void flip_axis_kernel(
    num_t* send_buffer,
    int dim,
    const size_t stride_axis,
    const size_t outer_rows,
    const size_t face_size,
    Flags flags
);

template <typename num_t, typename real_t>
struct PairParams {
    const num_t* data[4];
    const real_t* boundary[4];
};

template <typename num_t, typename real_t>
void convert_pair_wrapper_gpu(
    const num_t* face_data[4],
    const real_t* face_boundary[4],
    num_t* send_buffer,
    const size_t face_size,
    const int panel,
    const size_t coord_size,
    const size_t var_size
);

template <typename num_t, typename real_t>
__global__ void convert_pair_kernel(
    PairParams<num_t, real_t> params, num_t* send_buffer, const int panel, const size_t coord_size, const size_t var_size, const size_t face_size);

#endif // __CUDACC__

#endif //EXCHANGES_HPP_