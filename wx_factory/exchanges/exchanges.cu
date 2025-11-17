#include <iostream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include <pybind11/complex.h>

#include "kernels/kernels.h"

#include <vector>
#include <cuda_runtime.h>
#include <cmath>

#include "exchanges.hpp"

namespace py = pybind11;

template <typename T>
T* get_device_ptr(py::object& obj) {
    auto iface = obj.attr("__cuda_array_interface__");
    auto data_tuple = iface["data"].cast<py::tuple>();
    uintptr_t ptr_value = data_tuple[0].cast<uintptr_t>();
    return reinterpret_cast<T*>(ptr_value);
}


template <typename T, typename U>
void start_exchange_euler_3d(
    py::object& send_buffer_obj,
    py::object& south_obj,
    py::object& north_obj,
    py::object& west_obj,
    py::object& east_obj,
    py::object& boundary_sn_obj,
    py::object& boundary_we_obj,

    // reference slice shape (n_variables, n_vert, n_hori, n*n nodal pts)
    const std::vector<int>& slice_shape,
    const std::vector<int>& flip_axes,
    const std::vector<bool>& flip_flags,
    const int panel
)
{
    T* send_buffer = get_device_ptr<T>(send_buffer_obj);
    const T* south = get_device_ptr<T>(south_obj);
    const T* north = get_device_ptr<T>(north_obj);
    const T* west  = get_device_ptr<T>(west_obj);
    const T* east  = get_device_ptr<T>(east_obj);

    const U* boundary_sn = get_device_ptr<U>(boundary_sn_obj);
    const U* boundary_we = get_device_ptr<U>(boundary_we_obj);

    const T* face_data[4] = { south, north, west, east };
    const U* face_boundary[4] = { boundary_sn, boundary_sn, boundary_we, boundary_we };

    
    const int n_var = slice_shape[0]; // number of variables, e.g. temperature, density, etc
    const size_t var_size = std::accumulate(slice_shape.begin() + 1, slice_shape.end(), 1, std::multiplies<>()); // elements per variable
    const size_t coord_size = std::accumulate(slice_shape.begin() + 2, slice_shape.end(), 1, std::multiplies<>()); // size per vertical coordinate
    const size_t face_size = n_var * var_size; // total size face

    memcpy_faces_wrapper<T>(send_buffer, south, north, west, east, face_size);

    convert_pair_wrapper_gpu<T, U>(face_data, face_boundary, send_buffer, face_size, panel, coord_size, var_size);

    flip_axis_wrapper_gpu<T>(send_buffer, slice_shape, flip_axes, face_size, flip_flags);
}

void start_exchange_euler_3d_wrapper(
    py::object& send_buffer,
    py::object& south,
    py::object& north,
    py::object& west,
    py::object& east,
    py::object& boundary_sn,
    py::object& boundary_we,
    const std::vector<int>& slice_shape, // reference slice shape (n_variables, n_vert, n_hori, n*n nodal pts)
    const std::vector<int>& flip_axes,
    const std::vector<bool>& flip_flags,
    const int panel
)
{

    std::string T_type = py::str(send_buffer.attr("dtype").attr("name"));

    // Different template according to buffer type
    if (T_type == "float64") {
        start_exchange_euler_3d<double, double>(
            send_buffer,
            south, north, west, east,
            boundary_sn, boundary_we,
            slice_shape, flip_axes, flip_flags, panel
        );
    }
    else if (T_type == "complex128") {
        start_exchange_euler_3d<complex_t, double>(
            send_buffer,
            south, north, west, east,
            boundary_sn, boundary_we,
            slice_shape, flip_axes, flip_flags, panel
        );
    }
    else {
        throw std::runtime_error("Unsupported buffer type: " + T_type);
    }
}

// Copy 4 face buffers into 1 contigous buffer
template <typename T>
void memcpy_faces_wrapper(
    T* send_buffer,
    const T* south,
    const T* north,
    const T* west,
    const T* east,
    size_t face_size
) {

    cudaMemcpyAsync(send_buffer, south, face_size * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(send_buffer + face_size, north, face_size * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(send_buffer + 2*face_size, west,  face_size * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(send_buffer + 3*face_size, east,  face_size * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T, typename U>
void convert_pair_wrapper_gpu(
    const T* face_data[4],
    const U* face_boundary[4],
    T* send_buffer,
    const size_t face_size,
    const int panel,
    const size_t coord_size,
    const size_t var_size
) {

    const int BLOCK_SIZE = 128;
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((var_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 4);

    PairParams<T, U> params;
    for (int i = 0; i < 4; ++i) {
        params.data[i] = face_data[i];
        params.boundary[i] = face_boundary[i];
    }
    
    convert_pair_kernel<T, U><<<blocks, threads>>>(params, send_buffer, panel, coord_size, var_size, face_size);
}

template <typename T, typename U>
__global__ void convert_pair_kernel(
        PairParams<T, U> params, T* send_buffer, const int panel, const size_t coord_size, const size_t var_size, const size_t face_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int face_idx = blockIdx.y;

    if (idx >= var_size || face_idx > 3) return;
    
    // Convert coordinate for variable 1 and 2
    const TransformRule& rule = RULES[panel][face_idx];

    const T* a1 = params.data[face_idx] + 1 * var_size; // var 1
    const T* a2 = params.data[face_idx] + 2 * var_size; // var 2
    const U* coord = params.boundary[face_idx];

    T* face_base = send_buffer + face_idx * face_size;
    T* o1 = face_base + 1 * var_size;
    T* o2 = face_base + 2 * var_size;

    convert_pair_kernel_shared(a1, a2, coord, o1, o2, idx, coord_size, rule);
}


template <typename T>
void flip_axis_wrapper_gpu(
    T* send_buffer,
    const std::vector<int>& slice_shape,
    const std::vector<int>& flip_axes,
    const size_t face_size,
    const std::vector<bool>& flip_flags) {
    
    const int ndim = slice_shape.size();

    // Array strides - row major
    std::vector<size_t> stride(ndim);
    stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        stride[d] = stride[d + 1] * slice_shape[d + 1];

    int total_size = std::accumulate(slice_shape.begin(), slice_shape.end(), 1, std::multiplies<>());
    
    // Store packet - which face are flipped
    Flags flags{};
    for (int i = 0; i < 4 && i < (int) flip_flags.size(); ++i) flags.f[i] = flip_flags[i] ? 1 : 0;

    for (int axis : flip_axes) { // on which axis to flip

        // adapt to python negative array index syntax
        if (axis < 0)
            axis += ndim;

        const int dim = slice_shape[axis];
        const size_t stride_axis = stride[axis];       
        int outer_rows = total_size / (dim * stride_axis);
 
        const dim3 grid(outer_rows, 4); 
        const dim3 outer(total_size / (dim * stride_axis), 4);

        flip_axis_kernel<T><<<outer, dim/2>>>(send_buffer, dim, stride_axis, outer_rows, face_size, flags);
    }
}

template <typename T>
__global__ void flip_axis_kernel(
    T* send_buffer,
    int dim,
    const size_t stride_axis,
    const size_t outer_rows,
    const size_t face_size,
    Flags flags
) {
    const size_t outer_row = blockIdx.x;
    int i = threadIdx.x; // pair index on axis
    int face_idx = blockIdx.y;

    
    if (face_idx >= 4) return;
    if (outer_row >= outer_rows || i >= dim / 2) return;
    
    if (!flags.f[face_idx]) return;

    
    // Compute base index for this face and outer-row:
    const size_t face_offset = static_cast<size_t>(face_idx) * face_size;
    const size_t base_idx    = face_offset + outer_row * static_cast<size_t>(dim) * stride_axis;

    const size_t idx_lo  = base_idx + static_cast<size_t>(i) * stride_axis;
    const size_t idx_hi  = base_idx + (static_cast<size_t>(dim) - 1 - i) * stride_axis;

    // Swap along the axis for each contiguous lane inside the stride
    for (size_t j = 0; j < stride_axis; ++j) {
        flip_axis_kernel_shared(send_buffer, idx_lo + j, idx_hi + j);
    }
}

PYBIND11_MODULE(exchanges_cuda, m) {
    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d_wrapper);
}