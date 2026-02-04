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


template <typename num_t>
num_t* get_device_ptr(py::object& obj) {
    auto iface = obj.attr("__cuda_array_interface__");
    auto data_tuple = iface["data"].cast<py::tuple>();
    uintptr_t ptr_value = data_tuple[0].cast<uintptr_t>();
    return reinterpret_cast<num_t*>(ptr_value);
}

template <typename num_t, typename real_t>
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
    num_t* send_buffer = get_device_ptr<num_t>(send_buffer_obj);
    const num_t* south = get_device_ptr<num_t>(south_obj);
    const num_t* north = get_device_ptr<num_t>(north_obj);
    const num_t* west  = get_device_ptr<num_t>(west_obj);
    const num_t* east  = get_device_ptr<num_t>(east_obj);

    const real_t* boundary_sn = get_device_ptr<real_t>(boundary_sn_obj);
    const real_t* boundary_we = get_device_ptr<real_t>(boundary_we_obj);

    const num_t* face_data[4] = { south, north, west, east };
    const real_t* face_boundary[4] = { boundary_sn, boundary_sn, boundary_we, boundary_we };

    const int n_var = slice_shape[0]; // number of variables, e.g. temperature, density, etc
    const size_t var_size = std::accumulate(slice_shape.begin() + 1, slice_shape.end(), 1, std::multiplies<>()); // elements per variable
    const size_t coord_size = std::accumulate(slice_shape.begin() + 2, slice_shape.end(), 1, std::multiplies<>()); // size per vertical coordinate
    const size_t face_size = n_var * var_size; // total size face

    memcpy_faces_wrapper<num_t>(send_buffer, south, north, west, east, face_size);

    convert_pair_wrapper_gpu<num_t, real_t>(face_data, face_boundary, send_buffer, face_size, panel, coord_size, var_size);

    flip_axis_wrapper_gpu<num_t>(send_buffer, slice_shape, flip_axes, face_size, flip_flags);
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

    std::string num_t = py::str(send_buffer.attr("dtype").attr("name"));

    // Different template according to buffer type
    if (num_t == "float64") {
        start_exchange_euler_3d<double, double>(
            send_buffer,
            south, north, west, east,
            boundary_sn, boundary_we,
            slice_shape, flip_axes, flip_flags, panel
        );
    }
    else if (num_t == "complex128") {
        start_exchange_euler_3d<complex_t, double>(
            send_buffer,
            south, north, west, east,
            boundary_sn, boundary_we,
            slice_shape, flip_axes, flip_flags, panel
        );
    }
    else {
        throw std::runtime_error("Unsupported buffer type: " + num_t);
    }
}

// Copy 4 face buffers into 1 contigous buffer
template <typename num_t>
void memcpy_faces_wrapper(
    num_t* send_buffer,
    const num_t* south,
    const num_t* north,
    const num_t* west,
    const num_t* east,
    size_t face_size
) {

    const int BLOCK_SIZE = 128;
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((face_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 4);

    memcpy_faces_kernel<<<blocks, threads>>>(send_buffer, south, north, west, east, face_size);
}

template <typename num_t>
__global__ void memcpy_faces_kernel(
    num_t* send_buffer,
    const num_t* south,
    const num_t* north,
    const num_t* west,
    const num_t* east,
    size_t face_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t face_id = blockIdx.y;

    if (idx < face_size) {
        const num_t* src;
        switch (face_id) {
            case 0: src = south; break;
            case 1: src = north; break;
            case 2: src = west;  break;
            case 3: src = east;  break;
        }
        send_buffer[face_id * face_size + idx] = src[idx];
    }
}


template <typename num_t, typename real_t>
void convert_pair_wrapper_gpu(
    const num_t* face_data[4],
    const real_t* face_boundary[4],
    num_t* send_buffer,
    const size_t face_size,
    const int panel,
    const size_t coord_size,
    const size_t var_size
) {

    
    const int BLOCK_SIZE = 128;
    const dim3 threads(BLOCK_SIZE);
    const dim3 blocks((var_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 4);

    PairParams<num_t, real_t> params;
    for (int i = 0; i < 4; ++i) {
        params.data[i] = face_data[i];
        params.boundary[i] = face_boundary[i];
    }
    
    convert_pair_kernel<num_t, real_t><<<blocks, threads>>>(params, send_buffer, panel, coord_size, var_size, face_size);
}

template <typename num_t, typename real_t>
__global__ void convert_pair_kernel(
        PairParams<num_t, real_t> params, num_t* send_buffer, const int panel, const size_t coord_size, const size_t var_size, const size_t face_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int face_idx = blockIdx.y;

    if (idx >= var_size || face_idx > 3) return;
    
    // Convert coordinate for variable 1 and 2
    const TransformRule& rule = RULES[panel][face_idx];

    const num_t* a1 = params.data[face_idx] + 1 * var_size; // var 1
    const num_t* a2 = params.data[face_idx] + 2 * var_size; // var 2
    const real_t* coord = params.boundary[face_idx];

    num_t* face_base = send_buffer + face_idx * face_size;
    num_t* o1 = face_base + 1 * var_size;
    num_t* o2 = face_base + 2 * var_size;

    convert_pair_kernel_shared(a1, a2, coord, o1, o2, idx, coord_size, rule);
}


template <typename num_t>
void flip_axis_wrapper_gpu(
    num_t* send_buffer,
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

        flip_axis_kernel<num_t><<<outer, dim/2>>>(send_buffer, dim, stride_axis, outer_rows, face_size, flags);
    }
}

template <typename num_t>
__global__ void flip_axis_kernel(
    num_t* send_buffer,
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
    init_transform_rules_cuda();
    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d_wrapper);
}