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

template <typename T>
void memcpy_faces_wrapper(
    T* send_buffer,
    const T* south,
    const T* north,
    const T* west,
    const T* east,
    int block_size
) {

    // Faster than kenrel since contiguous?
    cudaMemcpyAsync(send_buffer, south, block_size * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(send_buffer + block_size, north, block_size * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(send_buffer + 2*block_size, west,  block_size * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(send_buffer + 3*block_size, east,  block_size * sizeof(T), cudaMemcpyDeviceToDevice);

}

template <typename T, typename U>
void start_exchange_euler_3d_cu (
    py::object& p_send_buffer_obj,
    py::object& p_south_obj,
    py::object& p_north_obj,
    py::object& p_west_obj,
    py::object& p_east_obj,
    py::object& p_boundary_sn_obj,
    py::object& p_boundary_we_obj,

    // reference slice shape (n_variables, n_vert, n_hori, n*n nodal pts)
    const std::vector<int>& shape,
    const std::vector<int>& flip_dims,
    const std::vector<bool>& flip_flags,
    const int panel
)
{
    
    T* p_send_buffer = get_device_ptr<T>(p_send_buffer_obj);
    const T* p_south = get_device_ptr<T>(p_south_obj);
    const T* p_north = get_device_ptr<T>(p_north_obj);
    const T* p_west  = get_device_ptr<T>(p_west_obj);
    const T* p_east  = get_device_ptr<T>(p_east_obj);
    const U* p_boundary_sn = get_device_ptr<U>(p_boundary_sn_obj);
    const U* p_boundary_we = get_device_ptr<U>(p_boundary_we_obj);


    const T* p_data[4] = {p_south, p_north, p_west, p_east};
    const U* p_boundary[4] = {p_boundary_sn, p_boundary_sn, p_boundary_we, p_boundary_we};

    
    const int n_var = shape[0];
    const int var_size = std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<>());
    const int n_coord = std::accumulate(shape.begin() + 2, shape.end(), 1, std::multiplies<>());
    const int block_size = n_var * var_size;

    memcpy_faces_wrapper<T>(p_send_buffer, p_south, p_north, p_west, p_east, block_size);

    convert_pair_wrapper_gpu<T, U>(p_data, p_boundary, p_send_buffer, block_size, panel, n_coord, var_size);

    flip_axis_wrapper_gpu<T>(p_send_buffer, shape, flip_dims, block_size, flip_flags);

    // for (int i = 0; i < 4; i++) {
    //     if (flip_flags[i]) {
    //         flip_axis_wrapper_gpu<T>(p_send_buffer + i * block_size, shape, flip_dims);
    //     }
    // }
}

void start_exchange_euler_3d_wrapper(
    py::object& p_send_buffer,
    py::object& p_south,
    py::object& p_north,
    py::object& p_west,
    py::object& p_east,
    py::object& p_boundary_sn,
    py::object& p_boundary_we,

    // reference slice shape (n_variables, n_vert, n_hori, n*n nodal pts)
    const std::vector<int>& shape,
    const std::vector<int>& flip_dims,
    const std::vector<bool>& flip_flags,
    const int panel
)
{

    std::string T_type = py::str(p_send_buffer.attr("dtype").attr("name"));
    std::string U_type = py::str(p_boundary_sn.attr("dtype").attr("name"));

    if (T_type == "float64") {
        start_exchange_euler_3d_cu<double, double>(
            p_send_buffer,
            p_south,
            p_north,
            p_west,
            p_east,
            p_boundary_sn,
            p_boundary_we,

            // reference slice shape (n_variables, n_vert, n_hori, n*n nodal pts)
            shape,
            flip_dims,
            flip_flags,
            panel
        );
    }
    else if (T_type == "complex128") {
        start_exchange_euler_3d_cu<complex_t, double>(
            p_send_buffer,
            p_south,
            p_north,
            p_west,
            p_east,
            p_boundary_sn,
            p_boundary_we,

            // reference slice shape (n_variables, n_vert, n_hori, n*n nodal pts)
            shape,
            flip_dims,
            flip_flags,
            panel
        );
    }
}

template <typename T, typename U>
void convert_pair_wrapper_gpu(
    const T* p_data[4], const U* p_boundary[4],
    T* p_send_buffer,
    int block_size,
    int panel,
    int n_coord, int var_size
) {

    const T** d_p_data     = nullptr;
    const U** d_p_boundary = nullptr;
    cudaMalloc(&d_p_data,     4 * sizeof(T*));
    cudaMalloc(&d_p_boundary, 4 * sizeof(U*));

    cudaMemcpy(d_p_data, p_data, 4 * sizeof(T*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_boundary, p_boundary, 4 * sizeof(U*), cudaMemcpyHostToDevice);


    int BLOCK_SIZE = 128;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((var_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 4);
    
    convert_pair_kernel<T, U><<<blocks, threads>>>(d_p_data, d_p_boundary, p_send_buffer, panel, n_coord, var_size, block_size);

    cudaFree(d_p_data);
    cudaFree(d_p_boundary);
}

template <typename T, typename U>
__global__ void convert_pair_kernel(
    const T* const* p_data,
    const U* const* p_boundary, T* p_send_buffer, int panel, int n_coord, int var_size, int block_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int face = blockIdx.y;

    if (idx >= var_size || face > 3) return;
    
    const TransformRule& rule = rules[panel][face];
    
    const T* a1 = p_data[face] + 1 * var_size;
    const T* a2 = p_data[face] + 2 * var_size;
    const U* coord = p_boundary[face];

    T* base = p_send_buffer + face * block_size;
    T* o1 = base + 1 * var_size;
    T* o2 = base + 2 * var_size;

    convert_pair_kernel_shared(a1, a2, coord, o1, o2, idx, n_coord, rule);
}

template <typename T>
__global__ void flip_axis_kernel(
    T* arr,
    int total_size,
    int dim,
    int stride_axis,
    int outer_rows,
    int block_size,
    const bool* flip_flags
) {
    int o = blockIdx.x;
    int i = threadIdx.x;
    int face = blockIdx.y;

    
    if (face >= 4) return;
    if (o >= outer_rows || i >= dim / 2) return;


    if (flip_flags && !flip_flags[face]) return;


    int base_idx = o * dim * stride_axis;
    int idx = base_idx + i * stride_axis;
    int idx_opp = base_idx + (dim - 1 - i) * stride_axis;

    for (int j = 0; j < stride_axis; ++j) {
        flip_axis_kernel_shared(arr + block_size*face, idx + j, idx_opp + j);
    }
}

template <typename T>
void flip_axis_wrapper_gpu(T* d_arr, const std::vector<int>& shape, const std::vector<int>& axes, int block_size, const std::vector<bool>& flip_flags) {
    int ndim = shape.size();

    // strides
    std::vector<int> stride(ndim);
    stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        stride[d] = stride[d + 1] * shape[d + 1];

    int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    
    bool h_flags[4] = {true, true, true, true};
    if (flip_flags.size() == 4) {
        for (int i = 0; i < 4; ++i) h_flags[i] = flip_flags[i];
    }
    bool* d_flags = nullptr;
    cudaCheck(cudaMalloc(&d_flags, 4 * sizeof(bool)));
    cudaCheck(cudaMemcpy(d_flags, h_flags, 4 * sizeof(bool), cudaMemcpyHostToDevice));

    for (int axis : axes) {
        if (axis < 0)
            axis += ndim;

        int dim = shape[axis];
        int stride_axis = stride[axis];
        
        int outer_rows = total_size / (dim * stride_axis);
        dim3 grid(outer_rows, 4); 

        dim3 outer(total_size / (dim * stride_axis), 4);

        flip_axis_kernel<T><<<outer, dim/2>>>(d_arr, total_size, dim, stride_axis, outer_rows, block_size, d_flags);
        cudaDeviceSynchronize();
    }
}

PYBIND11_MODULE(exchanges_cuda, m) {
    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d_wrapper);
}