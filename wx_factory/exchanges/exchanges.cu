#include <iostream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include <pybind11/complex.h>

// #include "kernels/kernels.h"

#include <vector>
#include <cuda_runtime.h>

#include "exchanges.hpp"

__constant__ TransformRule rules_c[6][4];


namespace py = pybind11;

template <typename T>
T* get_device_ptr(py::object& obj) {
    auto iface = obj.attr("__cuda_array_interface__");
    auto data_tuple = iface["data"].cast<py::tuple>();
    uintptr_t ptr_value = data_tuple[0].cast<uintptr_t>();
    return reinterpret_cast<T*>(ptr_value);
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
    const size_t block_size = static_cast<size_t>(n_var) * var_size;

    
    memcpy_faces_wrapper<T>(p_send_buffer, p_south, p_north, p_west, p_east, block_size);

    for (int i = 0; i < 4; ++i) {

        // T* a1 = const_cast<T*>(p_data[i]) + 1 * var_size;
        // T* a2 = const_cast<T*>(p_data[i]) + 2 * var_size;
        const T* a1 = p_data[i] + 1 * var_size;
        const T* a2 = p_data[i] + 2 * var_size;

        const U* coord = p_boundary[i];

        T* o1 = p_send_buffer + i * block_size + 1 * var_size;
        T* o2 = p_send_buffer + i * block_size + 2 * var_size;

        convert_pair_wrapper<T, U>(a1, a2, coord, o1, o2, panel, i, n_coord, var_size);

        if (flip_flags[i]) {
            flip_axis_wrapper<T>(p_send_buffer + i * block_size, shape, flip_dims);
        }
    }
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


template <typename T>
void memcpy_faces_wrapper(
    T* send_buffer,
    const T* south,
    const T* north,
    const T* west,
    const T* east,
    int block_size
) {

    const int BLOCK_SIZE = 128; // ** Distinction BLOCK_SIZE vs block_size **
    const int total = 4 * block_size;
    const int NUM_BLOCKS = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    memcpy_faces_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>(send_buffer, south, north, west, east, block_size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(std::string("memcpy_faces_kernel launch: ") + cudaGetErrorString(err));
    cudaCheck(cudaDeviceSynchronize());
}

template <typename T>
__global__ void memcpy_faces_kernel(
    T* send_buffer,
    const T* south,
    const T* north,
    const T* west,
    const T* east,
    int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 4 * block_size;
    if (idx >= total) return; // guard

    int face = idx / block_size;
    int offset = idx % block_size;

    const T* src = (face == 0) ? south :
                   (face == 1) ? north :
                   (face == 2) ? west  :
                   east;

    send_buffer[face * block_size + offset] = src[offset];
}

template <typename T, typename U>
void convert_pair_wrapper(const T* a1, const T* a2, const U* coord, T* o1, T* o2, int panel, int neighbour, int n_coord, int var_size) {
    
    int BLOCK_SIZE = 128;
    const int NUM_BLOCKS = (var_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // cudaError_t err = cudaMemcpyToSymbol(
    //     rules_c,
    //     rules,
    //     sizeof(rules), 0, cudaMemcpyHostToDevice
    // );
    cudaError_t err = cudaMemcpyToSymbol(rules_c, rules, sizeof(rules));

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to copy rules to GPU: ") +
                                 cudaGetErrorString(err));
    }
    convert_pair_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>> (
        a1, a2, coord, o1, o2,
        panel, neighbour, n_coord, var_size
    );
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) throw std::runtime_error(std::string("convert_pair_kernel launch: ") + cudaGetErrorString(err));
    cudaCheck(cudaDeviceSynchronize());
}

template <typename T, typename U>
__global__ void convert_pair_kernel(const T* a1, const T* a2, const U* coord, T* o1, T* o2, int panel, int neighbour, int n_coord, int var_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= var_size) return; // guard

    const TransformRule rule = rules_c[panel][neighbour];
    // if (idx == 0) {
    //     printf("GPU rule check: panel=%d neighbour=%d\n", panel, neighbour);
    //     printf("  s11=%d s12=%d s13=%d s14=%d | s21=%d s22=%d s23=%d s24=%d\n",
    //            rule.s11, rule.s12, rule.s13, rule.s14,
    //            rule.s21, rule.s22, rule.s23, rule.s24);
    // }

    // This can be precomputed for each coord, but first check individual thread performance (see cpu equivalent)
    U x = coord[idx % n_coord];
    U c = (2.0 * x) / (1.0 + x*x);

    T p11 = rule.s11 + c * rule.s13;
    T p12 = rule.s12 + c * rule.s14;
    T p21 = rule.s21 + c * rule.s23;
    T p22 = rule.s22 + c * rule.s24;
    // ---

    T A1 = a1[idx];
    T A2 = a2[idx];

    o1[idx] = p11 * A1 + p12 * A2;
    o2[idx] = p21 * A1 + p22 * A2;

    // if (idx < 4) {
    //     printf("idx=%d x=%f c=%f A1=(%f,%f) A2=(%f,%f) o1=(%f,%f) o2=(%f,%f)\n",
    //            idx, (double)x, (double)c,
    //            (double)A1.x, (double)A1.y, (double)A2.x, (double)A2.y,
    //            (double)o1[idx].x, (double)o1[idx].y,
    //            (double)o2[idx].x, (double)o2[idx].y);
    // }
}

template <typename T>
void flip_axis_wrapper(
    T* arr,
    const std::vector<int>& shape,
    const std::vector<int>& axes
) {
    const int ndim = shape.size();
    // if (ndim == 0) return;

    // stride per dimension
    std::vector<int> stride(ndim);
    stride[ndim-1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        stride[d] = stride[d+1] * shape[d+1];
    }
    
    int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

    // Trasnfer dims information to cuda
    int *shape_c, *stride_c;
    int info_size = ndim * sizeof(int);
    // cudaMalloc(&shape_c, info_size);
    // cudaMalloc(&stride_c, info_size);
    // cudaMemcpy(shape_c, &shape, info_size, cudaMemcpyHostToDevice); // .data() instead?
    // cudaMemcpy(stride_c, &stride, info_size, cudaMemcpyHostToDevice);
    cudaMalloc(&shape_c, info_size);
    cudaMalloc(&stride_c, info_size);
    cudaMemcpy(shape_c, shape.data(), info_size, cudaMemcpyHostToDevice);
    cudaMemcpy(stride_c, stride.data(), info_size, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 128;
    const int NUM_BLOCKS = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int ax : axes) {
        int axis = ax < 0 ? ax + ndim : ax; // python negative format
        flip_axis_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>(arr, total_size, shape_c, stride_c, ndim, axis);

    }

    cudaDeviceSynchronize();

    cudaFree(shape_c);
    cudaFree(stride_c);
}

template <typename T>
__global__ void flip_axis_kernel(
    T* arr,
    int total_size,
    const int* shape,
    const int* stride,
    int ndim,
    int axis
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return; // guard

    int dim = shape[axis];
    int stride_axis = stride[axis];

    int pos = (idx / stride_axis) % dim;
    if (pos >= dim / 2) return;


    int pos_opp = dim - 1 - pos;

    int idx_opp = idx + (pos_opp - pos) * stride_axis;
    
    T tmp = arr[idx];
    arr[idx] = arr[idx_opp];
    arr[idx_opp] = tmp;

}


PYBIND11_MODULE(exchanges_cuda, m) {

    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d_wrapper);
    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d_wrapper);
    // m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d<double, double>,
    //     py::arg("send_buffer"),
    //     py::arg("south"),
    //     py::arg("north"),
    //     py::arg("west"),
    //     py::arg("east"),
    //     py::arg("boundary_sn"),
    //     py::arg("boundary_we"),
    //     py::arg("shape"),
    //     py::arg("flip_dim"),
    //     py::arg("flip_flags"),
    //     py::arg("panel"),
    //     "hpp version of euler buffers packing"
    // );

    // m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d<std::complex<double>, double>,
    //     py::arg("send_buffer"),
    //     py::arg("south"),
    //     py::arg("north"),
    //     py::arg("west"),
    //     py::arg("east"),
    //     py::arg("boundary_sn"),
    //     py::arg("boundary_we"),
    //     py::arg("shape"),
    //     py::arg("flip_dim"),
    //     py::arg("flip_flags"),
    //     py::arg("panel"),
    //     "hpp version of euler buffers packing"
    // );
}