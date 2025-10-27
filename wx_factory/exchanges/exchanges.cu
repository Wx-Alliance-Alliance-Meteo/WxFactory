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

// __constant__ TransformRule rules_c[6][4];


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
)
{
    T* p_send_buffer = static_cast<T*>(send_buffer.request().ptr);
    T* p_south = static_cast<T*>(south.request().ptr);
    T* p_north = static_cast<T*>(north.request().ptr);
    T* p_west = static_cast<T*>(west.request().ptr);
    T* p_east = static_cast<T*>(east.request().ptr);
    U* p_boundary_sn = static_cast<U*>(boundary_sn.request().ptr);
    U* p_boundary_we = static_cast<U*>(boundary_we.request().ptr);

    T* p_data[4] = {p_south, p_north, p_west, p_east};
    U* p_boundary[4] = {p_boundary_sn, p_boundary_sn, p_boundary_we, p_boundary_we};

    
    const int n_var = shape[0];
    const int var_size = std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<>());
    const int n_coord = std::accumulate(shape.begin() + 2, shape.end(), 1, std::multiplies<>());
    const size_t block_size = static_cast<size_t>(n_var) * var_size;

    T *send_buffer_c, *south_c, *north_c, *west_c, *east_c;
    cudaCheck(cudaMalloc(&send_buffer_c,  4 * block_size * sizeof(T)));
    cudaCheck(cudaMalloc(&south_c, block_size * sizeof(T)));
    cudaCheck(cudaMalloc(&north_c, block_size * sizeof(T)));
    cudaCheck(cudaMalloc(&west_c,  block_size * sizeof(T)));
    cudaCheck(cudaMalloc(&east_c,  block_size * sizeof(T)));

    cudaCheck(cudaMemcpy(south_c, p_south, block_size * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(north_c, p_north, block_size * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(west_c,  p_west,  block_size * sizeof(T), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(east_c,  p_east,  block_size * sizeof(T), cudaMemcpyHostToDevice));


    memcpy_faces_wrapper(send_buffer_c, south_c, north_c, west_c, east_c, block_size);

    cudaCheck(cudaMemcpy(p_send_buffer, send_buffer_c, 4 * block_size * sizeof(T), cudaMemcpyDeviceToHost));

    cudaFree(send_buffer_c);
    cudaFree(south_c);
    cudaFree(north_c);
    cudaFree(west_c);
    cudaFree(east_c);

    // int total_elements = 4 * n_var * var_size;

    // const int BLOCK_SIZE = 128; // gpu block size, note all caps
    // const int total = 4 * block_size;
    // const int NUM_BLOCKS = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // memcpy_faces_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>(d_send, d_south, d_north, d_west, d_east, block_size);

    // // memory copy wrapper call

    // for (int i = 0; i < 4; ++i) {

    //     // allocation

    //     std::memcpy(p_send_buffer + i * block_size, p_data[i], block_size*sizeof(T));

    //     convert_pair_wrapper(a1, a2, coord, o1, o2, panel, i, n_coord, var_size);

    //     if (flip_flags[i]) {
    //         flip_axis_wrapper(arr, shape, flip_dims);
    //     }
    // }
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
    
    T* a1_c, a2_c, o1_c, o2_c;
    U* coord_c;

    cudaMalloc(&a1_c, var_size * sizeof(T));
    cudaMalloc(&a2_c, var_size * sizeof(T));
    cudaMalloc(&coord_c, n_coord * sizeof(U));
    cudaMalloc(&o1_c, var_size * sizeof(T));
    cudaMalloc(&o2_c, var_size * sizeof(T));

    cudaMemcpy(a1_c, a1, var_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(a2_c, a2, var_size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(coord_c, coord, n_coord * sizeof(U), cudaMemcpyHostToDevice);

    // Check for 
    static bool rules_copied = false;
    if (!rules_copied) {
        cudaMemcpyToSymbol(rules_c, rules, sizeof(rules));
        rules_copied = true;
    }

    int BLOCK_SIZE = 128;
    const int NUM_BLOCKS = (var_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    convert_pair_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>> (
        a1_c, a2_c, coord_c, o1_c, o2_c,
        panel, neighbour, n_coord, var_size
    );
    cudaDeviceSynchronize();

    // Can allocate directly on host?
    cudaMemcpy(o1, o1_c, var_size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(o2, o2_c, var_size * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(a1_c);
    cudaFree(a2_c);
    cudaFree(coord_c);
    cudaFree(o1_c);
    cudaFree(o2_c);

}


template <typename T, typename U>
__global__ void convert_pair_kernel(const T* a1, const T* a2, const U* coord, T* o1, T* o2, int panel, int neighbour, int n_coord, int var_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= var_size) return; // guard

    const TransformRule rule = rules_c[panel][neighbour];
    
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

}

template <typename T>
void flip_axis_wrapper(
    T* arr,
    const std::vector<int>& shape,
    const std::vector<int>& axes
) {
    const int ndim = shape.size();

    // stride per dimension
    std::vector<int> stride(ndim);
    stride[ndim-1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        stride[d] = stride[d+1] * shape[d+1];
    }
    
    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

    // Trasnfer dims information to cuda
    int *shape_c, *stride_c;
    size_t info_size = ndim * sizeof(size_t);
    cudaMalloc(&shape_c, info_size);
    cudaMalloc(&stride_c, info_size);
    cudaMemcpy(shape_c, &shape, info_size, cudaMemcpyHostToDevice); // .data() instead?
    cudaMemcpy(stride_c, &stride, info_size, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 128;
    const int NUM_BLOCKS = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int ax : axes) {
        int axis = ax < 0 ? ax + ndim : ax; // python negative format
        flip_axis_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>(arr, shape_c, stride_c, ndim, axis, total_size);
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

    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d<double, double>,
        py::arg("send_buffer"),
        py::arg("south"),
        py::arg("north"),
        py::arg("west"),
        py::arg("east"),
        py::arg("boundary_sn"),
        py::arg("boundary_we"),
        py::arg("shape"),
        py::arg("flip_dim"),
        py::arg("flip_flags"),
        py::arg("panel"),
        "hpp version of euler buffers packing"
    );

    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d<std::complex<double>, double>,
        py::arg("send_buffer"),
        py::arg("south"),
        py::arg("north"),
        py::arg("west"),
        py::arg("east"),
        py::arg("boundary_sn"),
        py::arg("boundary_we"),
        py::arg("shape"),
        py::arg("flip_dim"),
        py::arg("flip_flags"),
        py::arg("panel"),
        "hpp version of euler buffers packing"
    );
}