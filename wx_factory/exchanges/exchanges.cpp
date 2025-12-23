#include <iostream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include <pybind11/complex.h>

#include "exchanges.hpp"
#include "kernels/kernels.h"

namespace py = pybind11;

template<typename num_t>
void flip_axis_wrapper_cpu(num_t* arr, const std::vector<int>& shape, const std::vector<int>& axes) {
    const int ndim = shape.size();

    std::vector<int> stride(ndim);
    stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        stride[d] = stride[d + 1] * shape[d + 1];
    }

    int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

    for (int axis : axes) {
        // convert python negative format
        if (axis < 0) {
            axis += ndim;
        }

        int dim = shape[axis];
        int stride_axis = stride[axis];
        int outer = total_size / (dim * stride_axis);

        for (int o = 0; o < outer; ++o) {
            int base_idx = o * dim * stride_axis;
            for (int i = 0; i < dim / 2; ++i) {
                int idx = base_idx + i * stride_axis;
                int idx_opp = base_idx + (dim - 1 - i) * stride_axis;
                for (int j = 0; j < stride_axis; ++j) {
                    flip_axis_kernel_shared(arr, idx + j, idx_opp + j);
                }
            }
        }
    }
}

template <typename num_t, typename real_t>
void convert_pair_wrapper_cpu(
    const num_t* p_data_face, const real_t* p_boundary_face,
    num_t* p_send_buffer,
    int panel, int neighbour,
    int n_coord, int var_size
) {


    const TransformRule& rule = RULES[panel][neighbour];

    const num_t* a1 = p_data_face + 1 * var_size;
    const num_t* a2 = p_data_face + 2 * var_size;
    const real_t* coord = p_boundary_face;

    num_t* o1 = p_send_buffer + 1 * var_size;
    num_t* o2 = p_send_buffer + 2 * var_size;

    for (int i = 0; i < var_size; ++i) {
        convert_pair_kernel_shared(a1, a2, coord, o1, o2, i, n_coord, rule);
    }
}


/*
We allocate the buffers for ghost cells between panels (called for one panel)
For each neighbour, the slice corresponds to a z * (x or y) vertical-horizontal slice

4 faces, 5 variables
*/
template <typename num_t, typename real_t>
void start_exchange_euler_3d_cpp(
    py::array_t<num_t, py::array::c_style> send_buffer,
    py::array_t<num_t, py::array::c_style> south,
    py::array_t<num_t, py::array::c_style> north,
    py::array_t<num_t, py::array::c_style> west,
    py::array_t<num_t, py::array::c_style> east,
    py::array_t<real_t, py::array::c_style> boundary_sn,
    py::array_t<real_t, py::array::c_style> boundary_we,

    // reference slice shape (n_variables, n_vert, n_hori, n*n nodal pts)
    const std::vector<int>& shape,
    const std::vector<int>& flip_dims,
    const std::vector<bool>& flip_flags,
    const int panel
)
{

    // pointer unpacking
    num_t* p_send_buffer = static_cast<num_t*>(send_buffer.request().ptr);
    num_t* p_south = static_cast<num_t*>(south.request().ptr);
    num_t* p_north = static_cast<num_t*>(north.request().ptr);
    num_t* p_west = static_cast<num_t*>(west.request().ptr);
    num_t* p_east = static_cast<num_t*>(east.request().ptr);
    real_t* p_boundary_sn = static_cast<real_t*>(boundary_sn.request().ptr); // Different type
    real_t* p_boundary_we = static_cast<real_t*>(boundary_we.request().ptr);

    num_t* p_data[4] = {p_south, p_north, p_west, p_east};
    real_t* p_boundary[4] = {p_boundary_sn, p_boundary_sn, p_boundary_we, p_boundary_we};

    
    const int n_var = shape[0];
    const int var_size = std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<>());
    const int n_coord = std::accumulate(shape.begin() + 2, shape.end(), 1, std::multiplies<>());
    const size_t block_size = static_cast<size_t>(n_var) * var_size;
    
    for (int i = 0; i < 4; ++i) {

        // allocation
        std::memcpy(p_send_buffer + i * block_size, p_data[i], block_size*sizeof(num_t));


        const num_t* p_data_face = p_data[i];
        const real_t* p_boundary_face = p_boundary[i];
        num_t* p_send_buffer_face = p_send_buffer + i * block_size;

        // Convert pairs - transformation to

        p_data_face = p_data[i];
        p_boundary_face = p_boundary[i];
        convert_pair_wrapper_cpu(p_data_face, p_boundary_face, p_send_buffer_face, panel, i, n_coord, var_size);

        if (flip_flags[i]) {
            flip_axis_wrapper_cpu(p_send_buffer + i * block_size, shape, flip_dims);
        }
    } // for

}


/*
Bindings python - cpp
Ref: https://pybind11.readthedocs.io/en/stable/basics.html
The first argument "kernel_template_cpp" here needs to correspond to the one compiled in compile_kernels.py/device.py
*/

PYBIND11_MODULE(exchanges_cpp, m) {

    // ** Keep double first as double can overload to complex
    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d_cpp<double, double>,
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

    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d_cpp<std::complex<double>, double>,
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