#include <iostream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include <pybind11/complex.h>

#include <typeinfo>

#include "kernels/kernels.h"

namespace py = pybind11;

// template <typename T>
// T* allocate_buffer(const std::vector<int>& shape);

// T - match np dtype
// c_style - row-major C

/*
We allocate the buffers for ghost cells between panels (called for one panel)
For each neighbour, the slice corresponds to a z * (x or y) vertical-horizontal slice

4 faces, 5 variables
*/
template <typename T, typename U>
void start_exchange_euler_3d_cpp(
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

    
    // pointer unpacking
    // TODO: .mutable_data() ?
    T* p_send_buffer = static_cast<T*>(send_buffer.request().ptr);
    T* p_south = static_cast<T*>(south.request().ptr);
    T* p_north = static_cast<T*>(north.request().ptr);
    T* p_west = static_cast<T*>(west.request().ptr);
    T* p_east = static_cast<T*>(east.request().ptr);
    U* p_boundary_sn = static_cast<U*>(boundary_sn.request().ptr); // Different type
    U* p_boundary_we = static_cast<U*>(boundary_we.request().ptr);

    T* p_data[4] = {p_south, p_north, p_west, p_east};
    U* p_boundary[4] = {p_boundary_sn, p_boundary_sn, p_boundary_we, p_boundary_we};

    
    const int n_var = shape[0];
    const int var_size = std::accumulate(shape.begin() + 1, shape.end(), 1, std::multiplies<>());
    const int n_coord = std::accumulate(shape.begin() + 2, shape.end(), 1, std::multiplies<>());
    const size_t block_size = static_cast<size_t>(n_var) * var_size;

    int total_elements = 4 * n_var * var_size;

    
    for (int i = 0; i < 4; ++i) {

        // allocation
        std::memcpy(p_send_buffer + i * block_size, p_data[i], block_size*sizeof(T));

        // Convert pairs - transformation to

        T* a1 = p_data[i] + 1 * var_size;
        T* a2 = p_data[i] + 2 * var_size;

        U* coord = p_boundary[i];

        T* o1 = p_send_buffer + i * block_size + 1 * var_size;
        T* o2 = p_send_buffer + i * block_size + 2 * var_size;

        convert_pair(a1, a2, coord, o1, o2, panel, i, n_coord, var_size);

    if (flip_flags[i]) {


        flip_axis(p_send_buffer + i * block_size, shape, flip_dims);

        }
    }
}


/*
Bindings python - cpp
Ref: https://pybind11.readthedocs.io/en/stable/basics.html
The first argument "kernel_template_cpp" here needs to correspond to the one compiled in compile_kernels.py/device.py
*/
PYBIND11_MODULE(kernel_template_cpp, m) {

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