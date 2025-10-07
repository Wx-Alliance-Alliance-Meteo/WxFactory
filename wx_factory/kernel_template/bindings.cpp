#include <iostream>
#include <string>

#include "bindings.hpp"

namespace py = pybind11;

// T - match np dtype
// c_style - row-major C
template <typename T>
py::tuple start_exchange_euler_3d_cpp(
    py::array_t<T, py::array::c_style> south,
    py::array_t<T, py::array::c_style> north,
    py::array_t<T, py::array::c_style> west,
    py::array_t<T, py::array::c_style> east,
    py::array_t<T, py::array::c_style> boundary_sn,
    py::array_t<T, py::array::c_style> boundary_we,
    const std::vector<int>& shape,
    const std::vector<int>& flip_dims,
    bool covariant // further use
)
{

    T* p_south = static_cast<T*>(south.request().ptr);
    T* p_north = static_cast<T*>(north.request().ptr);
    T* p_west = static_cast<T*>(west.request().ptr);
    T* p_east = static_cast<T*>(east.request().ptr);
    T* p_boundary_sn = static_cast<T*>(p_boundary_sn.request().ptr);
    T* p_boundary_we = static_cast<T*>(boundary_we.request().ptr);

    

    std::vector<T> send_buffer = allocate_buffer<T>(shape);

    convert_pair(ptr_south_1, ptr_south_2, ptr_boundary_sn, o_south_1, o_south_2,
             panel_id, 0, n); // 0 = South
    convert_pair(ptr_north_1, ptr_north_2, ptr_boundary_sn, o_north_1, o_north_2,
             panel_id, 1, n); // 1 = North
    convert_pair(ptr_west_1,  ptr_west_2,  ptr_boundary_we, o_west_1,  o_west_2,
             panel_id, 2, n); // 2 = West
    convert_pair(ptr_east_1,  ptr_east_2,  ptr_boundary_we, o_east_1,  o_east_2,
             panel_id, 3, n); // 3 = East
}

// Create buffer
template <typename T>
std::vector<T> allocate_buffer(const std::vector<int>& shape) {
    const int n_faces = 4;
    const int total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    std::vector<T> buffer(n_faces * total);
    return buffer;
}

/*
Bindings python - cpp 
Ref: https://pybind11.readthedocs.io/en/stable/basics.html
The first argument "kernel_template_cpp" here needs to correspond to the one compiled in compile_kernels.py/device.py
*/
PYBIND11_MODULE(kernel_template_cpp, m) {

    m.def("start_exchange_euler_3d_cpp", &start_exchange_euler_3d_cpp<float>,
        py::arg("south"),
        py::arg("north"),
        py::arg("west"),
        py::arg("east"),
        py::arg("boundary_sn"),
        py::arg("boundary_we"),
        py::arg("flip_dim") = -1,
        "hpp version of euler buffers packing"
    );
}