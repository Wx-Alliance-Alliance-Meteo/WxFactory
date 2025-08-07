#include "operators.hpp"

#include <iostream>

#include <pybind11/stl.h>

#include "kernels/kernels.h"

namespace py = pybind11;

#ifdef WX_OMP
template <typename T>
using py_array = py::object;
#define MODULE_NAME operators_omp
#else
template <typename T>
using py_array = py::array_t<T>;
#define MODULE_NAME operators_cpp
#endif

template <typename Func>
void extrap_3d(const size_t num_threads, const int verbose, Func func) {
#pragma omp target teams distribute
  for (size_t index = 0; index < num_threads; index++)
  {
    func.params.set_index(index);
    func(nullptr, verbose);
  }
}

template <typename real_t, typename num_t>
void select_extrap_all_3d(
    const py_array<num_t>& q_in,
    py_array<num_t>&       result_x_in,
    py_array<num_t>&       result_y_in,
    py_array<num_t>&       result_z_in,
    const int              verbose) {

  const auto shape       = q_in.request().shape;
  const int  num_elem_x3 = shape[1];
  const int  num_elem_x2 = shape[2];
  const int  num_elem_x1 = shape[3];
  const int  num_solpts  = static_cast<int>(std::cbrt(shape[4]));

  const size_t num_threads =
      5 * num_elem_x1 * num_elem_x2 * num_elem_x3 * num_solpts * num_solpts;
  switch (num_solpts)
  {
    // clang-format off
  case 1: extrap_3d(num_threads, verbose, extrap_all_kernel<real_t, num_t, 1>(q_in, result_x_in, result_y_in, result_z_in)); break;
  case 2: extrap_3d(num_threads, verbose, extrap_all_kernel<real_t, num_t, 2>(q_in, result_x_in, result_y_in, result_z_in)); break;
  case 3: extrap_3d(num_threads, verbose, extrap_all_kernel<real_t, num_t, 3>(q_in, result_x_in, result_y_in, result_z_in)); break;
  case 4: extrap_3d(num_threads, verbose, extrap_all_kernel<real_t, num_t, 4>(q_in, result_x_in, result_y_in, result_z_in)); break;
  case 5: extrap_3d(num_threads, verbose, extrap_all_kernel<real_t, num_t, 5>(q_in, result_x_in, result_y_in, result_z_in)); break;
  case 6: extrap_3d(num_threads, verbose, extrap_all_kernel<real_t, num_t, 6>(q_in, result_x_in, result_y_in, result_z_in)); break;
  default: std::cerr << __func__ << ": Not implemented for order " << num_solpts << "\n"; break;
    // clang-format on
  }
}

#ifdef WX_OMP
void select_extrap_all_3d_type(
    const py::object& q_in,
    py::object&       result_x_in,
    py::object&       result_y_in,
    py::object&       result_z_in,
    const int         num_elem_x1,
    const int         num_elem_x2,
    const int         num_elem_x3,
    const int         num_solpts,
    const int         verbose) {

  std::string dtype = py::str(q_in.attr("dtype").attr("name"));
  if (dtype == "float64")
  {
    select_extrap_all_3d<double, double>(
        q_in,
        result_x_in,
        result_y_in,
        result_z_in,
        num_elem_x1,
        num_elem_x2,
        num_elem_x3,
        num_solpts,
        verbose);
  }
  else if (dtype == "complex128")
  {
    select_extrap_all_3d<double, complex_t>(
        q_in,
        result_x_in,
        result_y_in,
        result_z_in,
        num_elem_x1,
        num_elem_x2,
        num_elem_x3,
        num_solpts,
        verbose);
  }
  else
  {
    std::cerr << __func__ << ": Unrecognized array type " << dtype << "\n";
  }
}

PYBIND11_MODULE(operators_omp, m) {
  m.def("extrap_all_3d", &select_extrap_all_3d_type);
}
#else  // WX_OMP
PYBIND11_MODULE(operators_cpp, m) {
  m.def("extrap_all_3d", &select_extrap_all_3d<double, double>);
  m.def("extrap_all_3d", &select_extrap_all_3d<double, complex_t>);
}
#endif // WX_OMP
