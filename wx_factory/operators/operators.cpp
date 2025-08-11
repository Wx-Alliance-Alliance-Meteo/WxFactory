#include "operators.hpp"

#include <iostream>

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

template <typename real_t, typename num_t, int order, typename Func>
void extrap_3d(
    const py_array<num_t>& q_in,
    py_array<num_t>&       result_x_in,
    py_array<num_t>&       result_y_in,
    py_array<num_t>&       result_z_in,
    const int              num_elem_x1,
    const int              num_elem_x2,
    const int              num_elem_x3,
    const int              verbose,
    Func                   func) {

  const num_t* q        = get_c_ptr<num_t>(q_in);
  num_t*       result_x = get_c_ptr<num_t>(result_x_in);
  num_t*       result_y = get_c_ptr<num_t>(result_y_in);
  num_t*       result_z = get_c_ptr<num_t>(result_z_in);

#pragma omp target teams distribute collapse(5)
  for (int var = 0; var < 5; var++)
  {
    for (int i = 0; i < num_elem_x3; i++)
    {
      for (int j = 0; j < num_elem_x2; j++)
      {
        for (int k = 0; k < num_elem_x1; k++)
        {
          for (int s = 0; s < order * order; s++)
          {
            const int index =
                ((((var * num_elem_x3) + i) * num_elem_x2 + j) * num_elem_x1 + k) *
                    order * order +
                s;

            extrap_params_cubedsphere<num_t, order>
                p(q, index, result_x, result_y, result_z);

            func(p, nullptr, verbose);
          }
        }
      }
    }
  }
}

template <typename real_t, typename num_t>
void select_extrap_x_3d(
    const py::array_t<num_t>& q_in,
    py::array_t<num_t>&       result_in,
    const int                 num_elem_x1,
    const int                 num_elem_x2,
    const int                 num_elem_x3,
    const int                 num_solpts,
    const int                 verbose) {
  switch (num_solpts)
  {
    // clang-format off
  case 1: extrap_3d<real_t, num_t, 1>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_x_kernel<real_t, num_t, 1>()); break;
  case 2: extrap_3d<real_t, num_t, 2>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_x_kernel<real_t, num_t, 2>()); break;
  case 3: extrap_3d<real_t, num_t, 3>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_x_kernel<real_t, num_t, 3>()); break;
  case 4: extrap_3d<real_t, num_t, 4>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_x_kernel<real_t, num_t, 4>()); break;
  case 5: extrap_3d<real_t, num_t, 5>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_x_kernel<real_t, num_t, 5>()); break;
  case 6: extrap_3d<real_t, num_t, 6>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_x_kernel<real_t, num_t, 6>()); break;
  default: std::cerr << __func__ << ": Not implemented for order " << num_solpts << "\n"; break;
    // clang-format on
  }
}

template <typename real_t, typename num_t>
void select_extrap_y_3d(
    const py::array_t<num_t>& q_in,
    py::array_t<num_t>&       result_in,
    const int                 num_elem_x1,
    const int                 num_elem_x2,
    const int                 num_elem_x3,
    const int                 num_solpts,
    const int                 verbose) {
  switch (num_solpts)
  {
    // clang-format off
  case 1: extrap_3d<real_t, num_t, 1>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_y_kernel<real_t, num_t, 1>()); break;
  case 2: extrap_3d<real_t, num_t, 2>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_y_kernel<real_t, num_t, 2>()); break;
  case 3: extrap_3d<real_t, num_t, 3>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_y_kernel<real_t, num_t, 3>()); break;
  case 4: extrap_3d<real_t, num_t, 4>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_y_kernel<real_t, num_t, 4>()); break;
  case 5: extrap_3d<real_t, num_t, 5>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_y_kernel<real_t, num_t, 5>()); break;
  case 6: extrap_3d<real_t, num_t, 6>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_y_kernel<real_t, num_t, 6>()); break;
  default: std::cerr << __func__ << ": Not implemented for order " << num_solpts << "\n"; break;
    // clang-format on
  }
}

template <typename real_t, typename num_t>
void select_extrap_z_3d(
    const py::array_t<num_t>& q_in,
    py::array_t<num_t>&       result_in,
    const int                 num_elem_x1,
    const int                 num_elem_x2,
    const int                 num_elem_x3,
    const int                 num_solpts,
    const int                 verbose) {
  switch (num_solpts)
  {
    // clang-format off
  case 1: extrap_3d<real_t, num_t, 1>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_z_kernel<real_t, num_t, 1>()); break;
  case 2: extrap_3d<real_t, num_t, 2>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_z_kernel<real_t, num_t, 2>()); break;
  case 3: extrap_3d<real_t, num_t, 3>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_z_kernel<real_t, num_t, 3>()); break;
  case 4: extrap_3d<real_t, num_t, 4>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_z_kernel<real_t, num_t, 4>()); break;
  case 5: extrap_3d<real_t, num_t, 5>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_z_kernel<real_t, num_t, 5>()); break;
  case 6: extrap_3d<real_t, num_t, 6>(q_in, result_in, result_in, result_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_z_kernel<real_t, num_t, 6>()); break;
  default: std::cerr << __func__ << ": Not implemented for order " << num_solpts << "\n"; break;
    // clang-format on
  }
}

template <typename real_t, typename num_t>
void select_extrap_all_3d(
    const py_array<num_t>& q_in,
    py_array<num_t>&       result_x_in,
    py_array<num_t>&       result_y_in,
    py_array<num_t>&       result_z_in,
    const int              num_elem_x1,
    const int              num_elem_x2,
    const int              num_elem_x3,
    const int              num_solpts,
    const int              verbose) {
  switch (num_solpts)
  {
    // clang-format off
  case 1: extrap_3d<real_t, num_t, 1>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 1>()); break;
  case 2: extrap_3d<real_t, num_t, 2>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 2>()); break;
  case 3: extrap_3d<real_t, num_t, 3>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 3>()); break;
  case 4: extrap_3d<real_t, num_t, 4>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 4>()); break;
  case 5: extrap_3d<real_t, num_t, 5>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 5>()); break;
  case 6: extrap_3d<real_t, num_t, 6>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 6>()); break;
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
