#include "operators.hpp"

#include <cstdio>
#include <iostream>

#include "definitions/definitions.hpp"
#include "kernels/kernels.h"

namespace py = pybind11;

template <
    typename real_t,
    typename num_t,
    int order,
    template <typename, typename, int> class MyFunc,
    template <typename, int> class ParamType>
__global__ void element_wise_kernel(
    ParamType<num_t, order> params,
    const size_t            max_num_threads,
    const bool              verbose) {

  constexpr size_t group_size         = order * order;
  constexpr size_t num_elem_per_block = EXTRAP_3D_BLOCK_SIZE / group_size;
  constexpr size_t useful_block_size  = num_elem_per_block * group_size;

  __shared__ num_t elem[num_elem_per_block][group_size][order];

  // const size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t thread_id = threadIdx.x + blockIdx.x * useful_block_size;
  const size_t elem_id   = threadIdx.x / group_size;

  if (threadIdx.x < useful_block_size && thread_id < max_num_threads)
  {
    params.set_index(thread_id);
    const int row_id = threadIdx.x % group_size;
    for (int i = 0; i < order; i++)
    {
      elem[elem_id][row_id][i] = params.elem[row_id * order + i];
    }
  }

  __syncthreads();

  if (threadIdx.x < useful_block_size && thread_id < max_num_threads)
  {
    // func(params, &elem[elem_id][0][0], verbose);
    MyFunc<real_t, num_t, order>()(params, &elem[elem_id][0][0], verbose);
  }
}

template <
    typename real_t,
    typename num_t,
    int order,
    template <typename, int> class ParamType,
    template <typename, typename, int> class MyFunc,
    typename... Args>
void launch_kernel(
    const int num_elem_x1,
    const int num_elem_x2,
    const int num_elem_x3,
    const int verbose,
    Args... args) {

  ParamType<num_t, order> p(0, args...);

  constexpr size_t o2 = order * order;
  // constexpr size_t BLOCK_SIZE            = 257;
  constexpr size_t num_elem_per_block    = EXTRAP_3D_BLOCK_SIZE / o2;
  constexpr size_t num_threads_per_block = num_elem_per_block * o2;

  const size_t num_active_threads =
      5 * num_elem_x1 * num_elem_x2 * num_elem_x3 * order * order;
  const size_t num_blocks =
      (num_active_threads + num_threads_per_block - 1) / num_threads_per_block;
  // const size_t num_blocks = (max_num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (verbose)
  {
    std::cout << "Launching " << num_active_threads << " threads in " << num_blocks
              << " block(s) of size " << EXTRAP_3D_BLOCK_SIZE << " (with "
              << num_threads_per_block << " active threads per block, in "
              << num_elem_per_block << " elements)\n";
    std::cout << "Problem size " << num_elem_x1 << "x" << num_elem_x2 << "x"
              << num_elem_x3 << " elements, order " << order << std::endl;
  }

  element_wise_kernel<real_t, num_t, order, MyFunc>
      <<<num_blocks, EXTRAP_3D_BLOCK_SIZE>>>(p, num_active_threads, bool(verbose));
}

template <
    typename real_t,
    typename num_t,
    template <typename, int> class ParamType,
    template <typename, typename, int> class KernelType,
    typename... Args>
void select_order(
    const int num_elem_x1,
    const int num_elem_x2,
    const int num_elem_x3,
    const int num_solpts,
    const int verbose,
    Args... args) {
  switch (num_solpts)
  {
    // clang-format off
  case 1: launch_kernel<real_t, num_t, 1, ParamType, KernelType>(num_elem_x1, num_elem_x2, num_elem_x3, verbose, args...); break;
  case 2: launch_kernel<real_t, num_t, 2, ParamType, KernelType>(num_elem_x1, num_elem_x2, num_elem_x3, verbose, args...); break;
  case 3: launch_kernel<real_t, num_t, 3, ParamType, KernelType>(num_elem_x1, num_elem_x2, num_elem_x3, verbose, args...); break;
  case 4: launch_kernel<real_t, num_t, 4, ParamType, KernelType>(num_elem_x1, num_elem_x2, num_elem_x3, verbose, args...); break;
  case 5: launch_kernel<real_t, num_t, 5, ParamType, KernelType>(num_elem_x1, num_elem_x2, num_elem_x3, verbose, args...); break;
  case 6: launch_kernel<real_t, num_t, 6, ParamType, KernelType>(num_elem_x1, num_elem_x2, num_elem_x3, verbose, args...); break;
  default: std::cerr << __func__ << ": Not implemented for order " << num_solpts << "\n"; break;
    // clang-format on
  }
}

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
    select_order<double, double, extrap_params_cubedsphere, extrap_all_kernel>(
        num_elem_x1,
        num_elem_x2,
        num_elem_x3,
        num_solpts,
        verbose,
        q_in,
        result_x_in,
        result_y_in,
        result_z_in);
  }
  else if (dtype == "complex128")
  {
    select_order<double, complex_t, extrap_params_cubedsphere, extrap_all_kernel>(
        num_elem_x1,
        num_elem_x2,
        num_elem_x3,
        num_solpts,
        verbose,
        q_in,
        result_x_in,
        result_y_in,
        result_z_in);
  }
  else
  {
    std::cerr << __func__ << ": Unrecognized array type " << dtype << "\n";
  }
}

__constant__ double deriv_x_operator[MAX_DERIV_ORDER * MAX_DERIV_ORDER * MAX_DERIV_ORDER];

template <typename real_t, typename num_t, int order, typename Func>
__global__ void deriv_3d(
    extrap_params_cubedsphere<num_t, order> params,
    const size_t                            max_num_threads,
    const bool                              verbose,
    Func                                    func) {
  constexpr size_t o3                 = order * order * order;
  constexpr size_t num_elem_per_block = EXTRAP_3D_BLOCK_SIZE / o3;
  constexpr size_t useful_block_size  = num_elem_per_block * o3;

  __shared__ num_t elem[num_elem_per_block][o3][order];

  const size_t thread_id = threadIdx.x + blockIdx.x * useful_block_size;
  const size_t elem_id   = threadIdx.x / o3;

  if (threadIdx.x < useful_block_size && thread_id < max_num_threads)
  {
    params.set_index(thread_id);
    const int row_id = threadIdx.x % o3;
    for (int i = 0; i < order; i++)
    {
      elem[elem_id][row_id][i] = params.elem[row_id * order + i];
    }
  }

  __syncthreads();

  if (threadIdx.x < useful_block_size && thread_id < max_num_threads)
  {
    func(params, &elem[elem_id][0][0], verbose);
  }
}

template <typename real_t, typename num_t, int order, typename Func>
void launch_deriv_3d(
    const py::object& field_in,
    const py::object& operator_in,
    py::object&       result_x_in,
    const int         num_fields,
    const int         num_elem_x1,
    const int         num_elem_x2,
    const int         num_elem_x3,
    const int         verbose,
    Func              func) {
  const num_t*  field    = get_cupy_pointer<const num_t>(field_in);
  const real_t* op       = get_cupy_pointer<const real_t>(operator_in);
  num_t*        result_x = get_cupy_pointer<num_t>(result_x_in);

  (void)func; // TODO remove this

  constexpr size_t o3 = order * order * order;

  cudaCheck(cudaMemcpyToSymbol(
      deriv_x_operator,
      op,
      o3 * sizeof(real_t),
      0,
      cudaMemcpyDeviceToDevice));

  // extrap_params_cubedsphere<num_t, order> p(q, 0, result_x, result_y, result_z);

  // constexpr size_t BLOCK_SIZE            = 257;
  constexpr size_t num_elem_per_block    = EXTRAP_3D_BLOCK_SIZE / o3;
  constexpr size_t num_threads_per_block = num_elem_per_block * o3;

  const size_t num_active_threads =
      num_fields * num_elem_x1 * num_elem_x2 * num_elem_x3 * order * order * order;
  const size_t num_blocks =
      (num_active_threads + num_threads_per_block - 1) / num_threads_per_block;
  // const size_t num_blocks = (max_num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

  if (verbose)
  {
    // std::cout << std::format(
    //     "Launching {} threads in {} block(s) of size {}.\n"
    //     "Problem size {}x{}x{} elements, order {}\n",
    //     num_active_threads,
    //     num_blocks,
    //     EXTRAP_3D_BLOCK_SIZE,
    //     num_elem_x1,
    //     num_elem_x2,
    //     num_elem_x3,
    //     order);
    std::cout << "Launching " << num_active_threads << " threads in " << num_blocks
              << " block(s) of size " << EXTRAP_3D_BLOCK_SIZE << " (with "
              << num_threads_per_block << " active threads per block, in "
              << num_elem_per_block << " elements)\n";
    std::cout << "Problem size " << num_fields << " vars, " << num_elem_x1 << "x"
              << num_elem_x2 << "x" << num_elem_x3 << " elements, order " << order
              << std::endl;
  }
  // return;

  // deriv_3d<real_t, num_t, order>
  //     <<<num_blocks, EXTRAP_3D_BLOCK_SIZE>>>(p, num_active_threads, bool(verbose),
  //     func);
}
template <typename real_t, typename num_t>
void select_deriv_x_3d(
    const py::object& field_in,
    const py::object& operator_in,
    py::object&       result_x_in,
    const int         num_fields,
    const int         num_elem_x1,
    const int         num_elem_x2,
    const int         num_elem_x3,
    const int         num_solpts,
    const int         verbose) {
  switch (num_solpts)
  {
    // clang-format off
  // case 1: launch_deriv_3d<real_t, num_t, 1>(field_in, operator_in, result_x_in, num_fields, num_elem_x1, num_elem_x2, num_elem_x3, verbose, deriv_x_kernel<real_t, num_t, 1>()); break;
  // case 2: launch_deriv_3d<real_t, num_t, 2>(field_in, operator_in, result_x_in, num_fields, num_elem_x1, num_elem_x2, num_elem_x3, verbose, deriv_x_kernel<real_t, num_t, 2>()); break;
  case 3: launch_deriv_3d<real_t, num_t, 3>(field_in, operator_in, result_x_in, num_fields, num_elem_x1, num_elem_x2, num_elem_x3, verbose, deriv_x_kernel<real_t, num_t, 3>()); break;
  // case 4: launch_deriv_3d<real_t, num_t, 4>(field_in, operator_in, result_x_in, num_fields, num_elem_x1, num_elem_x2, num_elem_x3, verbose, deriv_x_kernel<real_t, num_t, 4>()); break;
  // case 5: launch_deriv_3d<real_t, num_t, 5>(field_in, operator_in, result_x_in, num_fields, num_elem_x1, num_elem_x2, num_elem_x3, verbose, deriv_x_kernel<real_t, num_t, 5>()); break;
  // case 6: launch_deriv_3d<real_t, num_t, 6>(field_in, operator_in, result_x_in, num_fields, num_elem_x1, num_elem_x2, num_elem_x3, verbose, deriv_x_kernel<real_t, num_t, 6>()); break;
  default: std::cerr << __func__ << ": Not implemented for order " << num_solpts << "\n"; break;
    // clang-format on
  }
}
void select_deriv_x_3d_type(
    const py::object& field_in,
    const py::object& operator_in,
    py::object&       result_x_in,
    const int         num_fields,
    const int         num_elem_x1,
    const int         num_elem_x2,
    const int         num_elem_x3,
    const int         num_solpts,
    const int         verbose) {
  std::string dtype = py::str(field_in.attr("dtype").attr("name"));
  if (dtype == "float64")
  {
    select_deriv_x_3d<double, double>(
        field_in,
        operator_in,
        result_x_in,
        num_fields,
        num_elem_x1,
        num_elem_x2,
        num_elem_x3,
        num_solpts,
        verbose);
  }
  else if (dtype == "complex128")
  {
    select_deriv_x_3d<double, complex_t>(
        field_in,
        operator_in,
        result_x_in,
        num_fields,
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

PYBIND11_MODULE(operators_cuda, m) {
  m.def("extrap_all_3d", &select_extrap_all_3d_type);
  m.def("deriv_x_3d", &select_deriv_x_3d_type);
}
