#include "operators.hpp"

#include <cstdio>
#include <iostream>

#include "definitions/definitions.hpp"
#include "kernels/kernels.h"

namespace py = pybind11;

template <typename real_t, typename num_t, int order, typename Func>
__global__ void extrap_3d(
    extrap_params_cubedsphere<num_t, order> params,
    const size_t                            max_num_threads,
    const bool                              verbose,
    Func                                    func) {

  constexpr size_t o2                 = order * order;
  constexpr size_t num_elem_per_block = EXTRAP_3D_BLOCK_SIZE / o2;
  constexpr size_t useful_block_size  = num_elem_per_block * o2;

  __shared__ num_t elem[num_elem_per_block][o2][order];

  // const size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t thread_id = threadIdx.x + blockIdx.x * useful_block_size;
  const size_t elem_id   = threadIdx.x / o2;

  if (threadIdx.x < useful_block_size && thread_id < max_num_threads)
  {
    params.set_index(thread_id);
    const int row_id = threadIdx.x % o2;
    for (int i = 0; i < order; i++)
    {
      elem[elem_id][row_id][i] = params.elem[row_id * order + i];
      // if (verbose && thread_id < o2 * 10 && thread_id % o2 < 2)
      // {
      //   printf(
      //       "elem[%d][%2d][%d] = params.elem[%3d] = %f\n",
      //       (int)elem_id,
      //       (int)row_id,
      //       (int)i,
      //       (int)(row_id * o2 + 1),
      //       to_real(params.elem[row_id * o2 + i]));
      // }
    }
  }

  __syncthreads();

  // if (thread_id < max_num_threads)
  if (threadIdx.x < useful_block_size && thread_id < max_num_threads)
  {
    // if (verbose && thread_id < o2 * 10 && thread_id % o2 < 2)
    // {
    //   printf(
    //       "Thread %3ld: index = %3ld"
    //       ", elem[0-6] = %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e"
    //       "\n                         "
    //       "elem[0-6] = %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e %9.3e"
    //       // ", sides %3.0f %3.0f %3.0f %3.0f %3.0f %3.0f"
    //       "\n",
    //       thread_id,
    //       params.index,
    //       to_real(params.elem[0]),
    //       to_real(params.elem[1]),
    //       to_real(params.elem[2]),
    //       to_real(params.elem[3]),
    //       to_real(params.elem[4]),
    //       to_real(params.elem[5]),
    //       to_real(params.elem[6]),
    //       to_real(((num_t*)elem[elem_id])[0]),
    //       to_real(((num_t*)elem[elem_id])[1]),
    //       to_real(((num_t*)elem[elem_id])[2]),
    //       to_real(((num_t*)elem[elem_id])[3]),
    //       to_real(((num_t*)elem[elem_id])[4]),
    //       to_real(((num_t*)elem[elem_id])[5]),
    //       to_real(((num_t*)elem[elem_id])[6])
    //       // to_real(*params.side_x1),
    //       // to_real(*params.side_x2),
    //       // to_real(*params.side_y1),
    //       // to_real(*params.side_y2),
    //       // to_real(*params.side_z1),
    //       // to_real(*params.side_z2)
    //   );
    // }
    func(params, &elem[elem_id][0][0], verbose);
  }
}

template <typename real_t, typename num_t, int order, typename Func>
void launch_extrap_3d(
    const py::object& q_in,
    py::object&       result_x_in,
    py::object&       result_y_in,
    py::object&       result_z_in,
    const int         num_elem_x1,
    const int         num_elem_x2,
    const int         num_elem_x3,
    const int         verbose,
    Func              func) {

  const num_t* q        = get_cupy_pointer<const num_t>(q_in);
  num_t*       result_x = get_cupy_pointer<num_t>(result_x_in);
  num_t*       result_y = get_cupy_pointer<num_t>(result_y_in);
  num_t*       result_z = get_cupy_pointer<num_t>(result_z_in);

  extrap_params_cubedsphere<num_t, order> p(q, 0, result_x, result_y, result_z);

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
    std::cout << "Problem size " << num_elem_x1 << "x" << num_elem_x2 << "x"
              << num_elem_x3 << " elements, order " << order << std::endl;
  }
  // return;

  extrap_3d<real_t, num_t, order>
      <<<num_blocks, EXTRAP_3D_BLOCK_SIZE>>>(p, num_active_threads, bool(verbose), func);
}

template <typename real_t, typename num_t>
void select_extrap_all_3d(
    const py::object& q_in,
    py::object&       result_x_in,
    py::object&       result_y_in,
    py::object&       result_z_in,
    const int         num_elem_x1,
    const int         num_elem_x2,
    const int         num_elem_x3,
    const int         num_solpts,
    const int         verbose) {
  switch (num_solpts)
  {
    // clang-format off
  case 1: launch_extrap_3d<real_t, num_t, 1>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 1>()); break;
  case 2: launch_extrap_3d<real_t, num_t, 2>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 2>()); break;
  case 3: launch_extrap_3d<real_t, num_t, 3>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 3>()); break;
  case 4: launch_extrap_3d<real_t, num_t, 4>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 4>()); break;
  case 5: launch_extrap_3d<real_t, num_t, 5>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 5>()); break;
  case 6: launch_extrap_3d<real_t, num_t, 6>(q_in, result_x_in, result_y_in, result_z_in, num_elem_x1, num_elem_x2, num_elem_x3, verbose, extrap_all_kernel<real_t, num_t, 6>()); break;
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

PYBIND11_MODULE(operators_cuda, m) {
  m.def("extrap_all_3d", &select_extrap_all_3d_type);
}
