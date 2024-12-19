#include "definitions.hpp"
#include "interface.hpp"

#include "kernels/kernels.h"

namespace py = pybind11;

template <typename num_t>
__global__ void pointwise_eulercartesian_2d(
    const num_t* q,
    num_t*       flux_x1,
    num_t*       flux_x2,
    const int    num_elem_x1,
    const int    num_elem_x2,
    const int    num_solpts_tot)
{
  const int ind    = threadIdx.x + blockIdx.x * blockDim.x;
  const int nmax   = num_elem_x1 * num_elem_x2 * num_solpts_tot;
  const int stride = nmax;
  if (ind < nmax)
  {
    // Store variables and pointers to compute the fluxes
    kernel_params<num_t, euler_state_2d>
        params(q, flux_x1, flux_x2, nullptr, ind, stride);

    // Call the pointwise flux kernel
    pointwise_eulercartesian_2d_kernel(params);
  }
}

template <typename num_t>
__global__ void riemann_eulercartesian_ausm_2d(
    const num_t* q_itf,
    num_t*       flux_itf,
    const int    num_elem_x1,
    const int    num_elem_x2,
    const int    num_solpts,
    const int    direction,
    const int    nmax_x,
    const int    nmax_y,
    const int    nmax_z)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;
  const int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix < nmax_x && iy < nmax_y && iz < nmax_z)
  {
    const int num_solpts_riem = 2 * num_solpts;
    const int stride          = num_elem_x1 * num_elem_x2 * num_solpts_riem;
    const int array_shape[4]  = {4, num_elem_x2, num_elem_x1, num_solpts_riem};

    if (direction == 0)
    {
      // Initialize left-hand side parameters
      const int indl = get_c_index(0, ix, iy, num_solpts + iz, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_l(q_itf, flux_itf, nullptr, nullptr, indl, stride);

      // Initialize right-hand-size parameters
      const int indr = get_c_index(0, ix, iy + 1, iz, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_r(q_itf, flux_itf, nullptr, nullptr, indr, stride);

      // Call Riemann kernel on the horizontal direction
      riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, direction);
    }
    else if (direction == 1)
    {
      // Initialize left-hand side parameters
      const int indl = get_c_index(0, ix, iy, num_solpts + iz, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_l(q_itf, nullptr, flux_itf, nullptr, indl, stride);

      // Initialize right-hand-size parameters
      const int indr = get_c_index(0, ix + 1, iy, iz, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_r(q_itf, nullptr, flux_itf, nullptr, indr, stride);

      // Call Riemann kernel on the horizontal direction
      riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, direction);
    }
  }
}

template <typename num_t>
__global__ void boundary_eulercartesian_2d(
    const num_t* q_itf,
    num_t*       flux_itf,
    const int    num_elem_x1,
    const int    num_elem_x2,
    const int    num_solpts,
    const int    direction,
    const int    nmax_x,
    const int    nmax_y)
{
  const int ix = blockIdx.x * blockDim.x + threadIdx.x;
  const int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ix < nmax_x && iy < nmax_y)
  {
    const int num_solpts_riem = 2 * num_solpts;
    const int stride          = num_elem_x1 * num_elem_x2 * num_solpts_riem;
    const int array_shape[4]  = {4, num_elem_x2, num_elem_x1, num_solpts_riem};

    if (direction == 0)
    {
      // Left flux
      const int indl = get_c_index(0, ix, 0, iy, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_l(q_itf, flux_itf, nullptr, nullptr, indl, stride);
      boundary_eulercartesian_2d_kernel(params_l, 0);

      // Right flux
      const int indr = get_c_index(0, ix, num_elem_x1 - 1, num_solpts + iy, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_r(q_itf, flux_itf, nullptr, nullptr, indr, stride);
      boundary_eulercartesian_2d_kernel(params_r, 0);
    }

    if (direction == 1)
    {
      // Bottom flux
      const int indb = get_c_index(0, 0, ix, iy, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_b(q_itf, nullptr, flux_itf, nullptr, indb, stride);
      boundary_eulercartesian_2d_kernel(params_b, 1);

      // Top flux
      const int indt = get_c_index(0, num_elem_x2 - 1, ix, num_solpts + iy, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_t(q_itf, nullptr, flux_itf, nullptr, indt, stride);
      boundary_eulercartesian_2d_kernel(params_t, 1);
    }
  }
}

template <typename num_t>
void launch_pointwise_euler_cartesian_2d(
    py::object q,
    py::object flux_x1,
    py::object flux_x2,
    const int  num_elem_x1,
    const int  num_elem_x2,
    const int  num_solpts_tot)
{
  // Extract CuPy pointers
  uintptr_t cupy_q_ptr       = q.attr("data").attr("ptr").cast<size_t>();
  uintptr_t cupy_flux_x1_ptr = flux_x1.attr("data").attr("ptr").cast<size_t>();
  uintptr_t cupy_flux_x2_ptr = flux_x2.attr("data").attr("ptr").cast<size_t>();

  const int num_blocks =
      (num_elem_x1 * num_elem_x2 * num_solpts_tot + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const num_t* q_ptr       = reinterpret_cast<const num_t*>(cupy_q_ptr);
  num_t*       flux_x1_ptr = reinterpret_cast<num_t*>(cupy_flux_x1_ptr);
  num_t*       flux_x2_ptr = reinterpret_cast<num_t*>(cupy_flux_x2_ptr);
  pointwise_eulercartesian_2d<num_t><<<num_blocks, BLOCK_SIZE>>>(
      q_ptr,
      flux_x1_ptr,
      flux_x2_ptr,
      num_elem_x1,
      num_elem_x2,
      num_solpts_tot);
}

void select_pointwise_eulercartesian_2d(
    py::object q,
    py::object flux_x1,
    py::object flux_x2,
    const int  num_elem_x1,
    const int  num_elem_x2,
    const int  num_solpts_tot)
{
  // Determine the CuPy array dtype
  std::string dtype = py::str(q.attr("dtype").attr("name"));

  // Dispatch according to type
  if (dtype == "float64")
  {
    launch_pointwise_euler_cartesian_2d<double>(
        q,
        flux_x1,
        flux_x2,
        num_elem_x1,
        num_elem_x2,
        num_solpts_tot);
  }
  else if (dtype == "complex128")
  {
    launch_pointwise_euler_cartesian_2d<complex_t>(
        q,
        flux_x1,
        flux_x2,
        num_elem_x1,
        num_elem_x2,
        num_solpts_tot);
  }
}

template <typename num_t>
void launch_riemann_eulercartesian_ausm_2d(
    py::object q_itf_x1,
    py::object q_itf_x2,
    py::object flux_itf_x1,
    py::object flux_itf_x2,
    const int  num_elem_x1,
    const int  num_elem_x2,
    const int  num_solpts)
{
  // Extract CuPy pointers
  uintptr_t cupy_q_x1_ptr    = q_itf_x1.attr("data").attr("ptr").cast<size_t>();
  uintptr_t cupy_q_x2_ptr    = q_itf_x2.attr("data").attr("ptr").cast<size_t>();
  uintptr_t cupy_flux_x1_ptr = flux_itf_x1.attr("data").attr("ptr").cast<size_t>();
  uintptr_t cupy_flux_x2_ptr = flux_itf_x2.attr("data").attr("ptr").cast<size_t>();

  // Reinterpret as appropriate pointers
  const num_t* q_x1_ptr = reinterpret_cast<const num_t*>(cupy_q_x1_ptr);
  const num_t* q_x2_ptr = reinterpret_cast<const num_t*>(cupy_q_x2_ptr);
  num_t*       f_x1_ptr = reinterpret_cast<num_t*>(cupy_flux_x1_ptr);
  num_t*       f_x2_ptr = reinterpret_cast<num_t*>(cupy_flux_x2_ptr);

  int width, height, depth;

  // Call Riemann solver on the horizontal direction
  width  = num_elem_x2;
  height = num_elem_x1 - 1;
  depth  = num_solpts;

  dim3 threads_per_block(8, 8, 8);
  dim3 num_blocks1(
      (width + threads_per_block.x - 1) / threads_per_block.x,
      (height + threads_per_block.y - 1) / threads_per_block.y,
      (depth + threads_per_block.z - 1) / threads_per_block.z);

  riemann_eulercartesian_ausm_2d<num_t><<<num_blocks1, threads_per_block>>>(
      q_x1_ptr,
      f_x1_ptr,
      num_elem_x1,
      num_elem_x2,
      num_solpts,
      0,
      width,
      height,
      depth);

  // Call Riemann solver on the vertical direction
  width  = num_elem_x2 - 1;
  height = num_elem_x1;
  depth  = num_solpts;

  dim3 num_blocks2(
      (width + threads_per_block.x - 1) / threads_per_block.x,
      (height + threads_per_block.y - 1) / threads_per_block.y,
      (depth + threads_per_block.z - 1) / threads_per_block.z);

  riemann_eulercartesian_ausm_2d<num_t><<<num_blocks2, threads_per_block>>>(
      q_x2_ptr,
      f_x2_ptr,
      num_elem_x1,
      num_elem_x2,
      num_solpts,
      1,
      width,
      height,
      depth);

  // Set the boundary fluxes on the horizontal direction
  dim3 threads_per_block2(16, 16);

  width  = num_elem_x2;
  height = num_solpts;

  dim3 num_blocks3(
      (width + threads_per_block2.x - 1) / threads_per_block2.x,
      (height + threads_per_block2.y - 1) / threads_per_block2.y);

  boundary_eulercartesian_2d<num_t><<<num_blocks3, threads_per_block2>>>(
      q_x1_ptr,
      f_x1_ptr,
      num_elem_x1,
      num_elem_x2,
      num_solpts,
      0,
      width,
      height);

  width  = num_elem_x1;
  height = num_solpts;

  dim3 num_blocks4(
      (width + threads_per_block2.x - 1) / threads_per_block2.x,
      (height + threads_per_block2.y - 1) / threads_per_block2.y);

  boundary_eulercartesian_2d<num_t><<<num_blocks4, threads_per_block2>>>(
      q_x2_ptr,
      f_x2_ptr,
      num_elem_x1,
      num_elem_x2,
      num_solpts,
      1,
      width,
      height);
}

void select_riemann_eulercartesian_ausm_2d(
    py::object q_itf_x1,
    py::object q_itf_x2,
    py::object flux_itf_x1,
    py::object flux_itf_x2,
    const int  num_elem_x1,
    const int  num_elem_x2,
    const int  num_solpts)
{
  // Determine the CuPy array dtype
  std::string dtype = py::str(q_itf_x1.attr("dtype").attr("name"));

  if (dtype == "float64")
  {
    launch_riemann_eulercartesian_ausm_2d<double>(
        q_itf_x1,
        q_itf_x2,
        flux_itf_x1,
        flux_itf_x2,
        num_elem_x1,
        num_elem_x2,
        num_solpts);
  }
  else if (dtype == "complex128")
  {
    launch_riemann_eulercartesian_ausm_2d<complex_t>(
        q_itf_x1,
        q_itf_x2,
        flux_itf_x1,
        flux_itf_x2,
        num_elem_x1,
        num_elem_x2,
        num_solpts);
  }
}

PYBIND11_MODULE(interface_cuda, m)
{
  // Pointwise fluxes
  m.def("pointwise_eulercartesian_2d", &select_pointwise_eulercartesian_2d);

  // Riemann fluxes
  m.def("riemann_eulercartesian_ausm_2d", &select_riemann_eulercartesian_ausm_2d);
}
