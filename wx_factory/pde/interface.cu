#include "interface.hpp"

#include <cstddef>
#include <iostream>

#include "definitions/definitions.hpp"
#include "kernels/kernels.h"

namespace py = pybind11;

template <typename num_t>
__global__ void pointwise_eulercartesian_2d(
    const num_t* q,
    num_t*       flux_x1,
    num_t*       flux_x2,
    const int    num_elem_x1,
    const int    num_elem_x2,
    const int    num_solpts_tot) {
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
    const int    nmax_z) {
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
    const int    nmax_y) {
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
    else if (direction == 1)
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

template <typename real_t, typename num_t>
__global__ void pointwise_euler_cubedsphere_3d(
    kernel_params_cubedsphere<real_t, num_t> params,
    const size_t                  max_num_threads,
    const bool                    verbose)
{
  const size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  if (thread_id < max_num_threads)
  {
    params.set_index(thread_id);
    pointwise_euler_cubedsphere_3d_kernel<real_t, num_t>(params, verbose);
  }
}

template <typename real_t, typename num_t>
__global__ void forcing_euler_cubesphere_3d(
    forcing_params<real_t, num_t> params,
    const size_t                  max_num_threads,
    const bool                    verbose) {
  const size_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  if (thread_id < max_num_threads)
  {
    params.set_index(thread_id);
    forcing_euler_cubesphere_3d_kernel<real_t, num_t>(params, verbose);
  }
}

template <typename num_t>
void launch_pointwise_euler_cartesian_2d(
    py::object q,
    py::object flux_x1,
    py::object flux_x2,
    const int  num_elem_x1,
    const int  num_elem_x2,
    const int  num_solpts_tot) {
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
    const int  num_solpts_tot) {
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

template <typename real_t, typename num_t>
void launch_pointwise_euler_cubedsphere_3d(
  const py::object  q_in,
  const py::object sqrt_g_in,
  const py::object h_in,
  py::object        flux_x1,
  py::object        flux_x2,
  py::object        flux_x3,
  py::object        pressure,
  py::object        wflux_adv_x1,
  py::object        wflux_adv_x2,
  py::object        wflux_adv_x3,
  py::object        wflux_pres_x1,
  py::object        wflux_pres_x2,
  py::object        wflux_pres_x3,
  py::object        log_pressure,
  const int                  num_elem_x1,
  const int                  num_elem_x2,
  const int                  num_elem_x3,
  const int                  num_solpts,
  const int                  verbose) {

  const num_t* q_ptr        = get_cupy_pointer<const num_t>(q_in);
  num_t*       flux_x1_ptr  = get_cupy_pointer<num_t>(flux_x1);
  num_t*       flux_x2_ptr  = get_cupy_pointer<num_t>(flux_x2);
  num_t*       flux_x3_ptr  = get_cupy_pointer<num_t>(flux_x3);
  num_t*       pressure_ptr = get_cupy_pointer<num_t>(pressure);

  num_t* wflux_adv_x1_ptr = get_cupy_pointer<num_t>(wflux_adv_x1);
  num_t* wflux_adv_x2_ptr = get_cupy_pointer<num_t>(wflux_adv_x2);
  num_t* wflux_adv_x3_ptr = get_cupy_pointer<num_t>(wflux_adv_x3);

  num_t*        wflux_pres_x1_ptr = get_cupy_pointer<num_t>(wflux_pres_x1);
  num_t*        wflux_pres_x2_ptr = get_cupy_pointer<num_t>(wflux_pres_x2);
  num_t*        wflux_pres_x3_ptr = get_cupy_pointer<num_t>(wflux_pres_x3);
  num_t*        log_pressure_ptr  = get_cupy_pointer<num_t>(log_pressure);
  const real_t* sqrt_g_ptr        = get_cupy_pointer<const real_t>(sqrt_g_in);
  const real_t* h_ptr             = get_cupy_pointer<const real_t>(h_in);

  const size_t stride = num_elem_x1 * num_elem_x2 * num_elem_x3 * num_solpts;
  kernel_params_cubedsphere<real_t, num_t> base_params(
    q_ptr,
    sqrt_g_ptr,
    h_ptr,
    0,
    stride,
    flux_x1_ptr,
    flux_x2_ptr,
    flux_x3_ptr,
    pressure_ptr,
    wflux_adv_x1_ptr,
    wflux_adv_x2_ptr,
    wflux_adv_x3_ptr,
    wflux_pres_x1_ptr,
    wflux_pres_x2_ptr,
    wflux_pres_x3_ptr,
    log_pressure_ptr);

  const int    BLOCK_SIZE = 128;
  const size_t num_blocks = (stride + BLOCK_SIZE - 1) / BLOCK_SIZE;
  pointwise_euler_cubedsphere_3d<real_t, num_t>
      <<<num_blocks, BLOCK_SIZE>>>(base_params, stride, bool(verbose));
}

void select_pointwise_euler_cubedsphere_3d(
  const py::object  q_in,
  const py::object sqrt_g_in,
  const py::object h_in,
  py::object        flux_x1,
  py::object        flux_x2,
  py::object        flux_x3,
  py::object        pressure,
  py::object        wflux_adv_x1,
  py::object        wflux_adv_x2,
  py::object        wflux_adv_x3,
  py::object        wflux_pres_x1,
  py::object        wflux_pres_x2,
  py::object        wflux_pres_x3,
  py::object        log_pressure,
  const int                  num_elem_x1,
  const int                  num_elem_x2,
  const int                  num_elem_x3,
  const int                  num_solpts,
  const int                  verbose) {

  // Determine cupy array dtype
  std::string dtype = py::str(q_in.attr("dtype").attr("name"));
  if (dtype == "float64")
  {
    launch_pointwise_euler_cubedsphere_3d<double, double>(
      q_in,
      sqrt_g_in,
      h_in,
      flux_x1,
      flux_x2,
      flux_x3,
      pressure,
      wflux_adv_x1,
      wflux_adv_x2,
      wflux_adv_x3,
      wflux_pres_x1,
      wflux_pres_x2,
      wflux_pres_x3,
      log_pressure,
      num_elem_x1,
      num_elem_x2,
      num_elem_x3,
      num_solpts,
      verbose);
  }
  else if (dtype == "complex128")
  {
    launch_pointwise_euler_cubedsphere_3d<double, complex_t>(
      q_in,
      sqrt_g_in,
      h_in,
      flux_x1,
      flux_x2,
      flux_x3,
      pressure,
      wflux_adv_x1,
      wflux_adv_x2,
      wflux_adv_x3,
      wflux_pres_x1,
      wflux_pres_x2,
      wflux_pres_x3,
      log_pressure,
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

template <typename num_t>
void launch_riemann_eulercartesian_ausm_2d(
    py::object q_itf_x1,
    py::object q_itf_x2,
    py::object flux_itf_x1,
    py::object flux_itf_x2,
    const int  num_elem_x1,
    const int  num_elem_x2,
    const int  num_solpts) {
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
    const int  num_solpts) {

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

template <typename real_t, typename num_t>
void launch_forcing_euler_cubesphere_3d(
    py::object q_in,
    py::object pressure_in,
    py::object sqrt_g_in,
    py::object h_in,
    py::object christoffel_in,
    py::object forcing_in,
    const int  num_elem_x1,
    const int  num_elem_x2,
    const int  num_elem_x3,
    const int  num_solpts,
    const int  verbose) {
  const num_t*  q           = get_cupy_pointer<const num_t>(q_in);
  const num_t*  pressure    = get_cupy_pointer<const num_t>(pressure_in);
  const real_t* sqrt_g      = get_cupy_pointer<const real_t>(sqrt_g_in);
  const real_t* h           = get_cupy_pointer<const real_t>(h_in);
  const real_t* christoffel = get_cupy_pointer<const real_t>(christoffel_in);
  num_t*        forcing     = get_cupy_pointer<num_t>(forcing_in);

  const size_t stride = num_elem_x1 * num_elem_x2 * num_elem_x3 * num_solpts;
  forcing_params<real_t, num_t>
      base_params(q, pressure, sqrt_g, h, christoffel, forcing, 0, stride);

  const int    BLOCK_SIZE = 128;
  const size_t num_blocks = (stride + BLOCK_SIZE - 1) / BLOCK_SIZE;
  forcing_euler_cubesphere_3d<real_t, num_t>
      <<<num_blocks, BLOCK_SIZE>>>(base_params, stride, bool(verbose));
}

void select_forcing_euler_cubesphere_3d(
    py::object q_in,
    py::object pressure_in,
    py::object sqrt_g_in,
    py::object h_in,
    py::object christoffel_in,
    py::object forcing_in,
    const int  num_elem_x1,
    const int  num_elem_x2,
    const int  num_elem_x3,
    const int  num_solpts,
    const int  verbose) {

  // Determine cupy array dtype
  std::string dtype = py::str(q_in.attr("dtype").attr("name"));
  if (dtype == "float64")
  {
    launch_forcing_euler_cubesphere_3d<double, double>(
        q_in,
        pressure_in,
        sqrt_g_in,
        h_in,
        christoffel_in,
        forcing_in,
        num_elem_x1,
        num_elem_x2,
        num_elem_x3,
        num_solpts,
        verbose);
  }
  else if (dtype == "complex128")
  {
    launch_forcing_euler_cubesphere_3d<double, complex_t>(
        q_in,
        pressure_in,
        sqrt_g_in,
        h_in,
        christoffel_in,
        forcing_in,
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

PYBIND11_MODULE(pde_cuda, m) {
  // Pointwise fluxes
  m.def("pointwise_eulercartesian_2d", &select_pointwise_eulercartesian_2d);
  // m.def("pointwise_eulercubedsphere_3d", &select_pointwise_euler_cubedsphere_3d);

  // Riemann fluxes
  m.def("riemann_eulercartesian_ausm_2d", &select_riemann_eulercartesian_ausm_2d);

  // Forcing
  m.def("forcing_euler_cubesphere_3d", &select_forcing_euler_cubesphere_3d);
}
