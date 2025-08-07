#include "pde.hpp"

#include <cstddef>
#include <iostream>

#include "common/parameters.hpp"
#include "kernels/kernels.h"

namespace py = pybind11;

const int BLOCK_SIZE = 256;

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
__global__ void riemann_euler_cubedsphere_rusanov_3d(
    const num_t*  q_itf,
    const real_t* sqrt_g,
    const real_t* h,
    num_t*        flux_itf,
    num_t*        pressure,
    num_t*        wflux_adv_itf,
    num_t*        wflux_pres_itf,
    const int     direction,
    const int     nmax_x1,
    const int     nmax_x2,
    const int     nmax_x3,
    const int     num_solpts_face) {
  // Get the thread id
  const size_t tid         = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t total_tasks = nmax_x1 * nmax_x2 * nmax_x3 * num_solpts_face;

  if (tid >= total_tasks)
    return;

  // Get the grid indices
  // i indexes along the x3-direction
  // j indexes along the x2-direction
  // k indexes along the x1-direction
  // l indexes the point in the face
  int l    = tid % num_solpts_face;
  int tid2 = tid / num_solpts_face;

  int k = tid2 % nmax_x1;
  tid2  = tid2 / nmax_x1;

  int j = tid2 % nmax_x2;
  int i = tid2 / nmax_x2;

  int index_l, index_r, stride;

  if (direction == 0)
  {
    const int array_shape[5] = {5, nmax_x3, nmax_x2, nmax_x1 + 1, 2 * num_solpts_face};
    stride                   = (nmax_x1 + 1) * nmax_x2 * nmax_x3 * num_solpts_face * 2;
    index_l                  = get_c_index(0, i, j, k, l + num_solpts_face, array_shape);
    index_r                  = get_c_index(0, i, j, k + 1, l, array_shape);
  }
  else if (direction == 1)
  {
    const int array_shape[5] = {5, nmax_x3, nmax_x2 + 1, nmax_x1, 2 * num_solpts_face};
    stride                   = nmax_x1 * (nmax_x2 + 1) * nmax_x3 * num_solpts_face * 2;
    index_l                  = get_c_index(0, i, j, k, l + num_solpts_face, array_shape);
    index_r                  = get_c_index(0, i, j + 1, k, l, array_shape);
  }
  else
  {
    const int array_shape[5] = {5, nmax_x3 + 1, nmax_x2, nmax_x1, 2 * num_solpts_face};
    stride                   = nmax_x1 * nmax_x2 * (nmax_x3 + 1) * num_solpts_face * 2;
    index_l                  = get_c_index(0, i, j, k, l + num_solpts_face, array_shape);
    index_r                  = get_c_index(0, i + 1, j, k, l, array_shape);
  }

  // Get the pointers to the left and right parameters
  riemann_params_cubedsphere<real_t, num_t> params_l(
      q_itf,
      sqrt_g,
      h,
      index_l,
      stride,
      flux_itf,
      pressure,
      wflux_adv_itf,
      wflux_pres_itf);

  riemann_params_cubedsphere<real_t, num_t> params_r(
      q_itf,
      sqrt_g,
      h,
      index_r,
      stride,
      flux_itf,
      pressure,
      wflux_adv_itf,
      wflux_pres_itf);

  // Compute the Riemann flux
  bool boundary = (direction == 2 && (i == 0 || i == nmax_x3 - 1));
  riemann_euler_cubedsphere_rusanov_3d_kernel<real_t, num_t>(
      params_l,
      params_r,
      direction,
      boundary);
}

template <typename real_t, typename num_t>
__global__ void boundary_euler_cubedsphere_3d(
    num_t* q_itf,
    int    num_elem_x1,
    int    num_elem_x2,
    int    num_elem_x3,
    int    num_solpts_face) {
  // Sets the boundary condition along the vertical direction
  int tid         = blockIdx.x * blockDim.x + threadIdx.x;
  int total_tasks = num_elem_x2 * num_elem_x1 * num_solpts_face;
  if (tid >= total_tasks)
    return;

  // Compute grid indices
  int l    = tid % num_solpts_face;
  int tid2 = tid / num_solpts_face;

  int k = tid2 % num_elem_x1;
  int j = tid2 / num_elem_x1;

  const int array_shape[5] =
      {5, num_elem_x3 + 2, num_elem_x2, num_elem_x1, 2 * num_solpts_face};
  const int stride = num_elem_x1 * num_elem_x2 * (num_elem_x3 + 2) * num_solpts_face * 2;

  // Bottom boundary
  const int index_b_bottom  = get_c_index(0, 0, j, k, l + num_solpts_face, array_shape);
  const int index_in_bottom = get_c_index(0, 1, j, k, l, array_shape);

  euler_state_3d<num_t>       params_b_bottom(q_itf, index_b_bottom, stride);
  euler_state_3d<const num_t> params_in_bottom(q_itf, index_in_bottom, stride);

  boundary_euler_cubedsphere_3d_kernel<real_t, num_t>(params_in_bottom, params_b_bottom);

  // Top boundary
  const int index_b_top = get_c_index(0, num_elem_x3 + 1, j, k, l, array_shape);
  const int index_in_top =
      get_c_index(0, num_elem_x3, j, k, l + num_solpts_face, array_shape);

  euler_state_3d<num_t>       params_b_top(q_itf, index_b_top, stride);
  euler_state_3d<const num_t> params_in_top(q_itf, index_in_top, stride);

  boundary_euler_cubedsphere_3d_kernel<real_t, num_t>(params_in_top, params_b_top);
}

template <typename real_t, typename num_t>
__global__ void pointwise_euler_cubedsphere_3d(
    kernel_params_cubedsphere<real_t, num_t> params,
    const size_t                             max_num_threads,
    const bool                               verbose) {
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
    const py::object q_in,
    const py::object sqrt_g_in,
    const py::object h_in,
    py::object       flux_x1,
    py::object       flux_x2,
    py::object       flux_x3,
    py::object       pressure,
    py::object       wflux_adv_x1,
    py::object       wflux_adv_x2,
    py::object       wflux_adv_x3,
    py::object       wflux_pres_x1,
    py::object       wflux_pres_x2,
    py::object       wflux_pres_x3,
    py::object       log_pressure,
    const int        num_elem_x1,
    const int        num_elem_x2,
    const int        num_elem_x3,
    const int        num_solpts,
    const int        verbose) {

  const num_t* q_ptr        = get_raw_ptr<const num_t>(q_in);
  num_t*       flux_x1_ptr  = get_raw_ptr<num_t>(flux_x1);
  num_t*       flux_x2_ptr  = get_raw_ptr<num_t>(flux_x2);
  num_t*       flux_x3_ptr  = get_raw_ptr<num_t>(flux_x3);
  num_t*       pressure_ptr = get_raw_ptr<num_t>(pressure);

  num_t* wflux_adv_x1_ptr = get_raw_ptr<num_t>(wflux_adv_x1);
  num_t* wflux_adv_x2_ptr = get_raw_ptr<num_t>(wflux_adv_x2);
  num_t* wflux_adv_x3_ptr = get_raw_ptr<num_t>(wflux_adv_x3);

  num_t*        wflux_pres_x1_ptr = get_raw_ptr<num_t>(wflux_pres_x1);
  num_t*        wflux_pres_x2_ptr = get_raw_ptr<num_t>(wflux_pres_x2);
  num_t*        wflux_pres_x3_ptr = get_raw_ptr<num_t>(wflux_pres_x3);
  num_t*        log_pressure_ptr  = get_raw_ptr<num_t>(log_pressure);
  const real_t* sqrt_g_ptr        = get_raw_ptr<const real_t>(sqrt_g_in);
  const real_t* h_ptr             = get_raw_ptr<const real_t>(h_in);

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
    const py::object q_in,
    const py::object sqrt_g_in,
    const py::object h_in,
    py::object       flux_x1,
    py::object       flux_x2,
    py::object       flux_x3,
    py::object       pressure,
    py::object       wflux_adv_x1,
    py::object       wflux_adv_x2,
    py::object       wflux_adv_x3,
    py::object       wflux_pres_x1,
    py::object       wflux_pres_x2,
    py::object       wflux_pres_x3,
    py::object       log_pressure,
    const int        num_elem_x1,
    const int        num_elem_x2,
    const int        num_elem_x3,
    const int        num_solpts,
    const int        verbose) {

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

template <typename real_t, typename num_t>
void launch_riemann_euler_cubedsphere_rusanov_3d(
    const py::object q_itf_x1_in,
    const py::object q_itf_x2_in,
    py::object       q_itf_x3_in,
    const py::object sqrt_g_itf_x1,
    const py::object sqrt_g_itf_x2,
    const py::object sqrt_g_itf_x3,
    const py::object h_x1,
    const py::object h_x2,
    const py::object h_x3,
    const int        num_elem_x1,
    const int        num_elem_x2,
    const int        num_elem_x3,
    const int        num_solpts,
    py::object       flux_itf_x1,
    py::object       flux_itf_x2,
    py::object       flux_itf_x3,
    py::object       pressure_itf_x1,
    py::object       pressure_itf_x2,
    py::object       pressure_itf_x3,
    py::object       wflux_adv_itf_x1,
    py::object       wflux_pres_itf_x1,
    py::object       wflux_adv_itf_x2,
    py::object       wflux_pres_itf_x2,
    py::object       wflux_adv_itf_x3,
    py::object       wflux_pres_itf_x3) {
  const num_t* q_itf_x1_ptr = get_raw_ptr<const num_t>(q_itf_x1_in);
  const num_t* q_itf_x2_ptr = get_raw_ptr<const num_t>(q_itf_x2_in);
  num_t*       q_itf_x3_ptr = get_raw_ptr<num_t>(q_itf_x3_in);

  const real_t* sqrt_g_itf_x1_ptr = get_raw_ptr<const real_t>(sqrt_g_itf_x1);
  const real_t* sqrt_g_itf_x2_ptr = get_raw_ptr<const real_t>(sqrt_g_itf_x2);
  const real_t* sqrt_g_itf_x3_ptr = get_raw_ptr<const real_t>(sqrt_g_itf_x3);

  const real_t* h_x1_ptr = get_raw_ptr<const real_t>(h_x1);
  const real_t* h_x2_ptr = get_raw_ptr<const real_t>(h_x2);
  const real_t* h_x3_ptr = get_raw_ptr<const real_t>(h_x3);

  num_t* flux_itf_x1_ptr = get_raw_ptr<num_t>(flux_itf_x1);
  num_t* flux_itf_x2_ptr = get_raw_ptr<num_t>(flux_itf_x2);
  num_t* flux_itf_x3_ptr = get_raw_ptr<num_t>(flux_itf_x3);

  num_t* pressure_itf_x1_ptr = get_raw_ptr<num_t>(pressure_itf_x1);
  num_t* pressure_itf_x2_ptr = get_raw_ptr<num_t>(pressure_itf_x2);
  num_t* pressure_itf_x3_ptr = get_raw_ptr<num_t>(pressure_itf_x3);

  num_t* wflux_adv_itf_x1_ptr = get_raw_ptr<num_t>(wflux_adv_itf_x1);
  num_t* wflux_adv_itf_x2_ptr = get_raw_ptr<num_t>(wflux_adv_itf_x2);
  num_t* wflux_adv_itf_x3_ptr = get_raw_ptr<num_t>(wflux_adv_itf_x3);

  num_t* wflux_pres_itf_x1_ptr = get_raw_ptr<num_t>(wflux_pres_itf_x1);
  num_t* wflux_pres_itf_x2_ptr = get_raw_ptr<num_t>(wflux_pres_itf_x2);
  num_t* wflux_pres_itf_x3_ptr = get_raw_ptr<num_t>(wflux_pres_itf_x3);

  dim3 threads_per_block(256);
  int  num_solpts_face = num_solpts * num_solpts;
  int  num_tasks, num_blocks;

  // x1-direction
  num_tasks  = num_elem_x3 * num_elem_x2 * (num_elem_x1 + 1) * num_solpts_face;
  num_blocks = ((num_tasks + threads_per_block.x - 1) / threads_per_block.x);

  riemann_euler_cubedsphere_rusanov_3d<real_t, num_t><<<num_blocks, threads_per_block>>>(
      q_itf_x1_ptr,
      sqrt_g_itf_x1_ptr,
      h_x1_ptr,
      flux_itf_x1_ptr,
      pressure_itf_x1_ptr,
      wflux_adv_itf_x1_ptr,
      wflux_pres_itf_x1_ptr,
      0,
      num_elem_x1 + 1,
      num_elem_x2,
      num_elem_x3,
      num_solpts_face);

  // x2-direction
  num_tasks  = num_elem_x3 * (num_elem_x2 + 1) * num_elem_x1 * num_solpts_face;
  num_blocks = ((num_tasks + threads_per_block.x - 1) / threads_per_block.x);

  riemann_euler_cubedsphere_rusanov_3d<real_t, num_t><<<num_blocks, threads_per_block>>>(
      q_itf_x2_ptr,
      sqrt_g_itf_x2_ptr,
      h_x2_ptr,
      flux_itf_x2_ptr,
      pressure_itf_x2_ptr,
      wflux_adv_itf_x2_ptr,
      wflux_pres_itf_x2_ptr,
      1,
      num_elem_x1,
      num_elem_x2 + 1,
      num_elem_x3,
      num_solpts_face);

  // Set boundary conditions along the vertical direction
  num_tasks  = num_elem_x2 * num_elem_x1 * num_solpts_face;
  num_blocks = ((num_tasks + threads_per_block.x - 1) / threads_per_block.x);

  boundary_euler_cubedsphere_3d<real_t, num_t><<<num_blocks, threads_per_block>>>(
      q_itf_x3_ptr,
      num_elem_x1,
      num_elem_x2,
      num_elem_x3,
      num_solpts_face);

  // x3-direction
  num_tasks  = (num_elem_x3 + 1) * num_elem_x2 * num_elem_x1 * num_solpts_face;
  num_blocks = ((num_tasks + threads_per_block.x - 1) / threads_per_block.x);

  riemann_euler_cubedsphere_rusanov_3d<real_t, num_t><<<num_blocks, threads_per_block>>>(
      q_itf_x3_ptr,
      sqrt_g_itf_x3_ptr,
      h_x3_ptr,
      flux_itf_x3_ptr,
      pressure_itf_x3_ptr,
      wflux_adv_itf_x3_ptr,
      wflux_pres_itf_x3_ptr,
      2,
      num_elem_x1,
      num_elem_x2,
      num_elem_x3 + 1,
      num_solpts_face);
}

void select_riemann_euler_cubedsphere_rusanov_3d(
    const py::object q_itf_x1_in,
    const py::object q_itf_x2_in,
    py::object       q_itf_x3_in,
    const py::object sqrt_g_itf_x1,
    const py::object sqrt_g_itf_x2,
    const py::object sqrt_g_itf_x3,
    const py::object h_x1,
    const py::object h_x2,
    const py::object h_x3,
    const int        num_elem_x1,
    const int        num_elem_x2,
    const int        num_elem_x3,
    const int        num_solpts,
    py::object       flux_itf_x1,
    py::object       flux_itf_x2,
    py::object       flux_itf_x3,
    py::object       pressure_itf_x1,
    py::object       pressure_itf_x2,
    py::object       pressure_itf_x3,
    py::object       wflux_adv_itf_x1,
    py::object       wflux_pres_itf_x1,
    py::object       wflux_adv_itf_x2,
    py::object       wflux_pres_itf_x2,
    py::object       wflux_adv_itf_x3,
    py::object       wflux_pres_itf_x3) {

  std::string dtype = py::str(q_itf_x1_in.attr("dtype").attr("name"));
  if (dtype == "float64")
  {
    launch_riemann_euler_cubedsphere_rusanov_3d<double, double>(
        q_itf_x1_in,
        q_itf_x2_in,
        q_itf_x3_in,
        sqrt_g_itf_x1,
        sqrt_g_itf_x2,
        sqrt_g_itf_x3,
        h_x1,
        h_x2,
        h_x3,
        num_elem_x1,
        num_elem_x2,
        num_elem_x3,
        num_solpts,
        flux_itf_x1,
        flux_itf_x2,
        flux_itf_x3,
        pressure_itf_x1,
        pressure_itf_x2,
        pressure_itf_x3,
        wflux_adv_itf_x1,
        wflux_pres_itf_x1,
        wflux_adv_itf_x2,
        wflux_pres_itf_x2,
        wflux_adv_itf_x3,
        wflux_pres_itf_x3);
  }
  else if (dtype == "complex128")
  {
    launch_riemann_euler_cubedsphere_rusanov_3d<double, complex_t>(
        q_itf_x1_in,
        q_itf_x2_in,
        q_itf_x3_in,
        sqrt_g_itf_x1,
        sqrt_g_itf_x2,
        sqrt_g_itf_x3,
        h_x1,
        h_x2,
        h_x3,
        num_elem_x1,
        num_elem_x2,
        num_elem_x3,
        num_solpts,
        flux_itf_x1,
        flux_itf_x2,
        flux_itf_x3,
        pressure_itf_x1,
        pressure_itf_x2,
        pressure_itf_x3,
        wflux_adv_itf_x1,
        wflux_pres_itf_x1,
        wflux_adv_itf_x2,
        wflux_pres_itf_x2,
        wflux_adv_itf_x3,
        wflux_pres_itf_x3);
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
  const num_t*  q           = get_raw_ptr<const num_t>(q_in);
  const num_t*  pressure    = get_raw_ptr<const num_t>(pressure_in);
  const real_t* sqrt_g      = get_raw_ptr<const real_t>(sqrt_g_in);
  const real_t* h           = get_raw_ptr<const real_t>(h_in);
  const real_t* christoffel = get_raw_ptr<const real_t>(christoffel_in);
  num_t*        forcing     = get_raw_ptr<num_t>(forcing_in);

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
  m.def("pointwise_euler_cubedsphere_3d", &select_pointwise_euler_cubedsphere_3d);

  // Riemann fluxes
  m.def("riemann_eulercartesian_ausm_2d", &select_riemann_eulercartesian_ausm_2d);
  m.def(
      "riemann_euler_cubedsphere_rusanov_3d",
      &select_riemann_euler_cubedsphere_rusanov_3d);

  // Forcing
  m.def("forcing_euler_cubesphere_3d", &select_forcing_euler_cubesphere_3d);
}
