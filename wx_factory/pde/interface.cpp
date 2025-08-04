#include "interface.hpp"

#include <iostream>

#include "kernels/kernels.h"

namespace py = pybind11;

#ifdef WX_OMP
#include <omp.h>
template <typename T>
using py_array = py::object;
#define MODULE_NAME operators_omp
#else
template <typename T>
using py_array = py::array_t<T>;
#define MODULE_NAME operators_cpp
#endif

// -------------------------------------
// Pointwise fluxes
// -------------------------------------

template <typename num_t>
void pointwise_eulercartesian_2d(
    const py_array<num_t>& q_in,
    py_array<num_t>&       flux_x1,
    py_array<num_t>&       flux_x2,
    const int              num_elem_x1,
    const int              num_elem_x2,
    const int              num_solpts_tot) {
  py::buffer_info buf1 = q_in.request();
  py::buffer_info buf2 = flux_x1.request();
  py::buffer_info buf3 = flux_x2.request();

  // Get the pointers
  num_t* q_ptr       = static_cast<num_t*>(buf1.ptr);
  num_t* flux_x1_ptr = static_cast<num_t*>(buf2.ptr);
  num_t* flux_x2_ptr = static_cast<num_t*>(buf3.ptr);

  const int stride         = num_elem_x1 * num_elem_x2 * num_solpts_tot;
  const int array_shape[4] = {4, num_elem_x2, num_elem_x1, num_solpts_tot};

  for (int i = 0; i < num_elem_x2; i++)
  {
    for (int j = 0; j < num_elem_x1; j++)
    {
      for (int k = 0; k < num_solpts_tot; k++)
      {
        const int ind = get_c_index(0, i, j, k, array_shape);

        // Store variables and pointers to compute the fluxes
        kernel_params<num_t, euler_state_2d>
            params(q_ptr, flux_x1_ptr, flux_x2_ptr, nullptr, ind, stride);

        // Call the pointwise flux kernel
        pointwise_eulercartesian_2d_kernel(params);
      }
    }
  }
}

template <typename real_t, typename num_t>
void pointwise_euler_cubedsphere_3d(
    const py_array<num_t>&  q_in,
    const py_array<real_t>& sqrt_g_in,
    const py_array<real_t>& h_in,
    py_array<num_t>&        flux_x1,
    py_array<num_t>&        flux_x2,
    py_array<num_t>&        flux_x3,
    py_array<num_t>&        pressure,
    py_array<num_t>&        wflux_adv_x1,
    py_array<num_t>&        wflux_adv_x2,
    py_array<num_t>&        wflux_adv_x3,
    py_array<num_t>&        wflux_pres_x1,
    py_array<num_t>&        wflux_pres_x2,
    py_array<num_t>&        wflux_pres_x3,
    py_array<num_t>&        log_pressure,
    const int               num_elem_x1,
    const int               num_elem_x2,
    const int               num_elem_x3,
    const int               num_solpts_tot,
    const int               verbose) {

  const num_t* q_ptr        = get_raw_ptr<num_t>(q_in);
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
  const real_t* sqrt_g_ptr        = get_raw_ptr<real_t>(sqrt_g_in);
  const real_t* h_ptr             = get_raw_ptr<real_t>(h_in);

  const uint64_t stride    = num_elem_x3 * num_elem_x2 * num_elem_x1 * num_solpts_tot;
  const int array_shape[5] = {5, num_elem_x3, num_elem_x2, num_elem_x1, num_solpts_tot};

#pragma omp target teams distribute collapse(4)
  for (int i = 0; i < num_elem_x3; i++)
  {
    for (int j = 0; j < num_elem_x2; j++)
    {
      for (int k = 0; k < num_elem_x1; k++)
      {
        for (int s = 0; s < num_solpts_tot; s++)
        {
          const int index = get_c_index(0, i, j, k, s, array_shape);

          kernel_params_cubedsphere<real_t, num_t> params(
              q_ptr,
              sqrt_g_ptr,
              h_ptr,
              index,
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

          pointwise_euler_cubedsphere_3d_kernel(params, verbose);
        }
      }
    }
  }
}

// -------------------------------------
// Riemann fluxes
// -------------------------------------

template <typename num_t>
void riemann_eulercartesian_ausm_2d(
    const py_array<num_t>& q_itf_x1_in,
    const py_array<num_t>& q_itf_x2_in,
    py_array<num_t>&       flux_itf_x1_in,
    py_array<num_t>&       flux_itf_x2_in,
    const int              num_elem_x1,
    const int              num_elem_x2,
    const int              num_solpts) {
  py::buffer_info buf1 = q_itf_x1_in.request();
  py::buffer_info buf2 = q_itf_x2_in.request();
  py::buffer_info buf3 = flux_itf_x1_in.request();
  py::buffer_info buf4 = flux_itf_x2_in.request();

  // Get the pointers
  num_t* q_itf_x1    = static_cast<num_t*>(buf1.ptr);
  num_t* q_itf_x2    = static_cast<num_t*>(buf2.ptr);
  num_t* flux_itf_x1 = static_cast<num_t*>(buf3.ptr);
  num_t* flux_itf_x2 = static_cast<num_t*>(buf4.ptr);

  const int num_solpts_riem = 2 * num_solpts;
  const int stride          = num_elem_x1 * num_elem_x2 * num_solpts_riem;
  const int array_shape[4]  = {4, num_elem_x2, num_elem_x1, num_solpts_riem};

  for (int i = 0; i < num_elem_x2; i++)
  {
    for (int j = 0; j < num_elem_x1; j++)
    {
      // Solve along the horizontal  direction
      if (j + 1 < num_elem_x1)
      {
        for (int k = 0; k < num_solpts; k++)
        {
          // Initialize left-hand-side parameters
          const int indl = get_c_index(0, i, j, num_solpts + k, array_shape);
          kernel_params<num_t, euler_state_2d>
              params_l(q_itf_x1, flux_itf_x1, nullptr, nullptr, indl, stride);

          // Initialize right-hand-size parameters
          const int indr = get_c_index(0, i, j + 1, k, array_shape);
          kernel_params<num_t, euler_state_2d>
              params_r(q_itf_x1, flux_itf_x1, nullptr, nullptr, indr, stride);

          // Call Riemann kernel on the horizontal direction
          riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, 0);
        }
      }

      // Solve the Riemann problem along the vertical direction
      if (i + 1 < num_elem_x2)
      {
        for (int k = 0; k < num_solpts; k++)
        {
          // Initialize left-hand-side parameters
          const int indl = get_c_index(0, i, j, num_solpts + k, array_shape);
          kernel_params<num_t, euler_state_2d>
              params_l(q_itf_x2, nullptr, flux_itf_x2, nullptr, indl, stride);

          // Initialize right-hand-size parameters
          const int indr = get_c_index(0, i + 1, j, k, array_shape);
          kernel_params<num_t, euler_state_2d>
              params_r(q_itf_x2, nullptr, flux_itf_x2, nullptr, indr, stride);

          // Call Riemann kernel on the vertical direction
          riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, 1);
        }
      }
    }
  }

  // Update boundary conditions

  // Set the boundary fluxes along the horizontal direction
  for (int i = 0; i < num_elem_x2; i++)
  {
    for (int j = 0; j < num_solpts; j++)
    {
      // Set the fluxes on the left boundary
      const int indl = get_c_index(0, i, 0, j, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_l(q_itf_x1, flux_itf_x1, nullptr, nullptr, indl, stride);
      boundary_eulercartesian_2d_kernel(params_l, 0);

      // Set the fluxes on the right boundary
      const int indr = get_c_index(0, i, num_elem_x1 - 1, j + num_solpts, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_r(q_itf_x1, flux_itf_x1, nullptr, nullptr, indr, stride);
      boundary_eulercartesian_2d_kernel(params_r, 0);
    }
  }

  // Set the boundary fluxes along the vertical direction
  for (int i = 0; i < num_elem_x1; i++)
  {
    for (int j = 0; j < num_solpts; j++)
    {
      // Set the fluxes on the bottom boundary
      const int indb = get_c_index(0, 0, i, j, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_b(q_itf_x2, nullptr, flux_itf_x2, nullptr, indb, stride);
      boundary_eulercartesian_2d_kernel(params_b, 1);

      // Set the fluxes on the top boundary
      const int indt = get_c_index(0, num_elem_x2 - 1, i, j + num_solpts, array_shape);
      kernel_params<num_t, euler_state_2d>
          params_t(q_itf_x2, nullptr, flux_itf_x2, nullptr, indt, stride);
      boundary_eulercartesian_2d_kernel(params_t, 1);
    }
  }
}

template <typename real_t, typename num_t>
void riemann_euler_cubedsphere_rusanov_3d(
    const py_array<num_t>&  q_itf_x1_in,
    const py_array<num_t>&  q_itf_x2_in,
    py_array<num_t>&        q_itf_x3_in,
    const py_array<real_t>& sqrt_g_itf_x1,
    const py_array<real_t>& sqrt_g_itf_x2,
    const py_array<real_t>& sqrt_g_itf_x3,
    const py_array<real_t>& h_x1,
    const py_array<real_t>& h_x2,
    const py_array<real_t>& h_x3,
    const int               num_elem_x1,
    const int               num_elem_x2,
    const int               num_elem_x3,
    const int               num_solpts,
    py_array<num_t>&        flux_itf_x1,
    py_array<num_t>&        flux_itf_x2,
    py_array<num_t>&        flux_itf_x3,
    py_array<num_t>&        pressure_itf_x1,
    py_array<num_t>&        pressure_itf_x2,
    py_array<num_t>&        pressure_itf_x3,
    py_array<num_t>&        wflux_adv_itf_x1,
    py_array<num_t>&        wflux_pres_itf_x1,
    py_array<num_t>&        wflux_adv_itf_x2,
    py_array<num_t>&        wflux_pres_itf_x2,
    py_array<num_t>&        wflux_adv_itf_x3,
    py_array<num_t>&        wflux_pres_itf_x3) {

  const num_t* q_itf_x1_ptr = get_raw_ptr<num_t>(q_itf_x1_in);
  const num_t* q_itf_x2_ptr = get_raw_ptr<num_t>(q_itf_x2_in);
  num_t*       q_itf_x3_ptr = get_raw_ptr<num_t>(q_itf_x3_in);

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

  const real_t* sqrt_g_itf_x1_ptr = get_raw_ptr<real_t>(sqrt_g_itf_x1);
  const real_t* sqrt_g_itf_x2_ptr = get_raw_ptr<real_t>(sqrt_g_itf_x2);
  const real_t* sqrt_g_itf_x3_ptr = get_raw_ptr<real_t>(sqrt_g_itf_x3);

  const real_t* h_x1_ptr = get_raw_ptr<real_t>(h_x1);
  const real_t* h_x2_ptr = get_raw_ptr<real_t>(h_x2);
  const real_t* h_x3_ptr = get_raw_ptr<real_t>(h_x3);

  const int      num_solpts_riem = 2 * num_solpts * num_solpts;
  const uint64_t stride_x1 =
      num_elem_x3 * num_elem_x2 * (num_elem_x1 + 2) * num_solpts_riem;
  const uint64_t stride_x2 =
      num_elem_x3 * (num_elem_x2 + 2) * num_elem_x1 * num_solpts_riem;
  const uint64_t stride_x3 =
      (num_elem_x3 + 2) * num_elem_x2 * num_elem_x1 * num_solpts_riem;

  // Ensure ghost elements are added to array shapes
  const int array_shape_x1[5] =
      {5, num_elem_x3, num_elem_x2, num_elem_x1 + 2, num_solpts_riem};
  const int array_shape_x2[5] =
      {5, num_elem_x3, num_elem_x2 + 2, num_elem_x1, num_solpts_riem};
  const int array_shape_x3[5] =
      {5, num_elem_x3 + 2, num_elem_x2, num_elem_x1, num_solpts_riem};

  // Compute the fluxes along the x1-direction
#pragma omp target teams distribute collapse(4)
  for (int i = 0; i < num_elem_x3; i++)
  {
    for (int j = 0; j < num_elem_x2; j++)
    {
      for (int k = 0; k < num_elem_x1 + 1; k++)
      {
        for (int l = 0; l < num_solpts * num_solpts; l++)
        {
          const int index_l =
              get_c_index(0, i, j, k, l + num_solpts * num_solpts, array_shape_x1);
          riemann_params_cubedsphere<real_t, num_t> params_l(
              q_itf_x1_ptr,
              sqrt_g_itf_x1_ptr,
              h_x1_ptr,
              index_l,
              stride_x1,
              flux_itf_x1_ptr,
              pressure_itf_x1_ptr,
              wflux_adv_itf_x1_ptr,
              wflux_pres_itf_x1_ptr);

          const int index_r = get_c_index(0, i, j, k + 1, l, array_shape_x1);
          riemann_params_cubedsphere<real_t, num_t> params_r(
              q_itf_x1_ptr,
              sqrt_g_itf_x1_ptr,
              h_x1_ptr,
              index_r,
              stride_x1,
              flux_itf_x1_ptr,
              pressure_itf_x1_ptr,
              wflux_adv_itf_x1_ptr,
              wflux_pres_itf_x1_ptr);

          riemann_euler_cubedsphere_rusanov_3d_kernel<real_t, num_t>(
              params_l,
              params_r,
              0,
              false); // Consider internal Riemann problem
        }
      }
    }
  }

  // Compute the fluxes along the x2-direction
#pragma omp target teams distribute collapse(4)
  for (int i = 0; i < num_elem_x3; i++)
  {
    for (int j = 0; j < num_elem_x2 + 1; j++)
    {
      for (int k = 0; k < num_elem_x1; k++)
      {
        for (int l = 0; l < num_solpts * num_solpts; l++)
        {
          const int index_l =
              get_c_index(0, i, j, k, l + num_solpts * num_solpts, array_shape_x2);
          riemann_params_cubedsphere<real_t, num_t> params_l(
              q_itf_x2_ptr,
              sqrt_g_itf_x2_ptr,
              h_x2_ptr,
              index_l,
              stride_x2,
              flux_itf_x2_ptr,
              pressure_itf_x2_ptr,
              wflux_adv_itf_x2_ptr,
              wflux_pres_itf_x2_ptr);

          const int index_r = get_c_index(0, i, j + 1, k, l, array_shape_x2);
          riemann_params_cubedsphere<real_t, num_t> params_r(
              q_itf_x2_ptr,
              sqrt_g_itf_x2_ptr,
              h_x2_ptr,
              index_r,
              stride_x2,
              flux_itf_x2_ptr,
              pressure_itf_x2_ptr,
              wflux_adv_itf_x2_ptr,
              wflux_pres_itf_x2_ptr);

          riemann_euler_cubedsphere_rusanov_3d_kernel<real_t, num_t>(
              params_l,
              params_r,
              1,
              false); // Consider internal Riemann problem
        }
      }
    }
  }

  // Set the x3-direction boundary conditions to ensure no flow via odd symmetry
#pragma omp target teams distribute collapse(3)
  for (int j = 0; j < num_elem_x2; j++)
  {
    for (int k = 0; k < num_elem_x1; k++)
    {
      for (int l = 0; l < num_solpts * num_solpts; l++)
      {
        // Set the bottom boundary
        const int index_b_bottom =
            get_c_index(0, 0, j, k, l + num_solpts * num_solpts, array_shape_x3);
        euler_state_3d<num_t> params_b_bottom(q_itf_x3_ptr, index_b_bottom, stride_x3);

        const int index_in_bottom = get_c_index(0, 1, j, k, l, array_shape_x3);
        euler_state_3d<const num_t> params_in_bottom(
            q_itf_x3_ptr,
            index_in_bottom,
            stride_x3);

        boundary_euler_cubedsphere_3d_kernel<real_t, num_t>(
            params_in_bottom,
            params_b_bottom);

        // Set the top boundary
        const int index_b_top = get_c_index(0, num_elem_x3 + 1, j, k, l, array_shape_x3);
        euler_state_3d<num_t> params_b_top(q_itf_x3_ptr, index_b_top, stride_x3);

        const int index_in_top = get_c_index(
            0,
            num_elem_x3,
            j,
            k,
            l + num_solpts * num_solpts,
            array_shape_x3);
        euler_state_3d<const num_t> params_in_top(q_itf_x3_ptr, index_in_top, stride_x3);

        boundary_euler_cubedsphere_3d_kernel<real_t, num_t>(params_in_top, params_b_top);
      }
    }
  }

  // Compute the fluxes along the x3-direction
#pragma omp target teams distribute collapse(4)
  for (int i = 0; i < num_elem_x3 + 1; i++)
  {
    for (int j = 0; j < num_elem_x2; j++)
    {
      for (int k = 0; k < num_elem_x1; k++)
      {
        for (int l = 0; l < num_solpts * num_solpts; l++)
        {
          const int index_l =
              get_c_index(0, i, j, k, l + num_solpts * num_solpts, array_shape_x3);
          riemann_params_cubedsphere<real_t, num_t> params_l(
              q_itf_x3_ptr,
              sqrt_g_itf_x3_ptr,
              h_x3_ptr,
              index_l,
              stride_x3,
              flux_itf_x3_ptr,
              pressure_itf_x3_ptr,
              wflux_adv_itf_x3_ptr,
              wflux_pres_itf_x3_ptr);

          const int index_r = get_c_index(0, i + 1, j, k, l, array_shape_x3);
          riemann_params_cubedsphere<real_t, num_t> params_r(
              q_itf_x3_ptr,
              sqrt_g_itf_x3_ptr,
              h_x3_ptr,
              index_r,
              stride_x3,
              flux_itf_x3_ptr,
              pressure_itf_x3_ptr,
              wflux_adv_itf_x3_ptr,
              wflux_pres_itf_x3_ptr);

          bool boundary_riemann = false;
          if (i == 0 || i == num_elem_x3)
            boundary_riemann = true;

          riemann_euler_cubedsphere_rusanov_3d_kernel<real_t, num_t>(
              params_l,
              params_r,
              2,
              boundary_riemann);
        }
      }
    }
  }
}

template <typename real_t, typename num_t>
void forcing_euler_cubesphere_3d(
    const py_array<num_t>&  q_in,
    const py_array<num_t>&  pressure_in,
    const py_array<real_t>& sqrt_g_in,
    const py_array<real_t>& h_in,
    const py_array<real_t>& christoffel_in,
    py_array<num_t>&        forcing_in,
    const int               num_elem_x1,
    const int               num_elem_x2,
    const int               num_elem_x3,
    const int               num_solpts,
    const int               verbose) {

  const num_t*  q           = get_raw_ptr<num_t>(q_in);
  const num_t*  pressure    = get_raw_ptr<num_t>(pressure_in);
  const real_t* sqrt_g      = get_raw_ptr<real_t>(sqrt_g_in);
  const real_t* h           = get_raw_ptr<real_t>(h_in);
  const real_t* christoffel = get_raw_ptr<real_t>(christoffel_in);
  num_t*        forcing     = get_raw_ptr<num_t>(forcing_in);

  const uint64_t stride = num_elem_x3 * num_elem_x2 * num_elem_x1 * num_solpts;

#pragma omp target teams distribute collapse(4)                                          \
    is_device_ptr(q, pressure, sqrt_g, h, christoffel)
  for (int i = 0; i < num_elem_x3; i++)
  {
    for (int j = 0; j < num_elem_x2; j++)
    {
      for (int k = 0; k < num_elem_x1; k++)
      {
        for (int s = 0; s < num_solpts; s++)
        {
          const int index = ((i * num_elem_x2 + j) * num_elem_x1 + k) * num_solpts + s;
          forcing_params<real_t, num_t>
              p(q, pressure, sqrt_g, h, christoffel, forcing, index, stride);
          forcing_euler_cubesphere_3d_kernel(p, verbose);
        }
      }
    }
  }
}

#ifdef WX_OMP

void select_pointwise_euler_cubedsphere_3d(
    const py::object& q_in,
    const py::object& sqrt_g_in,
    const py::object& h_in,
    py::object&       flux_x1,
    py::object&       flux_x2,
    py::object&       flux_x3,
    py::object&       pressure,
    py::object&       wflux_adv_x1,
    py::object&       wflux_adv_x2,
    py::object&       wflux_adv_x3,
    py::object&       wflux_pres_x1,
    py::object&       wflux_pres_x2,
    py::object&       wflux_pres_x3,
    py::object&       log_pressure,
    const int         num_elem_x1,
    const int         num_elem_x2,
    const int         num_elem_x3,
    const int         num_solpts_tot,
    const int         verbose) {

  std::string dtype = py::str(q_in.attr("dtype").attr("name"));
  if (dtype == "float64")
  {
    pointwise_euler_cubedsphere_3d<double, double>(
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
        num_solpts_tot,
        verbose);
  }
  else if (dtype == "complex128")
  {
    pointwise_euler_cubedsphere_3d<double, complex_t>(
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
        num_solpts_tot,
        verbose);
  }
  else
  {
    std::cerr << __func__ << ": Unrecognized array type " << dtype << std::endl;
  }
}

void select_riemann_euler_cubedsphere_rusanov_3d(
    const py::object& q_itf_x1_in,
    const py::object& q_itf_x2_in,
    py::object&       q_itf_x3_in,
    const py::object& sqrt_g_itf_x1,
    const py::object& sqrt_g_itf_x2,
    const py::object& sqrt_g_itf_x3,
    const py::object& h_x1,
    const py::object& h_x2,
    const py::object& h_x3,
    const int         num_elem_x1,
    const int         num_elem_x2,
    const int         num_elem_x3,
    const int         num_solpts,
    py::object&       flux_itf_x1,
    py::object&       flux_itf_x2,
    py::object&       flux_itf_x3,
    py::object&       pressure_itf_x1,
    py::object&       pressure_itf_x2,
    py::object&       pressure_itf_x3,
    py::object&       wflux_adv_itf_x1,
    py::object&       wflux_pres_itf_x1,
    py::object&       wflux_adv_itf_x2,
    py::object&       wflux_pres_itf_x2,
    py::object&       wflux_adv_itf_x3,
    py::object&       wflux_pres_itf_x3) {

  std::string dtype = py::str(q_itf_x1_in.attr("dtype").attr("name"));
  if (dtype == "float64")
  {
    riemann_euler_cubedsphere_rusanov_3d<double, double>(
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
    riemann_euler_cubedsphere_rusanov_3d<double, complex_t>(
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
    std::cerr << __func__ << ": Unrecognized array type " << dtype << std::endl;
  }
}

void select_forcing_euler_cubesphere_3d(
    const py::object& q_in,
    const py::object& pressure_in,
    const py::object& sqrt_g_in,
    const py::object& h_in,
    const py::object& christoffel_in,
    py::object&       forcing_in,
    const int         num_elem_x1,
    const int         num_elem_x2,
    const int         num_elem_x3,
    const int         num_solpts,
    const int         verbose) {

  std::string dtype = py::str(q_in.attr("dtype").attr("name"));
  if (dtype == "float64")
  {
    forcing_euler_cubesphere_3d<double, double>(
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
    forcing_euler_cubesphere_3d<double, complex_t>(
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
    std::cerr << __func__ << ": Unrecognized array type " << dtype << std::endl;
  }
}

void set_omp_device(const int device_id) {
  omp_set_default_device(device_id);
}

PYBIND11_MODULE(pde_omp, m) {
  m.def("pointwise_euler_cubedsphere_3d", &select_pointwise_euler_cubedsphere_3d);
  m.def(
      "riemann_euler_cubedsphere_rusanov_3d",
      &select_riemann_euler_cubedsphere_rusanov_3d);
  // The OpenMP offload forcing kernel seems slower than cupy
  // m.def("forcing_euler_cubesphere_3d", &select_forcing_euler_cubesphere_3d);
  m.def("set_omp_device", &set_omp_device);
}

#else // WX_OMP

PYBIND11_MODULE(pde_cpp, m) {
  // Pointwise fluxes
  m.def("pointwise_eulercartesian_2d", &pointwise_eulercartesian_2d<double>);
  m.def("pointwise_eulercartesian_2d", &pointwise_eulercartesian_2d<complex_t>);

  m.def(
      "pointwise_euler_cubedsphere_3d",
      &pointwise_euler_cubedsphere_3d<double, double>);
  m.def(
      "pointwise_euler_cubedsphere_3d",
      &pointwise_euler_cubedsphere_3d<double, complex_t>);

  // Riemann fluxes
  m.def("riemann_eulercartesian_ausm_2d", &riemann_eulercartesian_ausm_2d<double>);
  m.def("riemann_eulercartesian_ausm_2d", &riemann_eulercartesian_ausm_2d<complex_t>);
  m.def(
      "riemann_euler_cubedsphere_rusanov_3d",
      &riemann_euler_cubedsphere_rusanov_3d<double, double>);
  m.def(
      "riemann_euler_cubedsphere_rusanov_3d",
      &riemann_euler_cubedsphere_rusanov_3d<double, complex_t>);

  // Forcing functions
  m.def("forcing_euler_cubesphere_3d", &forcing_euler_cubesphere_3d<double, double>);
  m.def("forcing_euler_cubesphere_3d", &forcing_euler_cubesphere_3d<double, complex_t>);
}

#endif // WX_OMP
