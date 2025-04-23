#include "interface.hpp"

#include "kernels/kernels.h"

namespace py = pybind11;

// -------------------------------------
// Pointwise fluxes
// -------------------------------------

template <typename num_t>
void pointwise_eulercartesian_2d(
    const py::array_t<num_t>& q_in,
    py::array_t<num_t>&       flux_x1,
    py::array_t<num_t>&       flux_x2,
    const int                 num_elem_x1,
    const int                 num_elem_x2,
    const int                 num_solpts_tot) {
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
    const py::array_t<num_t>&  q_in,
    const py::array_t<real_t>& sqrt_g_in,
    const py::array_t<real_t>& h_in,
    py::array_t<num_t>&        flux_x1,
    py::array_t<num_t>&        flux_x2,
    py::array_t<num_t>&        flux_x3,
    py::array_t<num_t>&        pressure,
    py::array_t<num_t>&        wflux_adv_x1,
    py::array_t<num_t>&        wflux_adv_x2,
    py::array_t<num_t>&        wflux_adv_x3,
    py::array_t<num_t>&        wflux_pres_x1,
    py::array_t<num_t>&        wflux_pres_x2,
    py::array_t<num_t>&        wflux_pres_x3,
    py::array_t<num_t>&        log_pressure,
    const int                  num_elem_x1,
    const int                  num_elem_x2,
    const int                  num_elem_x3,
    const int                  num_solpts,
    const int                  verbose) {

  const num_t* q_ptr        = get_c_ptr(q_in);
  num_t*       flux_x1_ptr  = get_c_ptr(flux_x1);
  num_t*       flux_x2_ptr  = get_c_ptr(flux_x2);
  num_t*       flux_x3_ptr  = get_c_ptr(flux_x3);
  num_t*       pressure_ptr = get_c_ptr(pressure);

  num_t* wflux_adv_x1_ptr = get_c_ptr(wflux_adv_x1);
  num_t* wflux_adv_x2_ptr = get_c_ptr(wflux_adv_x2);
  num_t* wflux_adv_x3_ptr = get_c_ptr(wflux_adv_x3);

  num_t*        wflux_pres_x1_ptr = get_c_ptr(wflux_pres_x1);
  num_t*        wflux_pres_x2_ptr = get_c_ptr(wflux_pres_x2);
  num_t*        wflux_pres_x3_ptr = get_c_ptr(wflux_pres_x3);
  num_t*        log_pressure_ptr  = get_c_ptr(log_pressure);
  const real_t* sqrt_g_ptr        = get_c_ptr(sqrt_g_in);
  const real_t* h_ptr             = get_c_ptr(h_in);

  const uint64_t stride = num_elem_x3 * num_elem_x2 * num_elem_x1 * num_solpts;
  const int array_shape[5]  = {5, num_elem_x3, num_elem_x2, num_elem_x1, num_solpts * num_solpts};

  for (int i = 0; i < num_elem_x3; i++)
  {
    for (int j = 0; j < num_elem_x2; j++)
    {
      for (int k = 0; k < num_elem_x1; k++)
      {
        for (int s = 0; s < num_solpts; s++)
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
    const py::array_t<num_t>& q_itf_x1_in,
    const py::array_t<num_t>& q_itf_x2_in,
    py::array_t<num_t>&       flux_itf_x1_in,
    py::array_t<num_t>&       flux_itf_x2_in,
    const int                 num_elem_x1,
    const int                 num_elem_x2,
    const int                 num_solpts) {
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
    const py::array_t<num_t>& q_itf_x1_in,
    const py::array_t<num_t>& q_itf_x2_in,
    py::array_t<num_t>& q_itf_x3_in,
    const py::array_t<real_t>& sqrt_g_in,
    const py::array_t<real_t>& h_in,
    const int num_elem_x1,
    const int num_elem_x2,
    const int num_elem_x3,
    const int num_solpts,
    py::array_t<num_t>& flux_itf_x1,
    py::array_t<num_t>& flux_itf_x2,
    py::array_t<num_t>& flux_itf_x3,
    py::array_t<num_t>& pressure_itf_x1,
    py::array_t<num_t>& pressure_itf_x2,
    py::array_t<num_t>& pressure_itf_x3,
    py::array_t<num_t>& wflux_adv_itf_x1,
    py::array_t<num_t>& wflux_pres_itf_x1,
    py::array_t<num_t>& wflux_adv_itf_x2,
    py::array_t<num_t>& wflux_pres_itf_x2,
    py::array_t<num_t>& wflux_adv_itf_x3,
    py::array_t<num_t>& wflux_pres_itf_x3){

  const num_t* q_itf_x1_ptr = get_c_ptr(q_itf_x1_in);
  const num_t* q_itf_x2_ptr = get_c_ptr(q_itf_x2_in);
  num_t* q_itf_x3_ptr = get_c_ptr(q_itf_x3_in);

  num_t* flux_itf_x1_ptr = get_c_ptr(flux_itf_x1);
  num_t* flux_itf_x2_ptr = get_c_ptr(flux_itf_x2);
  num_t* flux_itf_x3_ptr = get_c_ptr(flux_itf_x3);

  num_t*  pressure_itf_x1_ptr = get_c_ptr(pressure_itf_x1);
  num_t*  pressure_itf_x2_ptr = get_c_ptr(pressure_itf_x2);
  num_t*  pressure_itf_x3_ptr = get_c_ptr(pressure_itf_x3);

  num_t* wflux_adv_itf_x1_ptr = get_c_ptr(wflux_adv_itf_x1);
  num_t* wflux_adv_itf_x2_ptr = get_c_ptr(wflux_adv_itf_x2);
  num_t* wflux_adv_itf_x3_ptr = get_c_ptr(wflux_adv_itf_x3);

  num_t* wflux_pres_itf_x1_ptr = get_c_ptr(wflux_pres_itf_x1);
  num_t* wflux_pres_itf_x2_ptr = get_c_ptr(wflux_pres_itf_x2);
  num_t* wflux_pres_itf_x3_ptr = get_c_ptr(wflux_pres_itf_x3);

  const real_t* sqrt_g_ptr        = get_c_ptr(sqrt_g_in);
  const real_t* h_ptr             = get_c_ptr(h_in);

  const int num_solpts_riem = 2 * num_solpts * num_solpts;
  const uint64_t stride = num_elem_x3 * num_elem_x2 * num_elem_x1 * num_solpts_riem;

  // Ensure ghost elements are added to array shapes
  const int array_shape_x1[5]  = {5, num_elem_x3, num_elem_x2, num_elem_x1+2, num_solpts_riem};
  const int array_shape_x2[5]  = {5, num_elem_x3, num_elem_x2+2, num_elem_x1, num_solpts_riem};
  const int array_shape_x3[5]  = {5, num_elem_x3+2, num_elem_x2, num_elem_x1, num_solpts_riem};

  // Compute the fluxes along the x1-direction
  for (int i = 0; i < num_elem_x3; i++)
  {
    for (int j = 0; j < num_elem_x2; j++)
    {
      for (int k = 0; k < num_elem_x1 + 1; k++)
      {
        for(int l=0 ; l < num_solpts * num_solpts ; l++)
        {
          const int index_l = get_c_index(0, i, j, k, l + num_solpts * num_solpts, array_shape_x1);
          kernel_params_cubedsphere<real_t, num_t> params_l(
            q_itf_x1_ptr,
            sqrt_g_ptr,
            h_ptr,
            index_l,
            stride,
            flux_itf_x1_ptr,
            nullptr,
            nullptr,
            pressure_itf_x1_ptr,
            wflux_adv_itf_x1_ptr,
            wflux_adv_itf_x2_ptr,
            wflux_adv_itf_x3_ptr,
            wflux_pres_itf_x1_ptr,
            wflux_pres_itf_x2_ptr,
            wflux_pres_itf_x3_ptr,
            nullptr);

          const int index_r = get_c_index(0, i, j, k + 1, l, array_shape_x1);
          kernel_params_cubedsphere<real_t, num_t> params_r(
            q_itf_x1_ptr,
            sqrt_g_ptr,
            h_ptr,
            index_r,
            stride,
            flux_itf_x1_ptr,
            nullptr,
            nullptr,
            pressure_itf_x1_ptr,
            wflux_adv_itf_x1_ptr,
            wflux_adv_itf_x2_ptr,
            wflux_adv_itf_x3_ptr,
            wflux_pres_itf_x1_ptr,
            wflux_pres_itf_x2_ptr,
            wflux_pres_itf_x3_ptr,
            nullptr);

          riemann_euler_cubedsphere_rusanov_3d_kernel<real_t, num_t>(params_l, params_r, 0);
        }
      }
    }
  }

  // Compute the fluxes along the x2-direction
  for (int i = 0; i < num_elem_x3; i++)
  {
    for (int j = 0; j < num_elem_x2 + 1; j++)
    {
      for (int k = 0; k < num_elem_x1; k++)
      {
        for(int l=0 ; l < num_solpts * num_solpts ; l++)
        {
          const int index_l = get_c_index(0, i, j, k, l + num_solpts * num_solpts, array_shape_x2);
          kernel_params_cubedsphere<real_t, num_t> params_l(
            q_itf_x2_ptr,
            sqrt_g_ptr,
            h_ptr,
            index_l,
            stride,
            nullptr,
            flux_itf_x2_ptr,
            nullptr,
            pressure_itf_x2_ptr,
            wflux_adv_itf_x1_ptr,
            wflux_adv_itf_x2_ptr,
            wflux_adv_itf_x3_ptr,
            wflux_pres_itf_x1_ptr,
            wflux_pres_itf_x2_ptr,
            wflux_pres_itf_x3_ptr,
            nullptr);

          const int index_r = get_c_index(0, i, j + 1, k, l, array_shape_x2);
          kernel_params_cubedsphere<real_t, num_t> params_r(
            q_itf_x2_ptr,
            sqrt_g_ptr,
            h_ptr,
            index_r,
            stride,
            nullptr,
            flux_itf_x2_ptr,
            nullptr,
            pressure_itf_x2_ptr,
            wflux_adv_itf_x1_ptr,
            wflux_adv_itf_x2_ptr,
            wflux_adv_itf_x3_ptr,
            wflux_pres_itf_x1_ptr,
            wflux_pres_itf_x2_ptr,
            wflux_pres_itf_x3_ptr,
            nullptr);

          riemann_euler_cubedsphere_rusanov_3d_kernel<real_t, num_t>(params_l, params_r, 1);
        }
      }
    }
  }

  // Set the x3-direction boundary conditions to ensure no flow via odd symmetry
  for (int j = 0; j < num_elem_x2; j++)
  {
    for (int k = 0; k < num_elem_x1; k++)
    {
      for(int l=0; l<num_solpts * num_solpts; l++)
      {
        // Set the bottom boundary
        const int index_b_bottom = get_c_index(0, 0, j, k, l + num_solpts * num_solpts, array_shape_x3);
        euler_state_3d<num_t> params_b_bottom(q_itf_x3_ptr, index_b_bottom, stride);

        const int index_in_bottom = get_c_index(0, 1, j, k, l, array_shape_x3);
        euler_state_3d<const num_t> params_in_bottom(q_itf_x3_ptr, index_in_bottom, stride);

        boundary_euler_cubedsphere_3d_kernel<real_t, num_t>(params_in_bottom, params_b_bottom);

        // Set the top boundary
        const int index_b_top = get_c_index(0, num_elem_x3 + 1, j, k, l, array_shape_x3);
        euler_state_3d<num_t> params_b_top(q_itf_x3_ptr, index_b_top, stride);

        const int index_in_top = get_c_index(0, num_elem_x3, j, k, l + num_solpts * num_solpts, array_shape_x3);
        euler_state_3d<const num_t> params_in_top(q_itf_x3_ptr, index_in_top, stride);

        boundary_euler_cubedsphere_3d_kernel<real_t, num_t>(params_in_top, params_b_top);
      }
    }
  }


  // Compute the fluxes along the x3-direction
  for (int i = 0; i < num_elem_x3 + 1; i++)
  {
    for (int j = 0; j < num_elem_x2; j++)
    {
      for (int k = 0; k < num_elem_x1; k++)
      {
        for(int l=0 ; l < num_solpts * num_solpts ; l++)
        {
          const int index_l = get_c_index(0, i, j, k, l + num_solpts * num_solpts, array_shape_x3);
          kernel_params_cubedsphere<real_t, num_t> params_l(
            q_itf_x3_ptr,
            sqrt_g_ptr,
            h_ptr,
            index_l,
            stride,
            nullptr,
            nullptr,
            flux_itf_x3_ptr,
            pressure_itf_x3_ptr,
            wflux_adv_itf_x1_ptr,
            wflux_adv_itf_x2_ptr,
            wflux_adv_itf_x3_ptr,
            wflux_pres_itf_x1_ptr,
            wflux_pres_itf_x2_ptr,
            wflux_pres_itf_x3_ptr,
            nullptr);

          const int index_r = get_c_index(0, i + 1, j, k, l, array_shape_x3);
          kernel_params_cubedsphere<real_t, num_t> params_r(
            q_itf_x3_ptr,
            sqrt_g_ptr,
            h_ptr,
            index_r,
            stride,
            nullptr,
            nullptr,
            flux_itf_x3_ptr,
            pressure_itf_x3_ptr,
            wflux_adv_itf_x1_ptr,
            wflux_adv_itf_x2_ptr,
            wflux_adv_itf_x3_ptr,
            wflux_pres_itf_x1_ptr,
            wflux_pres_itf_x2_ptr,
            wflux_pres_itf_x3_ptr,
            nullptr);

          riemann_euler_cubedsphere_rusanov_3d_kernel<real_t, num_t>(params_l, params_r, 2);
        }
      }
    }
  }
}


template <typename real_t, typename num_t>
void forcing_euler_cubesphere_3d(
    const py::array_t<num_t>&  q_in,
    const py::array_t<num_t>&  pressure_in,
    const py::array_t<real_t>& sqrt_g_in,
    const py::array_t<real_t>& h_in,
    const py::array_t<real_t>& christoffel_in,
    py::array_t<num_t>&        forcing_in,
    const int                  num_elem_x1,
    const int                  num_elem_x2,
    const int                  num_elem_x3,
    const int                  num_solpts,
    const int                  verbose) {

  const num_t*  q           = get_c_ptr(q_in);
  const num_t*  pressure    = get_c_ptr(pressure_in);
  const real_t* sqrt_g      = get_c_ptr(sqrt_g_in);
  const real_t* h           = get_c_ptr(h_in);
  const real_t* christoffel = get_c_ptr(christoffel_in);
  num_t*        forcing     = get_c_ptr(forcing_in);

  const uint64_t stride = num_elem_x3 * num_elem_x2 * num_elem_x1 * num_solpts;

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
  m.def("riemann_euler_cubedsphere_rusanov_3d", &riemann_euler_cubedsphere_rusanov_3d<double, double>);
  m.def("riemann_euler_cubedsphere_rusanov_3d", &riemann_euler_cubedsphere_rusanov_3d<double, complex_t>);

  // Forcing functions
  m.def("forcing_euler_cubesphere_3d", &forcing_euler_cubesphere_3d<double, double>);
  m.def("forcing_euler_cubesphere_3d", &forcing_euler_cubesphere_3d<double, complex_t>);
}
