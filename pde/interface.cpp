#include "definitions.hpp"
#include "interface.hpp"

#include "kernels/kernels.h"

namespace py = pybind11;

// -------------------------------------
// Pointwise fluxes
// -------------------------------------

template <typename num_t>
void pointwise_eulercartesian_2d(const py::array_t<num_t> &q_in, py::array_t<num_t> &flux_x1_in, py::array_t<num_t> &flux_x2_in, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts_tot)
{
  py::buffer_info buf1 = q_in.request();
  py::buffer_info buf2 = flux_x1_in.request();
  py::buffer_info buf3 = flux_x2_in.request();

  // Get the pointers
  num_t* q = static_cast<num_t*>(buf1.ptr);
  num_t* flux_x1 = static_cast<num_t*>(buf2.ptr);
  num_t* flux_x2 = static_cast<num_t*>(buf3.ptr);

  const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts_tot;
  const int array_shape[4] = {4, nb_elem_x2, nb_elem_x1, nb_solpts_tot};

  for(int i=0; i<nb_elem_x2; i++)
  {
    for(int j=0; j<nb_elem_x1; j++)
    {
      for(int k=0; k<nb_solpts_tot; k++)
      {
        const int ind = get_c_index(0, i, j, k, array_shape);
        
        // Store variables and pointers to compute the fluxes 
        kernel_params<num_t,euler_state_2d> params(q, flux_x1, flux_x2, nullptr, ind, stride);

        // Call the pointwise flux kernel
        pointwise_eulercartesian_2d_kernel(params);
      }
    }
  }
}

// -------------------------------------
// Riemann fluxes
// -------------------------------------

template <typename num_t>
void riemann_eulercartesian_ausm_2d(const py::array_t<num_t> &q_itf_x1_in, const py::array_t<num_t> &q_itf_x2_in, py::array_t<num_t> &flux_itf_x1_in, py::array_t<num_t> &flux_itf_x2_in, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts)
{
  py::buffer_info buf1 = q_itf_x1_in.request();
  py::buffer_info buf2 = q_itf_x2_in.request();
  py::buffer_info buf3 = flux_itf_x1_in.request();
  py::buffer_info buf4 = flux_itf_x2_in.request();

  // Get the pointers
  num_t* q_itf_x1 = static_cast<num_t*>(buf1.ptr);
  num_t* q_itf_x2 = static_cast<num_t*>(buf2.ptr);
  num_t* flux_itf_x1 = static_cast<num_t*>(buf3.ptr);
  num_t* flux_itf_x2 = static_cast<num_t*>(buf4.ptr);

  const int nb_solpts_riem = 2 * nb_solpts;
  const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts_riem;
  const int array_shape[4] = {4, nb_elem_x2, nb_elem_x1, nb_solpts_riem};

  for (int i=0; i<nb_elem_x2; i++)
  {
    for (int j=0; j<nb_elem_x1; j++)
    {
      // Solve along the horizontal  direction
      if (j + 1 < nb_elem_x1)
      {
        for(int k=0; k<nb_solpts; k++)
        {
          // Initialize left-hand-side parameters
          const int indl = get_c_index(0, i, j, nb_solpts + k, array_shape);
          kernel_params<num_t,euler_state_2d> params_l(q_itf_x1, flux_itf_x1, nullptr, nullptr, indl, stride);
          
          // Initialize right-hand-size parameters
          const int indr = get_c_index(0, i, j + 1, k, array_shape);
          kernel_params<num_t,euler_state_2d> params_r(q_itf_x1, flux_itf_x1, nullptr, nullptr, indr, stride);
  
          // Call Riemann kernel on the horizontal direction
          riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, 0);
        }
      }

      // Solve the Riemann problem along the vertical direc tion
      if (i + 1 < nb_elem_x2)
      {
        for(int k=0; k<nb_solpts; k++)
        {
          // Initialize left-hand-side parameters
          const int indl = get_c_index(0, i, j, nb_solpts + k, array_shape);
          kernel_params<num_t,euler_state_2d> params_l(q_itf_x2, nullptr, flux_itf_x2, nullptr, indl, stride);

          // Initialize right-hand-size parameters
          const int indr = get_c_index(0, i + 1, j, k, array_shape);
          kernel_params<num_t,euler_state_2d> params_r(q_itf_x2, nullptr, flux_itf_x2, nullptr, indr, stride);

          // Call Riemann kernel on the vertical direction
          riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, 1);
        }
      }
    }
  }

  // Update boundary conditions

  // Set the boundary fluxes along the horizontal direction
  for(int i=0; i<nb_elem_x2; i++)
  {
    for(int j=0; j<nb_solpts; j++)
    { 
      // Set the fluxes on the left boundary
      const int indl = get_c_index(0, i, 0, j, array_shape);
      kernel_params<num_t,euler_state_2d> params_l(q_itf_x1, flux_itf_x1, nullptr, nullptr, indl, stride);
      boundary_eulercartesian_2d_kernel(params_l, 0);

      // Set the fluxes on the right boundary
      const int indr = get_c_index(0, i, nb_elem_x1-1, j + nb_solpts, array_shape);
      kernel_params<num_t,euler_state_2d> params_r(q_itf_x1, flux_itf_x1, nullptr, nullptr, indr, stride);
      boundary_eulercartesian_2d_kernel(params_r, 0);
    }
  }
  
  // Set the boundary fluxes along the vertical direction
  for(int i=0; i<nb_elem_x1; i++)
  {
    for(int j=0; j<nb_solpts; j++)
    { 
      // Set the fluxes on the bottom boundary
      const int indb = get_c_index(0, 0, i, j, array_shape);
      kernel_params<num_t,euler_state_2d> params_b(q_itf_x2, nullptr, flux_itf_x2, nullptr, indb, stride);
      boundary_eulercartesian_2d_kernel(params_b, 1);

      // Set the fluxes on the top boundary
      const int indt = get_c_index(0, nb_elem_x2-1, i, j + nb_solpts, array_shape);
      kernel_params<num_t,euler_state_2d> params_t(q_itf_x2, nullptr, flux_itf_x2, nullptr, indt, stride);
      boundary_eulercartesian_2d_kernel(params_t, 1);
    }
  }
}


PYBIND11_MODULE(interface_c, m)
{
  // Pointwise fluxes
  m.def("pointwise_eulercartesian_2d", &pointwise_eulercartesian_2d<double>);
  m.def("pointwise_eulercartesian_2d", &pointwise_eulercartesian_2d<complex_t>);
  
  // Riemann fluxes
  m.def("riemann_eulercartesian_ausm_2d", &riemann_eulercartesian_ausm_2d<double>);
  m.def("riemann_eulercartesian_ausm_2d", &riemann_eulercartesian_ausm_2d<complex_t>);
}
