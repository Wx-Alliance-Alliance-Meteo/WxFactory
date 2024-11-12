#include "../definitions.hpp"

template<typename num_t>
DEVICE_SPACE void boundary_eulercartesian_2d_kernel(kernel_params<num_t, euler_state_2d> params, const int dir)
{
  // Set the the boundary fluxes pressure
  const int rho_theta = *params.q.rho_theta;
  if (dir == 0)
  {
    *params.flux[0].rho = 0.0;
    *params.flux[0].rho_u = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    *params.flux[0].rho_w = 0.0;
    *params.flux[0].rho_theta = 0.0;
  }
  else if (dir == 1)
  {
    *params.flux[1].rho = 0.0;
    *params.flux[1].rho_u = 0.0;
    *params.flux[1].rho_w = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    *params.flux[1].rho_theta = 0.0;
  }
}