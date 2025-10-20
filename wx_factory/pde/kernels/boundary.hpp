#include "common/parameters.hpp"
#include "common/physical_constants.hpp"

template <typename num_t>
DEVICE_SPACE void boundary_eulercartesian_2d_kernel(
    kernel_params<num_t, euler_state_2d> params,
    const int                            dir) {
  // Set the the boundary fluxes pressure
  const num_t rho_theta = *params.q.rho_theta;
  if (dir == 0)
  {
    *params.flux[0].rho       = 0.0;
    *params.flux[0].rho_u     = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    *params.flux[0].rho_w     = 0.0;
    *params.flux[0].rho_theta = 0.0;
  }
  else if (dir == 1)
  {
    *params.flux[1].rho       = 0.0;
    *params.flux[1].rho_u     = 0.0;
    *params.flux[1].rho_w     = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    *params.flux[1].rho_theta = 0.0;
  }
}

template <typename real_t, typename num_t>
DEVICE_SPACE void boundary_euler_cubedsphere_3d_kernel(
    euler_state_3d<const num_t> state_in,
    euler_state_3d<num_t>       state_b) {
  const num_t w_b = -1.0 * *state_in.rho_w / *state_in.rho;

  // Set symmetry/slip wall boundary
  // Extrapolate variables and mirror w-velocity
  *state_b.rho       = *state_in.rho;
  *state_b.rho_u     = *state_in.rho_u;
  *state_b.rho_v     = *state_in.rho_v;
  *state_b.rho_w     = *state_in.rho * w_b;
  *state_b.rho_theta = *state_in.rho_theta;
}
