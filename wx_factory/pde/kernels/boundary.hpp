#include "definitions/definitions.hpp"

template <typename num_t>
DEVICE_SPACE void boundary_eulercartesian_2d_kernel(
    kernel_params<num_t, euler_state_2d> params,
    const int                            dir) {


    const num_t rho   = *params.q.rho;
    const num_t rho_u = *params.q.rho_u;
    const num_t rho_w = *params.q.rho_w;
    const num_t rho_E = *params.q.rho_theta; // stored in rho_theta slot

    const num_t inv_rho = 1.0 / rho;
    const num_t u       = rho_u * inv_rho;
    const num_t w       = rho_w * inv_rho;

    const num_t kinetic = 0.5 * (rho_u * u + rho_w * w);
    const num_t p = (heat_capacity_ratio - 1.0) * (rho_E - kinetic);

    if (dir == 0)
    {
      // Wall normal in x direction
      *params.flux[0].rho       = 0.0;
      *params.flux[0].rho_u     = p;
      *params.flux[0].rho_w     = 0.0;
      *params.flux[0].rho_theta = 0.0;
    }
    else if (dir == 1)
    {
      // Wall normal in z/w direction
      *params.flux[1].rho       = 0.0;
      *params.flux[1].rho_u     = 0.0;
      *params.flux[1].rho_w     = p;
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
