#include "../definitions.hpp"

template<typename num_t>
DEVICE_SPACE void pointwise_eulercartesian_2d_kernel(kernel_params<num_t, euler_state_2d> params)
{ 
  // Extract variables from state pointer
  const num_t rho = *params.q.rho;
  const num_t rho_u = *params.q.rho_u;
  const num_t rho_w = *params.q.rho_w;
  const num_t rho_theta = *params.q.rho_theta;

  // Extract velocity components and compute pressure
  const num_t inv_rho = 1.0 / rho;
  const num_t u = rho_u * inv_rho;
  const num_t w = rho_w * inv_rho;

  // Get the pressure
  const num_t p = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_theta));

  // Set the values of the fluxes
  *params.flux[0].rho = rho_u;
  *params.flux[0].rho_u = rho_u * u + p;
  *params.flux[0].rho_w = rho_u * w;
  *params.flux[0].rho_theta = rho_theta * u;

  *params.flux[1].rho = rho_w;
  *params.flux[1].rho_u = rho_w * u;
  *params.flux[1].rho_w = rho_w * w + p;
  *params.flux[1].rho_theta = rho_theta * w;
}

