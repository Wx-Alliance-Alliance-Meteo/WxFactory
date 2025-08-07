#include "common/parameters.hpp"
#include "common/physical_constants.hpp"

template <typename num_t>
DEVICE_SPACE void
pointwise_eulercartesian_2d_kernel(kernel_params<num_t, euler_state_2d> params) {
  // Extract variables from state pointer
  const num_t rho       = *params.q.rho;
  const num_t rho_u     = *params.q.rho_u;
  const num_t rho_w     = *params.q.rho_w;
  const num_t rho_theta = *params.q.rho_theta;

  // Extract velocity components and compute pressure
  const num_t inv_rho = 1.0 / rho;
  const num_t u       = rho_u * inv_rho;
  const num_t w       = rho_w * inv_rho;

  // Get the pressure
  const num_t p = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_theta));

  // Set the values of the fluxes
  *params.flux[0].rho       = rho_u;
  *params.flux[0].rho_u     = rho_u * u + p;
  *params.flux[0].rho_w     = rho_u * w;
  *params.flux[0].rho_theta = rho_theta * u;

  *params.flux[1].rho       = rho_w;
  *params.flux[1].rho_u     = rho_w * u;
  *params.flux[1].rho_w     = rho_w * w + p;
  *params.flux[1].rho_theta = rho_theta * w;
}

template <typename real_t, typename num_t>
DEVICE_SPACE void pointwise_euler_cubedsphere_3d_kernel(
    kernel_params_cubedsphere<real_t, num_t> params,
    bool                                     verbose) {

  (void)verbose; // disable compiler warning. Only used for debugging

  // Extract metric
  const real_t sqrt_g = *params.sqrt_g;

  // Extract variables from state pointer
  const num_t rho       = *params.q.rho;
  const num_t rho_u     = *params.q.rho_u;
  const num_t rho_v     = *params.q.rho_v;
  const num_t rho_w     = *params.q.rho_w;
  const num_t rho_theta = *params.q.rho_theta;

  // Extract velocity components and compute pressure
  const num_t inv_rho = 1.0 / rho;
  const num_t u       = rho_u * inv_rho;
  const num_t v       = rho_v * inv_rho;
  const num_t w       = rho_w * inv_rho;

  // Get the pressure
  const num_t p = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_theta));

  // Set the fluxes
  *params.flux[0].rho       = sqrt_g * rho_u;
  *params.flux[0].rho_u     = sqrt_g * (rho_u * u + params.h[h11] * p);
  *params.flux[0].rho_v     = sqrt_g * (rho_v * u + params.h[h12] * p);
  *params.flux[0].rho_w     = sqrt_g * (rho_w * u + params.h[h13] * p);
  *params.flux[0].rho_theta = sqrt_g * rho_theta * u;

  *params.flux[1].rho       = sqrt_g * rho_v;
  *params.flux[1].rho_u     = sqrt_g * (rho_u * v + params.h[h21] * p);
  *params.flux[1].rho_v     = sqrt_g * (rho_v * v + params.h[h22] * p);
  *params.flux[1].rho_w     = sqrt_g * (rho_w * v + params.h[h23] * p);
  *params.flux[1].rho_theta = sqrt_g * rho_theta * v;

  *params.flux[2].rho       = sqrt_g * rho_w;
  *params.flux[2].rho_u     = sqrt_g * (rho_u * w + params.h[h31] * p);
  *params.flux[2].rho_v     = sqrt_g * (rho_v * w + params.h[h32] * p);
  *params.flux[2].rho_w     = sqrt_g * (rho_w * w + params.h[h33] * p);
  *params.flux[2].rho_theta = sqrt_g * rho_theta * w;

  // Set the fluxes
  *params.wflux_adv[0] = sqrt_g * rho_w * u;
  *params.wflux_adv[1] = sqrt_g * rho_w * v;
  *params.wflux_adv[2] = sqrt_g * rho_w * w;

  *params.wflux_pres[0] = sqrt_g * params.h[h13];
  *params.wflux_pres[1] = sqrt_g * params.h[h23];
  *params.wflux_pres[2] = sqrt_g * params.h[h33];

  *params.pressure = p;
  *params.logp     = log(p);
}
