#include "../definitions.hpp"

template<typename num_t>
DEVICE_SPACE void pointwise_eulercartesian_2d_kernel(const num_t *q, num_t *flux_x1, num_t *flux_x2, const int stride)
{
  // Compute the strides to access state variables
  const int idx_rho = 0;
  const int idx_rhou = stride;
  const int idx_rhow = 2 * stride;
  const int idx_rhot = 3 * stride;

  const num_t rho = q[idx_rho];
  const num_t rhou = q[idx_rhou];
  const num_t rhow = q[idx_rhow];
  const num_t rho_theta = q[idx_rhot];

  const num_t invrho = 1.0 / rho;
  const num_t u = rhou * invrho;
  const num_t w = rhow * invrho;

  const num_t p = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_theta));

  flux_x1[idx_rho] = rhou;
  flux_x1[idx_rhou] = rhou * u + p;
  flux_x1[idx_rhow] = rhou * w;
  flux_x1[idx_rhot] = rho_theta * u;

  flux_x2[idx_rho] = rhow;
  flux_x2[idx_rhou] = rhow * u;
  flux_x2[idx_rhow] = rhow * w + p;
  flux_x2[idx_rhot] = rho_theta * w;
}