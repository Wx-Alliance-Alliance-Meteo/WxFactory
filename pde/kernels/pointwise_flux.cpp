#include <cmath>
#include "definitions.h"

template<typename num_t>
void pointwise_eulercartesian_2d(const num_t *q, num_t *flux_x1, num_t *flux_x2, const int stride)
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
template void pointwise_eulercartesian_2d<double>(const double *q, double *flux_x1, double *flux_x2, const int stride);
template void pointwise_eulercartesian_2d<float>(const float *q, float *flux_x1, float *flux_x2, const int stride);
// template void pointwise_eulercartesian_2<complex>(const complex *q, complex *flux_x1, complex *flux_x2, const int stride)


template<typename num_t>
void pointwise_swcubedsphere_2d(const num_t *q, num_t *flux_x1, num_t *flux_x2, const double sqrt_g, const double *metrics, const int stride)
{
  // num_these declarations are temporary and must be done globally
  const double gravity = 9.80616;

  // Compute the strides to access state variables
  const int idx_h = 0;
  const int idx_hu1 = stride;
  const int idx_hu2 = 2*stride;

  const int i11 = 0;
  const int i12 = stride;
  const int i21 = 2*stride;
  const int i22 = 3*stride;

  const num_t h_sqr = q[idx_h]*q[idx_h];
  const num_t ghsqr = 0.5 * gravity * h_sqr;

  const num_t u1 = q[idx_hu1] / q[idx_h];
  const num_t u2 = q[idx_hu2] / q[idx_h];

  flux_x1[idx_h] = sqrt_g * q[idx_hu1];
  flux_x1[idx_hu1] = sqrt_g * (q[idx_hu1] * u1 + metrics[i11] * ghsqr);
  flux_x1[idx_hu2] = sqrt_g * (q[idx_hu2] * u1 + metrics[i21] * ghsqr);
  
  flux_x2[idx_h] = sqrt_g * q[idx_hu2];
  flux_x2[idx_hu1]  = sqrt_g * (q[idx_hu1] * u2 + metrics[i12] * ghsqr);
  flux_x2[idx_hu2]  = sqrt_g * (q[idx_hu2] * u2 + metrics[i22] * ghsqr);  
}

template void pointwise_swcubedsphere_2d<double>(const double *q, double *flux_x1, double *flux_x2, const double sqrt_g, const double *metrics, const int stride);template void pointwise_swcubedsphere_2d<float>(const float *q, float *flux_x1, float *flux_x2, const double sqrt_g, const double *metrics, const int stride);
// // template<complex> void pointwise_swcubedsphere_2d(const complex *q, complex *flux_x1, complex *flux_x2, const double sqrt_g, const double *metrics, const int stride);