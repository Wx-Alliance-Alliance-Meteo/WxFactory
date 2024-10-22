#include <cmath>
#include "definitions.h"

template<typename T>
void pointwise_eulercartesian_2d(T *q, T *flux_x1, T *flux_x2, const int stride)
{
  T rho, invrho, rhou, rhow, rho_theta;
  T u, w, p;

  // Compute the strides to access state variables
  const int idx_rho = 0;
  const int idx_rhou = stride;
  const int idx_rhow = 2 * stride;
  const int idx_rhot = 3 * stride;

  rho = q[idx_rho];
  rhou = q[idx_rhou];
  rhow = q[idx_rhow];
  rho_theta = q[idx_rhot];

  invrho = 1.0 / rho;
  u = rhou * invrho;
  w = rhow * invrho;

  p = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_theta));

  flux_x1[idx_rho] = rhou;
  flux_x1[idx_rhou] = rhou * u + p;
  flux_x1[idx_rhow] = rhou * w;
  flux_x1[idx_rhot] = rho_theta * u;

  flux_x2[idx_rho] = rhow;
  flux_x2[idx_rhou] = rhow * u;
  flux_x2[idx_rhow] = rhow * w + p;
  flux_x2[idx_rhot] = rho_theta * w;
}
template void pointwise_eulercartesian_2d<double>(double *q, double *flux_x1, double *flux_x2, const int stride);
template void pointwise_eulercartesian_2d<float>(float *q, float *flux_x1, float *flux_x2, const int stride);
// template void pointwise_eulercartesian_2<complex>(T *q, T *flux_x1, T *flux_x2, const int stride)


template<typename T>
void pointwise_swcubedsphere_2d(T *q, T *flux_x1, T *flux_x2, const double sqrt_g, const double *metrics, const int stride)
{
  // These declarations are temporary and must be done globally
  const double gravity = 9.80616;

  // Compute the strides to access state variables
  const int idx_h = 0;
  const int idx_hu1 = stride;
  const int idx_hu2 = 2*stride;

  const int i11 = 0;
  const int i12 = stride;
  const int i21 = 2*stride;
  const int i22 = 3*stride;

  const T h_sqr = q[idx_h]*q[idx_h];
  const T ghsqr = 0.5 * gravity * h_sqr;

  const T u1 = q[idx_hu1] / q[idx_h];
  const T u2 = q[idx_hu2] / q[idx_h];

  flux_x1[idx_h] = sqrt_g * q[idx_hu1];
  flux_x1[idx_hu1] = sqrt_g * (q[idx_hu1] * u1 + metrics[i11] * ghsqr);
  flux_x1[idx_hu2] = sqrt_g * (q[idx_hu2] * u1 + metrics[i21] * ghsqr);
  
  flux_x2[idx_h] = sqrt_g * q[idx_hu2];
  flux_x2[idx_hu1]  = sqrt_g * (q[idx_hu1] * u2 + metrics[i12] * ghsqr);
  flux_x2[idx_hu2]  = sqrt_g * (q[idx_hu2] * u2 + metrics[i22] * ghsqr);  
}

template void pointwise_swcubedsphere_2d<double>(double *q, double *flux_x1, double *flux_x2, const double sqrt_g, const double *metrics, const int stride);template void pointwise_swcubedsphere_2d<float>(float *q, float *flux_x1, float *flux_x2, const double sqrt_g, const double *metrics, const int stride);
// // template<complex> void pointwise_swcubedsphere_2d(complex *q, complex *flux_x1, complex *flux_x2, const double sqrt_g, const double *metrics, const int stride);