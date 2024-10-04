#include <cmath>
#include <iostream>

extern "C" void pointwise_flux_sw(double *q, double *flux_x, double *flux_y, double *flux_z, const double sqrt_g, const double *metrics, const int stride, const int num_dim)
{
  // These declarations are temporary and must be done globally
  const double gravity = 9.80616;

  // Compute the strides to access state variables
  const int idx_h = 0;
  const int idx_hu1 = stride;
  const int idx_hu2 = 2*stride;

  // Metrics contravariant indices 
  if(num_dim==2)
  {
    const int i11 = 0;
    const int i12 = stride;
    const int i21 = 2*stride;
    const int i22 = 3*stride;

    const double h_sqr = q[idx_h]*q[idx_h];
    const double ghsqr = 0.5 * gravity * h_sqr;

    const double u1 = q[idx_hu1] / q[idx_h];
    const double u2 = q[idx_hu2] / q[idx_h];

    flux_x[idx_h] = sqrt_g * q[idx_hu1];
    flux_x[idx_hu1] = sqrt_g * (q[idx_hu1] * u1 + metrics[i11] * ghsqr);
    flux_x[idx_hu2] = sqrt_g * (q[idx_hu2] * u1 + metrics[i21] * ghsqr);
    
    flux_z[idx_h] = sqrt_g * q[idx_hu2];
    flux_z[idx_hu1]  = sqrt_g * (q[idx_hu1] * u2 + metrics[i12] * ghsqr);
    flux_z[idx_hu2]  = sqrt_g * (q[idx_hu2] * u2 + metrics[i22] * ghsqr);
  }
}