#include <cmath>
#include "definitions.h"

template<typename num_t> 
void boundary_eulercartesian_2d(const num_t *q, num_t *flux, const int direction, const int stride)
{
  // Compute the strides to access state variables
  const int idx_rho = 0;
  const int idx_rhou = stride;
  const int idx_rhow = 2 * stride;
  const int idx_rhot = 3 * stride;

  if (direction == 0)
  {
    const num_t rho_theta = q[idx_rhot];
    flux[idx_rho] = 0.0;
    flux[idx_rhou] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    flux[idx_rhow] = 0.0;
    flux[idx_rhot] = 0.0;
  }
  else if (direction == 1)
  {
    const num_t rho_theta = q[idx_rhot];
    flux[idx_rho] = 0.0;
    flux[idx_rhou] = 0.0;
    flux[idx_rhow] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    flux[idx_rhot] = 0.0;
  }
}


template void boundary_eulercartesian_2d<double>(const double *q, double *flux, const int direction, const int stride);
template void boundary_eulercartesian_2d<float>(const float *q, float *flux, const int direction, const int stride);
// template void boundary_eulercartesian<complex>(complex *q, complex *flux, const int direction, const int stride);