#include <cmath>
#include "definitions.h"

template<typename T> 
void boundary_eulercartesian_2d(T *Q, T *flux, const int direction, const int stride)
{
  T rho_theta;

  // Compute the strides to access state variables
  const int idx_rho = 0;
  const int idx_rhou = stride;
  const int idx_rhow = 2 * stride;
  const int idx_rhot = 3 * stride;

  if (direction == 0)
  {
    rho_theta = Q[idx_rhot];
    flux[idx_rho] = 0.0;
    flux[idx_rhou] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    flux[idx_rhow] = 0.0;
    flux[idx_rhot] = 0.0;
  }

  if (direction == 1)
  {
    rho_theta = Q[idx_rhot];
    flux[idx_rho] = 0.0;
    flux[idx_rhou] = 0.0;
    flux[idx_rhow] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    flux[idx_rhot] = 0.0;
  }
}


template void boundary_eulercartesian_2d<double>(double *Q, double *flux, const int direction, const int stride);
template void boundary_eulercartesian_2d<float>(float *Q, float *flux, const int direction, const int stride);
// template void boundary_eulercartesian<complex>(complex *Q, complex *flux, const int direction, const int stride);