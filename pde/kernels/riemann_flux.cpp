#include <cmath>
#include "definitions.h"

template<typename T>
void riemann_eulercartesian_ausm_2d(T *ql, T *qr, T *fl, T *fr,
                            const int nvars, const int direction,
                            const int stride)
{
  T rhol, rhor, rho_thetal, rho_thetar;
  T rhoul, rhour, rhowl, rhowr;
  T ul, ur, wl, wr, invrhol, invrhor;
  T pl, pr, al, ar, vnl, vnr, M, Ml, Mr, Mmax, Mmin;

  // Compute the strides to access state variables
  const int idx_rho = 0;
  const int idx_rhou = stride;
  const int idx_rhow = 2 * stride;
  const int idx_rhot = 3 * stride;

  // Get the left and right variables
  rhol = ql[idx_rho];
  rhoul = ql[idx_rhou];
  rhowl = ql[idx_rhow];
  rho_thetal = ql[idx_rhot];

  rhor = qr[idx_rho];
  rhour = qr[idx_rhou];
  rhowr = qr[idx_rhow];
  rho_thetar = qr[idx_rhot];

  invrhol = 1.0 / rhol;
  ul = rhoul * invrhol;
  wl = rhowl * invrhol;

  invrhor = 1.0 / rhor;
  ur = rhour * invrhor;
  wr = rhowr * invrhor;

  // Compute the pressure and speed of sound
  pl = p0 * pow(rho_thetal * Rd * inp0, heat_capacity_ratio);
  pr = p0 * pow(rho_thetar * Rd * inp0, heat_capacity_ratio);

  al = sqrt(heat_capacity_ratio * pl * invrhol);
  ar = sqrt(heat_capacity_ratio * pr * invrhor);

  // Vertical faces
  if (direction == 0)
  {
    vnl = ul;
    vnr = ur;
  }
  if (direction == 1)
  {
    vnl = wl;
    vnr = wr;
  }

  // Compute the modified mach number
  Ml = vnl / al + 1.0;
  Mr = vnr / ar - 1.0;

  M = 0.25 * (Ml * Ml - Mr * Mr);
  Mmax = std::fmax(0.0, M) * al;
  Mmin = std::fmin(0.0, M) * ar;

  // Set the interface fluxes
  fl[idx_rho] = rhol * Mmax + rhor * Mmin;
  if (direction == 0)
  {
    fl[idx_rhou] = 0.5 * (Ml * pl - Mr * pr);
    fl[idx_rhow] = rhowl * Mmax + rhowr * Mmin;
  }
  if (direction == 1)
  {
    fl[idx_rhou] = rhoul * Mmax + rhour * Mmin;
    fl[idx_rhow] = 0.5 * (Ml * pl - Mr * pr);
  }
  fl[idx_rhot] = rho_thetal * Mmax + rho_thetar * Mmin;

  // Set the right fluxes
  for (int i = 0; i < nvars; i++)
    fr[i * stride] = fl[i * stride];
}

template void riemann_eulercartesian_ausm_2d<double> (double *ql, double *qr, double *fl, double *fr, const int nvars, const int direction, const int stride);
template void riemann_eulercartesian_ausm_2d<float>(float *ql, float *qr, float *fl, float *fr, const int nvars, const int direction, const int stride);
// template void riemann_eulercartesian_ausm<complex>(complex *ql, complex *qr, complex *fl, complex *fr, const int nvars, const int direction, const int stride);

