#include "../definitions.hpp"

template<typename num_t>
DEVICE_SPACE void riemann_eulercartesian_ausm_2d_kernel(const num_t *ql, const num_t *qr, num_t *fl, num_t *fr, const int direction, const int stride)
{
  // Compute variable stride
  const int idx_rho = 0;
  const int idx_rhou = stride;
  const int idx_rhow = 2 * stride;
  const int idx_rhot = 3 * stride;

  const num_t rhol = ql[idx_rho];
  const num_t rhoul = ql[idx_rhou];
  const num_t rhowl = ql[idx_rhow];
  const num_t rho_thetal = ql[idx_rhot];

  const num_t rhor = qr[idx_rho];
  const num_t rhour = qr[idx_rhou];
  const num_t rhowr = qr[idx_rhow];
  const num_t rho_thetar = qr[idx_rhot];

  const num_t invrhol = 1.0 / rhol;
  const num_t ul = rhoul * invrhol;
  const num_t wl = rhowl * invrhol;

  const num_t invrhor = 1.0 / rhor;
  const num_t ur = rhour * invrhor;
  const num_t wr = rhowr * invrhor;

  const num_t pl = p0 * pow(rho_thetal * Rd * inp0, heat_capacity_ratio);
  const num_t pr = p0 * pow(rho_thetar * Rd * inp0, heat_capacity_ratio);

  const num_t al = sqrt(heat_capacity_ratio * pl * invrhol);
  const num_t ar = sqrt(heat_capacity_ratio * pr * invrhor);

  num_t vnr = 0.0;
  num_t vnl = 0.0;

  // Fx faces
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

  const num_t Ml = vnl / al + 1.0;
  const num_t Mr = vnr / ar - 1.0;

  const num_t M = 0.25 * (Ml * Ml - Mr * Mr);
  const num_t Mmax = max(0.0, M) * al;
  const num_t Mmin = min(0.0, M) * ar;

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

  for (int i = 0; i < 4; i++)
  {
    fr[i * stride] = fl[i * stride];
  }

}
