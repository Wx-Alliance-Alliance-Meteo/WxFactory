#include <cmath>
#include <iostream>

extern "C" void pointwise_flux_euler(double *q, double *flux_x, double *flux_y, double *flux_z, const double sqrt_g, const double *metrics, const int stride, const int num_dim)
{
  // These declarations are temporary and must be done globally
  double p0 = 100000.;
  double Rd = 287.05;
  double cpd = 1005.46;
  double cvd = (cpd - Rd);
  double kappa = Rd / cpd;
  double heat_capacity_ratio = cpd / cvd;
  double inp0 = 1.0 / p0;
  double Rdinp0 = Rd * inp0;

  double rho, invrho, rhou, rhow, rho_theta;
  double u, w, p;

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

  flux_x[idx_rho] = rhou;
  flux_x[idx_rhou] = rhou * u + p;
  flux_x[idx_rhow] = rhou * w;
  flux_x[idx_rhot] = rho_theta * u;

  flux_z[idx_rho] = rhow;
  flux_z[idx_rhou] = rhow * u;
  flux_z[idx_rhow] = rhow * w + p;
  flux_z[idx_rhot] = rho_theta * w;
}

extern "C" void riemann_ausm_euler(double *ql, double *qr, double *fl, double *fr,
                                   const int nvars, const int direction,
                                   const int stride)
{
  // These declarations are temporary and must be done globally
  double p0 = 100000.;
  double Rd = 287.05;
  double cpd = 1005.46;
  double cvd = (cpd - Rd);
  double kappa = Rd / cpd;
  double heat_capacity_ratio = cpd / cvd;
  double inp0 = 1.0 / p0;
  double Rdinp0 = Rd * inp0;

  double rhol, rhor, rho_thetal, rho_thetar;
  double rhoul, rhour, rhowl, rhowr;
  double ul, ur, wl, wr, invrhol, invrhor;
  double pl, pr, al, ar, vnl, vnr, M, Ml, Mr, Mmax, Mmin;

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
  fl[idx_rhou] = rhoul * Mmax + rhour * Mmin;
  fl[idx_rhow] = rhowl * Mmax + rhowr * Mmin;
  fl[idx_rhot] = rho_thetal * Mmax + rho_thetar * Mmin;

  if (direction == 0)
  {
    fl[idx_rhou] += 0.5 * (Ml * pl - Mr * pr);
  }
  if (direction == 1)
  {
    fl[idx_rhow] += 0.5 * (Ml * pl - Mr * pr);
  }

  // Set the right fluxes
  for (int i = 0; i < nvars; i++)
    fr[i * stride] = fl[i * stride];
}

extern "C" void boundary_flux_euler(double *q, double *flux, const int direction, const int stride)
{
  // These declarations are temporary and must be done globally
  double p0 = 100000.;
  double Rd = 287.05;
  double cpd = 1005.46;
  double cvd = (cpd - Rd);
  double kappa = Rd / cpd;
  double heat_capacity_ratio = cpd / cvd;
  double inp0 = 1.0 / p0;
  double Rdinp0 = Rd * inp0;

  double rho_theta;

  // Compute the strides to access state variables
  const int idx_rho = 0;
  const int idx_rhou = stride;
  const int idx_rhow = 2 * stride;
  const int idx_rhot = 3 * stride;

  if (direction == 0)
  {
    rho_theta = q[idx_rhot];
    flux[idx_rho] = 0.0;
    flux[idx_rhou] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    flux[idx_rhow] = 0.0;
    flux[idx_rhot] = 0.0;
  }

  if (direction == 1)
  {
    rho_theta = q[idx_rhot];
    flux[idx_rho] = 0.0;
    flux[idx_rhou] = 0.0;
    flux[idx_rhow] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    flux[idx_rhot] = 0.0;
  }
}