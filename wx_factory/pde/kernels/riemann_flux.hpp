#include <stdio.h>

#include "common/parameters.hpp"
#include "common/physical_constants.hpp"

template <typename num_t>
DEVICE_SPACE void riemann_eulercartesian_ausm_2d_kernel(
    kernel_params<num_t, euler_state_2d> params_l,
    kernel_params<num_t, euler_state_2d> params_r,
    const int                            dir) {
  // Unpack variables for improved readability
  const num_t rhol = *params_l.q.rho;
  const num_t rhor = *params_r.q.rho;

  const num_t rho_ul = *params_l.q.rho_u;
  const num_t rho_ur = *params_r.q.rho_u;

  const num_t rho_wl = *params_l.q.rho_w;
  const num_t rho_wr = *params_r.q.rho_w;

  const num_t rho_thetal = *params_l.q.rho_theta;
  const num_t rho_thetar = *params_r.q.rho_theta;

  const num_t inv_rhol = 1.0 / rhol;
  const num_t ul       = rho_ul * inv_rhol;
  const num_t wl       = rho_wl * inv_rhol;

  const num_t inv_rhor = 1.0 / rhor;
  const num_t ur       = rho_ur * inv_rhor;
  const num_t wr       = rho_wr * inv_rhor;

  // Compute the left and right-hand side pressure states
  const num_t pl = p0 * pow(rho_thetal * Rd * inp0, heat_capacity_ratio);
  const num_t pr = p0 * pow(rho_thetar * Rd * inp0, heat_capacity_ratio);

  // Get the speed of sound on each side
  const num_t al = sqrt(heat_capacity_ratio * pl * inv_rhol);
  const num_t ar = sqrt(heat_capacity_ratio * pr * inv_rhor);

  num_t vnr = 0.0;
  num_t vnl = 0.0;

  if (dir == 0)
  {
    vnl = ul;
    vnr = ur;
  }

  if (dir == 1)
  {
    vnl = wl;
    vnr = wr;
  }

  const num_t Ml = vnl / al + 1.0;
  const num_t Mr = vnr / ar - 1.0;

  const num_t M    = 0.25 * (Ml * Ml - Mr * Mr);
  const num_t Mmax = fmax(0.0, M) * al;
  const num_t Mmin = fmin(0.0, M) * ar;

  // Set the interface fluxes
  *params_l.flux[dir].rho = rhol * Mmax + rhor * Mmin;
  if (dir == 0)
  {
    *params_l.flux[dir].rho_u = 0.5 * (Ml * pl - Mr * pr);
    *params_l.flux[dir].rho_w = rho_wl * Mmax + rho_wr * Mmin;
  }
  else
  {
    *params_l.flux[dir].rho_u = rho_ul * Mmax + rho_ur * Mmin;
    *params_l.flux[dir].rho_w = 0.5 * (Ml * pl - Mr * pr);
  }
  *params_l.flux[dir].rho_theta = rho_thetal * Mmax + rho_thetar * Mmin;

  // Copy values to the right-hand side flux
  *params_r.flux[dir].rho       = *params_l.flux[dir].rho;
  *params_r.flux[dir].rho_u     = *params_l.flux[dir].rho_u;
  *params_r.flux[dir].rho_w     = *params_l.flux[dir].rho_w;
  *params_r.flux[dir].rho_theta = *params_l.flux[dir].rho_theta;
}

template <typename real_t, typename num_t>
DEVICE_SPACE void riemann_euler_cubedsphere_rusanov_3d_kernel(
    riemann_params_cubedsphere<real_t, num_t> params_l,
    riemann_params_cubedsphere<real_t, num_t> params_r,
    const int                                 dir,
    const bool                                boundary) {

  // Extract necessary metrics
  const real_t sqrt_g_l = *params_l.sqrt_g;
  const real_t sqrt_g_r = *params_r.sqrt_g;

  const real_t hdir_0_l = params_l.h[3 * dir + 0];
  const real_t hdir_1_l = params_l.h[3 * dir + 1];
  const real_t hdir_2_l = params_l.h[3 * dir + 2];

  const real_t hdir_0_r = params_r.h[3 * dir + 0];
  const real_t hdir_1_r = params_r.h[3 * dir + 1];
  const real_t hdir_2_r = params_r.h[3 * dir + 2];

  // Unpack variables for improved readability
  const num_t rhol = *params_l.q.rho;
  const num_t rhor = *params_r.q.rho;

  const num_t rho_ul = *params_l.q.rho_u;
  const num_t rho_ur = *params_r.q.rho_u;

  const num_t rho_vl = *params_l.q.rho_v;
  const num_t rho_vr = *params_r.q.rho_v;

  const num_t rho_wl = *params_l.q.rho_w;
  const num_t rho_wr = *params_r.q.rho_w;

  const num_t rho_thetal = *params_l.q.rho_theta;
  const num_t rho_thetar = *params_r.q.rho_theta;

  const num_t inv_rhol = 1.0 / rhol;
  const num_t ul       = rho_ul * inv_rhol;
  const num_t vl       = rho_vl * inv_rhol;
  const num_t wl       = rho_wl * inv_rhol;

  const num_t inv_rhor = 1.0 / rhor;
  const num_t ur       = rho_ur * inv_rhor;
  const num_t vr       = rho_vr * inv_rhor;
  const num_t wr       = rho_wr * inv_rhor;

  // Compute the left and right-hand side pressure states
  const num_t pl = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_thetal));
  const num_t pr = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_thetar));

  // Get the speed of sound on each side (uses h_dir_dir contravariant metric)
  // 4dir = h[dir,dir]
  const num_t al = sqrt(params_l.h[4 * dir] * heat_capacity_ratio * pl * inv_rhol);
  const num_t ar = sqrt(params_r.h[4 * dir] * heat_capacity_ratio * pr * inv_rhor);

  num_t vnr = 0.0;
  num_t vnl = 0.0;

  if (dir == 0)
  {
    vnl = ul;
    vnr = ur;
  }

  if (dir == 1)
  {
    vnl = vl;
    vnr = vr;
  }

  if (dir == 2)
  {
    vnl = wl;
    vnr = wr;
  }

  // Get the maximum eigenvalue
  const num_t eig_l = (al + fabs(vnl));
  const num_t eig_r = (ar + fabs(vnr));

  num_t scaled_eig = sqrt_g_l * fmax(eig_l, eig_r);

  num_t flux_l[5];
  num_t flux_r[5];

  // Compute the left and right-hand side fluxes
  flux_l[0] = sqrt_g_l * vnl * rhol;
  flux_l[1] = sqrt_g_l * (vnl * rho_ul + hdir_0_l * pl);
  flux_l[2] = sqrt_g_l * (vnl * rho_vl + hdir_1_l * pl);
  flux_l[3] = sqrt_g_l * (vnl * rho_wl + hdir_2_l * pl);
  flux_l[4] = sqrt_g_l * vnl * rho_thetal;

  flux_r[0] = sqrt_g_r * vnr * rhor;
  flux_r[1] = sqrt_g_r * (vnr * rho_ur + hdir_0_r * pr);
  flux_r[2] = sqrt_g_r * (vnr * rho_vr + hdir_1_r * pr);
  flux_r[3] = sqrt_g_r * (vnr * rho_wr + hdir_2_r * pr);
  flux_r[4] = sqrt_g_r * vnr * rho_thetar;

  // Get the Riemann flux
  *params_l.flux.rho   = 0.5 * (flux_l[0] + flux_r[0] - scaled_eig * (rhor - rhol));
  *params_l.flux.rho_u = 0.5 * (flux_l[1] + flux_r[1] - scaled_eig * (rho_ur - rho_ul));
  *params_l.flux.rho_v = 0.5 * (flux_l[2] + flux_r[2] - scaled_eig * (rho_vr - rho_vl));
  *params_l.flux.rho_w = 0.5 * (flux_l[3] + flux_r[3] - scaled_eig * (rho_wr - rho_wl));
  *params_l.flux.rho_theta =
      0.5 * (flux_l[4] + flux_r[4] - scaled_eig * (rho_thetar - rho_thetal));

  // Copy values to the right-hand side flux
  *params_r.flux.rho       = *params_l.flux.rho;
  *params_r.flux.rho_u     = *params_l.flux.rho_u;
  *params_r.flux.rho_v     = *params_l.flux.rho_v;
  *params_r.flux.rho_w     = *params_l.flux.rho_w;
  *params_r.flux.rho_theta = *params_l.flux.rho_theta;

  // Ensure zero dissipation in advection flux at boundary points
  // Otherwise, this would create nonzero mass flux at boundaries
  if (boundary)
    scaled_eig = scaled_eig * 0.0;

  // Store the advection and pressure contribution to vertical fluxes
  *params_l.wflux_adv = 0.5 * (sqrt_g_l * rho_wl * vnl + sqrt_g_r * rho_wr * vnr -
                               scaled_eig * (rho_wr - rho_wl));
  *params_r.wflux_adv = *params_l.wflux_adv;

  const num_t wflux_pres_l   = sqrt_g_l * hdir_2_l * pl;
  const num_t wflux_pres_r   = sqrt_g_r * hdir_2_r * pr;
  const num_t wflux_pres_avg = 0.5 * (wflux_pres_l + wflux_pres_r);
  *params_l.wflux_pres       = wflux_pres_avg / pl;
  *params_r.wflux_pres       = wflux_pres_avg / pr;

  // Store pressure for later use
  *params_l.pressure = pl;
  *params_r.pressure = pr;
}
