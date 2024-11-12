#include "../definitions.hpp"

template<typename num_t>
DEVICE_SPACE void riemann_eulercartesian_ausm_2d_kernel(kernel_params<num_t, euler_state_2d> params_l, kernel_params<num_t, euler_state_2d> params_r, const int dir)
{
  // Unpack variables for improved readability
  const num_t rhol = *params_l.q.rho; 
  const num_t rhor = *params_r.q.rho; 

  const num_t rho_ul = *params_l.q.rho_u; 
  const num_t rho_ur = *params_r.q.rho_w; 

  const num_t rho_wl = *params_l.q.rho_w; 
  const num_t rho_wr = *params_r.q.rho_w; 

  const num_t rho_thetal = *params_l.q.rho_theta;
  const num_t rho_thetar = *params_r.q.rho_theta;

  const num_t inv_rhol = 1.0 / rhol;
  const num_t ul = rho_ul * inv_rhol;
  const num_t wl = rho_wl * inv_rhol;

  const num_t inv_rhor = 1.0 / rhor;
  const num_t ur = rho_ur * inv_rhor;
  const num_t wr = rho_wr * inv_rhor;

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

  const num_t M = 0.25 * (Ml * Ml - Mr * Mr);
  const num_t Mmax = max(0.0, M) * al;
  const num_t Mmin = min(0.0, M) * ar;

  // Set the interface fluxes
  *params_l.flux[dir].rho = rhol * Mmax + rhor * Mmin;
  if(dir==0)
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
  *params_r.flux[dir].rho = *params_l.flux[dir].rho;
  *params_r.flux[dir].rho_u = *params_l.flux[dir].rho_u;
  *params_r.flux[dir].rho_w = *params_l.flux[dir].rho_w;
  *params_r.flux[dir].rho_theta = *params_l.flux[dir].rho_theta;
}
