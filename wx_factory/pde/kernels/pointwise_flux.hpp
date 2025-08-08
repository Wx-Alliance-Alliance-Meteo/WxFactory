#ifndef PDE_KERNELS_POINTWISE_FLUX_HPP
#define PDE_KERNELS_POINTWISE_FLUX_HPP

#include "common/parameters.hpp"
#include "common/physical_constants.hpp"

template <typename num_t>
DEVICE_SPACE void
pointwise_eulercartesian_2d_kernel(kernel_params<num_t, euler_state_2d> params) {
  // Extract variables from state pointer
  const num_t rho       = *params.q.rho;
  const num_t rho_u     = *params.q.rho_u;
  const num_t rho_w     = *params.q.rho_w;
  const num_t rho_theta = *params.q.rho_theta;

  // Extract velocity components and compute pressure
  const num_t inv_rho = 1.0 / rho;
  const num_t u       = rho_u * inv_rho;
  const num_t w       = rho_w * inv_rho;

  // Get the pressure
  const num_t p = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_theta));

  // Set the values of the fluxes
  *params.flux[0].rho       = rho_u;
  *params.flux[0].rho_u     = rho_u * u + p;
  *params.flux[0].rho_w     = rho_u * w;
  *params.flux[0].rho_theta = rho_theta * u;

  *params.flux[1].rho       = rho_w;
  *params.flux[1].rho_u     = rho_w * u;
  *params.flux[1].rho_w     = rho_w * w + p;
  *params.flux[1].rho_theta = rho_theta * w;
}

template <typename real_t, typename num_t>
struct kernel_params_cubedsphere
{
  size_t index = 0;

  euler_state_3d<const num_t> q;
  var<const real_t>           sqrt_g;
  var_multi<const real_t, 9>  h;

  euler_state_3d<num_t> flux[3];
  var<num_t>            pressure;
  var<num_t>            wflux_adv[3];
  var<num_t>            wflux_pres[3];
  var<num_t>            logp;

  template <typename ArrayT1, typename ArrayT2>
  kernel_params_cubedsphere(
      const ArrayT1& q,      //!< Pointer to the various fields (each variable is grouped)
      const ArrayT2& sqrt_g, //!< Pointer to sqrt_g values for all points
      const ArrayT2& h, //!< Pointer to h contravariant values for all points (6 arrays)
      ArrayT1&       flux_x1,
      ArrayT1&       flux_x2,
      ArrayT1&       flux_x3,
      ArrayT1&       pressure,
      ArrayT1&       wflux_adv_x1,
      ArrayT1&       wflux_adv_x2,
      ArrayT1&       wflux_adv_x3,
      ArrayT1&       wflux_pres_x1,
      ArrayT1&       wflux_pres_x2,
      ArrayT1&       wflux_pres_x3,
      ArrayT1&       logp,
      const size_t   stride //!< How many entries in the input array for each variable
      ) :
      q(get_raw_ptr<num_t>(q), index, stride),
      sqrt_g(get_raw_ptr<real_t>(sqrt_g), index),
      h(get_raw_ptr<real_t>(h), index, stride),
      flux{
          euler_state_3d<num_t>(get_raw_ptr<num_t>(flux_x1), index, stride),
          euler_state_3d<num_t>(get_raw_ptr<num_t>(flux_x2), index, stride),
          euler_state_3d<num_t>(get_raw_ptr<num_t>(flux_x3), index, stride)},
      pressure(get_raw_ptr<num_t>(pressure), index),
      wflux_adv{
          {get_raw_ptr<num_t>(wflux_adv_x1), index},
          {get_raw_ptr<num_t>(wflux_adv_x2), index},
          {get_raw_ptr<num_t>(wflux_adv_x3), index}},
      wflux_pres{
          {get_raw_ptr<num_t>(wflux_pres_x1), index},
          {get_raw_ptr<num_t>(wflux_pres_x2), index},
          {get_raw_ptr<num_t>(wflux_pres_x3), index}},
      logp(get_raw_ptr<num_t>(logp), index) {}

  DEVICE_SPACE void set_index(const size_t new_index) {
    const size_t  old_index = index;
    const int64_t diff      = static_cast<int64_t>(new_index) - old_index;
    index                   = new_index;
    q.move_index(diff);
    sqrt_g.move_index(diff);
    h.move_index(diff);
    flux[0].move_index(diff);
    flux[1].move_index(diff);
    flux[2].move_index(diff);
    pressure.move_index(diff);
    wflux_adv[0].move_index(diff);
    wflux_adv[1].move_index(diff);
    wflux_adv[2].move_index(diff);
    wflux_pres[0].move_index(diff);
    wflux_pres[1].move_index(diff);
    wflux_pres[2].move_index(diff);
    logp.move_index(diff);
  }
};
template <typename real_t, typename num_t>
struct PointwiseFluxEuler3DKernel
{
  kernel_params_cubedsphere<real_t, num_t> params;

  template <typename... Args>
  PointwiseFluxEuler3DKernel(Args... args) : params(args...) {}

  DEVICE_SPACE void operator()(const size_t thread_id, bool verbose) {

    (void)verbose; // disable compiler warning. Only used for debugging

    params.set_index(thread_id);

    // Extract metric
    const real_t sqrt_g = *params.sqrt_g;

    // Extract variables from state pointer
    const num_t rho       = *params.q.rho;
    const num_t rho_u     = *params.q.rho_u;
    const num_t rho_v     = *params.q.rho_v;
    const num_t rho_w     = *params.q.rho_w;
    const num_t rho_theta = *params.q.rho_theta;

    // Extract velocity components and compute pressure
    const num_t inv_rho = 1.0 / rho;
    const num_t u       = rho_u * inv_rho;
    const num_t v       = rho_v * inv_rho;
    const num_t w       = rho_w * inv_rho;

    // Get the pressure
    const num_t p = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_theta));

    // Set the fluxes
    *params.flux[0].rho       = sqrt_g * rho_u;
    *params.flux[0].rho_u     = sqrt_g * (rho_u * u + params.h[h11] * p);
    *params.flux[0].rho_v     = sqrt_g * (rho_v * u + params.h[h12] * p);
    *params.flux[0].rho_w     = sqrt_g * (rho_w * u + params.h[h13] * p);
    *params.flux[0].rho_theta = sqrt_g * rho_theta * u;

    *params.flux[1].rho       = sqrt_g * rho_v;
    *params.flux[1].rho_u     = sqrt_g * (rho_u * v + params.h[h21] * p);
    *params.flux[1].rho_v     = sqrt_g * (rho_v * v + params.h[h22] * p);
    *params.flux[1].rho_w     = sqrt_g * (rho_w * v + params.h[h23] * p);
    *params.flux[1].rho_theta = sqrt_g * rho_theta * v;

    *params.flux[2].rho       = sqrt_g * rho_w;
    *params.flux[2].rho_u     = sqrt_g * (rho_u * w + params.h[h31] * p);
    *params.flux[2].rho_v     = sqrt_g * (rho_v * w + params.h[h32] * p);
    *params.flux[2].rho_w     = sqrt_g * (rho_w * w + params.h[h33] * p);
    *params.flux[2].rho_theta = sqrt_g * rho_theta * w;

    // Set the fluxes
    *params.wflux_adv[0] = sqrt_g * rho_w * u;
    *params.wflux_adv[1] = sqrt_g * rho_w * v;
    *params.wflux_adv[2] = sqrt_g * rho_w * w;

    *params.wflux_pres[0] = sqrt_g * params.h[h13];
    *params.wflux_pres[1] = sqrt_g * params.h[h23];
    *params.wflux_pres[2] = sqrt_g * params.h[h33];

    *params.pressure = p;
    *params.logp     = log(p);
  }
};

#endif // PDE_KERNELS_POINTWISE_FLUX_HPP
