#ifndef PDE_INTERFACE_H
#define PDE_INTERFACE_H

#include "definitions/definitions.hpp"

template <typename num_t>
struct euler_state_2d
{
  var<num_t> rho;
  var<num_t> rho_u;
  var<num_t> rho_w;
  var<num_t> rho_theta;

  //! euler_state_2d constructor
  DEVICE_SPACE euler_state_2d(
      num_t*       q,     //!< Pointer to the various fields (each variable is grouped)
      const size_t index, //!< Index of the grid point whose state we want to access
      const size_t stride //!< How many entries in the input array for each variable
  ) {
    if (q != nullptr)
    {
      rho       = var<num_t>(q, index);
      rho_u     = var<num_t>(q + stride, index);
      rho_w     = var<num_t>(q + (2 * stride), index);
      rho_theta = var<num_t>(q + (3 * stride), index);
    }
  }
};

template <typename num_t>
struct euler_state_3d
{
  var<num_t> rho;
  var<num_t> rho_u;
  var<num_t> rho_v;
  var<num_t> rho_w;
  var<num_t> rho_theta;

  // Constructor
  DEVICE_SPACE euler_state_3d(
      num_t*       q,     //!< Pointer to the various fields (each variable is grouped)
      const size_t index, //!< Index of the grid point whose state we want to access
      const size_t stride //!< How many entries in the input array for each variable
      ) :
      rho(q, index),
      rho_u(q + stride, index),
      rho_v(q + (2 * stride), index),
      rho_w(q + (3 * stride), index),
      rho_theta(q + (4 * stride), index) {}

  DEVICE_SPACE void move_index(const int64_t diff) {
    rho.move_index(diff);
    rho_u.move_index(diff);
    rho_v.move_index(diff);
    rho_w.move_index(diff);
    rho_theta.move_index(diff);
  }
};

template <typename num_t>
struct shallow_water_state_2d
{
  num_t h  = nullptr;
  num_t hu = nullptr;
  num_t hv = nullptr;

  // Constructor
  DEVICE_SPACE shallow_water_state_2d(num_t* q, const size_t ind, const size_t stride) {
    if (q != nullptr)
    {
      h  = &q[ind];
      hu = &q[ind + stride];
      hv = &q[ind + 2 * stride];
    }
  }
};

// Stores the parameters of type <num_t> for equations <state> required to call a given
// kernel
template <typename num_t, template <typename> class state>
struct kernel_params
{
  state<const num_t> q;
  state<num_t>       flux[3] = {};

  // Constructor
  DEVICE_SPACE kernel_params(
      const num_t* q,
      num_t*       f_x1,
      num_t*       f_x2,
      num_t*       f_x3,
      const int    ind,
      const int    stride) :
      q(q, ind, stride),
      flux{
          state<num_t>(f_x1, ind, stride),
          state<num_t>(f_x2, ind, stride),
          state<num_t>(f_x3, ind, stride)} {}
};

template <typename real_t, typename num_t>
struct forcing_params
{
  size_t index = std::numeric_limits<size_t>::max();

  euler_state_3d<const num_t> q;
  var<const num_t>            pressure;
  var<const real_t>           sqrt_g;
  var_multi<const real_t, 9>  h;
  var_multi<const real_t, 27> christoffel;

  euler_state_3d<num_t> forcing;

  HOST_DEVICE_SPACE forcing_params(
      const num_t*  q,      //!< Pointer to the various fields (each variable is grouped)
      const num_t*  p,      //!< Pointer to pressure values for all points
      const real_t* sqrt_g, //!< Pointer to sqrt_g values for all points
      const real_t* h, //!< Pointer to h contravariant values for all points (6 arrays)
      const real_t* christoffel,
      num_t* forcing,     //!< [out] Pointer to entire array where forcing will be stored
      const size_t index, //!< Index of the grid point whose state we want to access
      const size_t stride //!< How many entries in the input array for each variable
      ) :
      index(index),
      q(q, index, stride),
      pressure(p, index),
      sqrt_g(sqrt_g, index),
      h(h, index, stride),
      christoffel(christoffel, index, stride),
      forcing(forcing, index, stride) {}

  DEVICE_SPACE void set_index(const size_t new_index) {
    const size_t  old_index = index;
    const int64_t diff      = static_cast<int64_t>(new_index) - old_index;
    index                   = new_index;
    q.move_index(diff);
    pressure.move_index(diff);
    sqrt_g.move_index(diff);
    h.move_index(diff);
    christoffel.move_index(diff);
    forcing.move_index(diff);
  }
};

template <typename real_t, typename num_t>
struct kernel_params_cubedsphere
{
  size_t index = std::numeric_limits<size_t>::max();

  euler_state_3d<const num_t> q;
  var<const real_t>           sqrt_g;
  var_multi<const real_t, 9>  h;

  euler_state_3d<num_t> flux[3];
  var<const num_t>      pressure;
  euler_state_3d<num_t> wflux_adv[3];
  euler_state_3d<num_t> wflux_pres[3];
  var<const num_t>      logp;

  HOST_DEVICE_SPACE kernel_params_cubedsphere(
      const num_t*  q,      //!< Pointer to the various fields (each variable is grouped)
      const real_t* sqrt_g, //!< Pointer to sqrt_g values for all points
      const real_t* h, //!< Pointer to h contravariant values for all points (6 arrays)
      const size_t  index,  //!< Index of the grid point whose state we want to access
      const size_t  stride, //!< How many entries in the input array for each variable
      num_t*        flux_x1,
      num_t*        flux_x2,
      num_t*        flux_x3,
      num_t*        pressure,
      num_t*        wflux_adv_x1,
      num_t*        wflux_adv_x2,
      num_t*        wflux_adv_x3,
      num_t*        wflux_pres_x1,
      num_t*        wflux_pres_x2,
      num_t*        wflux_pres_x3,
      num_t*        logp) :
      index(index),
      q(q, index, stride),
      sqrt_g(sqrt_g, index),
      h(h, index, stride),
      flux{
          euler_state_3d<num_t>(flux_x1, index, stride),
          euler_state_3d<num_t>(flux_x2, index, stride),
          euler_state_3d<num_t>(flux_x3, index, stride)},
      pressure(pressure, index),
      wflux_adv{
          euler_state_3d<num_t>(wflux_adv_x1, index, stride),
          euler_state_3d<num_t>(wflux_adv_x2, index, stride),
          euler_state_3d<num_t>(wflux_adv_x3, index, stride)},
      wflux_pres{
          euler_state_3d<num_t>(wflux_adv_x1, index, stride),
          euler_state_3d<num_t>(wflux_adv_x2, index, stride),
          euler_state_3d<num_t>(wflux_adv_x3, index, stride)},
      logp(logp, index) {}

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

#endif // PDE_INTERFACE_H
