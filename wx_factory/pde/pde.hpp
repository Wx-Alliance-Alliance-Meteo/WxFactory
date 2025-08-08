#ifndef PDE_INTERFACE_H
#define PDE_INTERFACE_H

#include "common/parameters.hpp"

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
struct riemann_params_cubedsphere
{
  size_t index = std::numeric_limits<size_t>::max();

  euler_state_3d<const num_t> q;
  var<const real_t>           sqrt_g;
  var_multi<const real_t, 9>  h;

  euler_state_3d<num_t> flux;
  var<num_t>            pressure;
  var<num_t>            wflux_adv;
  var<num_t>            wflux_pres;

  HOST_DEVICE_SPACE riemann_params_cubedsphere(
      const num_t*  q,      //!< Pointer to the various fields (each variable is grouped)
      const real_t* sqrt_g, //!< Pointer to sqrt_g values for all points
      const real_t* h, //!< Pointer to h contravariant values for all points (6 arrays)
      const size_t  index,  //!< Index of the grid point whose state we want to access
      const size_t  stride, //!< How many entries in the input array for each variable
      num_t*        flux,
      num_t*        pressure,
      num_t*        wflux_adv,
      num_t*        wflux_pres) :
      index(index),
      q(q, index, stride),
      sqrt_g(sqrt_g, index),
      h(h, index, stride),
      flux(flux, index, stride),
      pressure(pressure, index),
      wflux_adv(wflux_adv, index),
      wflux_pres(wflux_pres, index) {}
};

#endif // PDE_INTERFACE_H
