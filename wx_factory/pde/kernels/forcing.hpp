#ifndef FORCING_H_
#define FORCING_H_

#include "common/parameters.hpp"

template <typename real_t, typename num_t>
struct forcing_params
{
  size_t index = 0;

  euler_state_3d<const num_t> q;
  var<const num_t>            pressure;
  var<const real_t>           sqrt_g;
  var_multi<const real_t, 9>  h;
  var_multi<const real_t, 27> christoffel;

  euler_state_3d<num_t> forcing;

  template <typename ArrayT1, typename ArrayT2>
  forcing_params(
      const ArrayT1& q,      //!< Pointer to the various fields (each variable is grouped)
      const ArrayT1& p,      //!< Pointer to pressure values for all points
      const ArrayT2& sqrt_g, //!< Pointer to sqrt_g values for all points
      const ArrayT2& h, //!< Pointer to h contravariant values for all points (6 arrays)
      const ArrayT2& christoffel,
      ArrayT1& forcing,   //!< [out] Pointer to entire array where forcing will be stored
      const size_t stride //!< How many entries in the input array for each variable
      ) :
      q(get_raw_ptr<num_t>(q), index, stride),
      pressure(get_raw_ptr<num_t>(p), index),
      sqrt_g(get_raw_ptr<real_t>(sqrt_g), index),
      h(get_raw_ptr<real_t>(h), index, stride),
      christoffel(get_raw_ptr<real_t>(christoffel), index, stride),
      forcing(get_raw_ptr<num_t>(forcing), index, stride) {}

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
DEVICE_SPACE num_t compute_single_forcing(
    // clang-format off
    const num_t  r, const num_t  u, const num_t  v, const num_t  w, const num_t  p,
    const real_t h11, const real_t h12, const real_t h13, const real_t h22, const real_t h23, const real_t h33,
    const real_t c01, const real_t c02, const real_t c03,
    const real_t c11, const real_t c12, const real_t c13,
    const real_t c22, const real_t c23, const real_t c33
) {
  return 2.0 *   r * (c01 * u + c02 * v + c03 * w) +
               c11 * (r * u * u + h11 * p) +
         2.0 * c12 * (r * u * v + h12 * p) +
         2.0 * c13 * (r * u * w + h13 * p) +
               c22 * (r * v * v + h22 * p) +
         2.0 * c23 * (r * v * w + h23 * p) +
               c33 * (r * w * w + h33 * p);
  // clang-format on
}

template <typename real_t, typename num_t>
struct ForcingKernel
{

  forcing_params<real_t, num_t> params;

  template <typename... Args>
  ForcingKernel(Args... args) : params(args...) {}

  DEVICE_SPACE void operator()(const size_t thread_id, const bool verbose) {

    (void)verbose; // only used for debugging

    params.set_index(thread_id);

    const num_t rho = *params.q.rho;
    const num_t u   = *params.q.rho_u / rho;
    const num_t v   = *params.q.rho_v / rho;
    const num_t w   = *params.q.rho_w / rho;

    // clang-format off
    *params.forcing.rho_u = compute_single_forcing<real_t, num_t>(
        rho, u, v, w, params.pressure,
        params.h[h11], params.h[h12], params.h[h13], params.h[h22], params.h[h23], params.h[h33],
        params.christoffel[c101], params.christoffel[c102], params.christoffel[c103],
        params.christoffel[c111], params.christoffel[c112], params.christoffel[c113],
        params.christoffel[c122], params.christoffel[c123], params.christoffel[c133]);

    *params.forcing.rho_v = compute_single_forcing<real_t, num_t>(
        rho, u, v, w, params.pressure,
        params.h[h11], params.h[h12], params.h[h13], params.h[h22], params.h[h23], params.h[h33],
        params.christoffel[c201], params.christoffel[c202], params.christoffel[c203],
        params.christoffel[c211], params.christoffel[c212], params.christoffel[c213],
        params.christoffel[c222], params.christoffel[c223], params.christoffel[c233]);

    *params.forcing.rho_w = compute_single_forcing<real_t, num_t>(
        rho, u, v, w, params.pressure,
        params.h[h11], params.h[h12], params.h[h13], params.h[h22], params.h[h23], params.h[h33],
        params.christoffel[c301], params.christoffel[c302], params.christoffel[c303],
        params.christoffel[c311], params.christoffel[c312], params.christoffel[c313],
        params.christoffel[c322], params.christoffel[c323], params.christoffel[c333]);
    // clang-format on
  }
};

#endif
