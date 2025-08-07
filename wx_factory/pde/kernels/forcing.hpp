#ifndef FORCING_H_
#define FORCING_H_

#include "../pde.hpp"

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
DEVICE_SPACE void forcing_euler_cubesphere_3d_kernel(
    forcing_params<real_t, num_t> params,
    const bool                    verbose) {

  (void)verbose; // only used for debugging

  const num_t rho = *params.q.rho;
  const num_t u   = *params.q.rho_u / rho;
  const num_t v   = *params.q.rho_v / rho;
  const num_t w   = *params.q.rho_w / rho;

  *params.forcing.rho_u = compute_single_forcing<real_t, num_t>(
      rho,
      u,
      v,
      w,
      params.pressure,
      params.h[h11],
      params.h[h12],
      params.h[h13],
      params.h[h22],
      params.h[h23],
      params.h[h33],
      params.christoffel[c101],
      params.christoffel[c102],
      params.christoffel[c103],
      params.christoffel[c111],
      params.christoffel[c112],
      params.christoffel[c113],
      params.christoffel[c122],
      params.christoffel[c123],
      params.christoffel[c133]);

  *params.forcing.rho_v = compute_single_forcing<real_t, num_t>(
      rho,
      u,
      v,
      w,
      params.pressure,
      params.h[h11],
      params.h[h12],
      params.h[h13],
      params.h[h22],
      params.h[h23],
      params.h[h33],
      params.christoffel[c201],
      params.christoffel[c202],
      params.christoffel[c203],
      params.christoffel[c211],
      params.christoffel[c212],
      params.christoffel[c213],
      params.christoffel[c222],
      params.christoffel[c223],
      params.christoffel[c233]);

  *params.forcing.rho_w = compute_single_forcing<real_t, num_t>(
      rho,
      u,
      v,
      w,
      params.pressure,
      params.h[h11],
      params.h[h12],
      params.h[h13],
      params.h[h22],
      params.h[h23],
      params.h[h33],
      params.christoffel[c301],
      params.christoffel[c302],
      params.christoffel[c303],
      params.christoffel[c311],
      params.christoffel[c312],
      params.christoffel[c313],
      params.christoffel[c322],
      params.christoffel[c323],
      params.christoffel[c333]);
}

#endif
