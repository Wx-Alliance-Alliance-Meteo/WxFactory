#ifndef FORCING_H_
#define FORCING_H_

#include "../interface.hpp"

// def compute_forcing_1(f, r, u1, u2, w, p, c01, c02, c03, c11, c12, c13, c22, c23, c33,
// h11, h12, h13, h22, h23, h33):
//     """Compute forcing for fluid velocity in a single direction based on metric terms
//     and coriolis effect.""" f[:] = (
//           2.0 *   r * (c01 * u1 + c02 * u2 + c03 * w)
//         +       c11 * (r * u1 * u1 + h11 * p)
//         + 2.0 * c12 * (r * u1 * u2 + h12 * p)
//         + 2.0 * c13 * (r * u1 * w  + h13 * p)
//         +       c22 * (r * u2 * u2 + h22 * p)
//         + 2.0 * c23 * (r * u2 * w  + h23 * p)
//         +       c33 * (r * w  * w  + h33 * p)
//     )

template <typename real_t, typename num_t>
num_t compute_single_forcing(
    const num_t  r,
    const num_t  u,
    const num_t  v,
    const num_t  w,
    const num_t  p,
    const real_t h11,
    const real_t h12,
    const real_t h13,
    const real_t h22,
    const real_t h23,
    const real_t h33,
    const real_t c01,
    const real_t c02,
    const real_t c03,
    const real_t c11,
    const real_t c12,
    const real_t c13,
    const real_t c22,
    const real_t c23,
    const real_t c33,
    const bool   verbose) {
  // if (verbose)
  // {
  //   const num_t tmp1 = 2.0 * r * (c01 * u + c02 * v + c03 * w);
  //   const num_t tmp2 = c11 * (r * u * u + h11 * p);
  //   const num_t tmp3 = 2.0 * c12 * (r * u * v + h12 * p);
  //   const num_t tmp4 = 2.0 * c13 * (r * u * w + h13 * p);
  //   const num_t tmp5 = c22 * (r * v * v + h22 * p);
  //   const num_t tmp6 = 2.0 * c23 * (r * v * w + h23 * p);
  //   const num_t tmp7 = c33 * (r * w * w + h33 * p);
  //   printf(
  //       "tmps = %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e\n"
  //       "       %11.4e %11.4e %11.4e %11.4e %11.4e\n",
  //       to_real(tmp1),
  //       to_real(tmp2),
  //       to_real(tmp3),
  //       to_real(tmp4),
  //       to_real(tmp5),
  //       to_real(tmp6),
  //       to_real(tmp7),
  //       to_real(c22),
  //       to_real(r),
  //       to_real(v),
  //       to_real(h22),
  //       to_real(p));
  // }
  // clang-format off
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
void forcing_euler_cubesphere_3d_kernel(
    forcing_params<real_t, num_t> params,
    const bool                    verbose) {

  const num_t rho = *params.q.rho;
  const num_t u   = *params.q.rho_u / rho;
  const num_t v   = *params.q.rho_v / rho;
  const num_t w   = *params.q.rho_w / rho;

  *params.forcing.rho       = -1.0;
  *params.forcing.rho_v     = -1.0;
  *params.forcing.rho_w     = -1.0;
  *params.forcing.rho_theta = -1.0;

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
      params.christoffel[c133],
      verbose && params.index == 0);

  // if (verbose && params.index == 0)
  // {
  //   printf(
  //       "%5i: rho = %11.4e, u = %11.4e, v = %11.4e, w = %11.4e, pressure = %11.4e\n",
  //       params.index,
  //       to_real(rho),
  //       to_real(u),
  //       to_real(v),
  //       to_real(w),
  //       to_real(params.pressure));
  //   printf(
  //       "%5i: h = %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e\n",
  //       params.index,
  //       params.h[h11],
  //       params.h[h12],
  //       params.h[h13],
  //       params.h[h22],
  //       params.h[h23],
  //       params.h[h33]);
  //   printf(
  //       "%5i: c = %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e\n",
  //       params.index,
  //       params.christoffel[c101],
  //       params.christoffel[c102],
  //       params.christoffel[c103],
  //       params.christoffel[c111],
  //       params.christoffel[c112],
  //       params.christoffel[c113],
  //       params.christoffel[c122],
  //       params.christoffel[c123],
  //       params.christoffel[c133]);
  //   printf("%5i: result = %11.4e\n", params.index, to_real(params.forcing.rho_u));
  // }

  // fprintf(
  //     stderr,
  //     "%5i: H = {%e, %e, %e, %e, %e, %e}\n",
  //     params.index,
  //     params.h[h11],
  //     params.h[h12],
  //     params.h[h13],
  //     params.h[h22],
  //     params.h[h23],
  //     params.h[h33]);
}

#endif
