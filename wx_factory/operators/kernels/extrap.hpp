#ifndef OPERATORS_EXTRAP_HPP_
#define OPERATORS_EXTRAP_HPP_

#include "../operators.hpp"

#include <cstdio>

constexpr int EXTRAP_3D_BLOCK_SIZE = 128;

template <typename real_t, typename num_t, int order>
struct extrap_all_kernel
{
  extrap_params_cubedsphere<num_t, order> params;

  template <typename... Args>
  extrap_all_kernel(Args... args) : params(0, args...) {}

  DEVICE_SPACE void operator()(const num_t* elem, const bool verbose) {

    // clang-format off
    constexpr real_t extrap_factors[][6] = {
      {1.366025403784438819, -0.366025403784438708},
      {1.478830557701235948, -0.666666666666666408, 0.187836108965430404},
      {1.526788125457266831, -0.813632449486927478, 0.400761520311650465, -0.113917196281990041},
      {1.551408049094313180, -0.893158392000071966, 0.533333333333333437, -0.267941652223387616, 0.076358661795812910},
      {1.565673200151072253, -0.940462843176349317, 0.616930055430489288, -0.379227702114613929, 0.191800014038668087, -0.054712724329265883}
    };
    // clang-format on

    static_assert(order >= 1 && order <= 6, "Order is not implemented\n");
    (void)verbose; // only used for debugging

    if (elem == nullptr)
    {
      elem = &params.elem[0];
    }

    const size_t o2       = order * order;
    const size_t rem      = params.index % o2;
    const size_t offset_x = params.index % o2;
    const size_t offset_y = rem % order + (rem / order) * o2;
    const size_t offset_z = params.index % o2;

    num_t side_x1(0.0);
    num_t side_x2(0.0);
    num_t side_y1(0.0);
    num_t side_y2(0.0);
    num_t side_z1(0.0);
    num_t side_z2(0.0);
    for (int i = 0; i < order; i++)
    {
      side_x1 += elem[offset_x * order + i] * extrap_factors[order - 2][i];
      side_x2 += elem[offset_x * order + i] * extrap_factors[order - 2][order - i - 1];
      side_y1 += elem[offset_y + order * i] * extrap_factors[order - 2][i];
      side_y2 += elem[offset_y + order * i] * extrap_factors[order - 2][order - i - 1];
      side_z1 += elem[offset_z + o2 * i] * extrap_factors[order - 2][i];
      side_z2 += elem[offset_z + o2 * i] * extrap_factors[order - 2][order - i - 1];
    }
    *params.side_x1 = side_x1;
    *params.side_x2 = side_x2;
    *params.side_y1 = side_y1;
    *params.side_y2 = side_y2;
    *params.side_z1 = side_z1;
    *params.side_z2 = side_z2;
  }
};

template <typename real_t, typename num_t, int order>
struct my_func
{
  DEVICE_SPACE void operator()(
      extrap_params_cubedsphere<num_t, order> params,
      const num_t*                            ptr,
      const bool                              verbose) {
    printf("hi. i = %d\n", verbose);
  }
};

#endif
