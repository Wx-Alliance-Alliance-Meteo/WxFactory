
constexpr size_t MAX_DERIV_ORDER = 6;

template <typename real_t, typename num_t, int order>
struct deriv_x_kernel
{
  DEVICE_SPACE void operator()(
      extrap_params_cubedsphere<num_t, order> params,
      const num_t*                            elem,
      const bool                              verbose) {

    static_assert(order >= 1 && order <= 6, "Order is not implemented\n");

    if (elem == nullptr)
    {
      elem = &params.elem[0];
    }

    num_t result(0.0);
    for (int i = 0; i < order; i++)
    {
    }
  }
};
