#ifndef PDE_INTERFACE_H
#define PDE_INTERFACE_H

#include <array>

#include "definitions.hpp"

template <typename num_t>
struct var
{
  num_t* value = nullptr;

  DEVICE_SPACE var() {}
  DEVICE_SPACE var(num_t* field, const size_t index) : value(field + index) {}
  DEVICE_SPACE var(num_t* field) : value(field) {}

  operator num_t() const { return *value; }

  num_t  operator*() const { return *value; }
  num_t& operator*() { return *value; }
};

template <typename num_t, int size>
std::array<var<num_t>, size> make_var_sequence(const num_t* offset, const size_t stride) {
  std::array<var<num_t>, size> result;
  for (int i = 0; i < size; i++)
  {
    result[i] = i * stride + offset;
  }
  return result;
}

template <typename num_t, int num_var>
struct var_multi
{
  std::array<var<num_t>, num_var> val;

  num_t  operator[](int i) const { return *val[i]; }
  num_t& operator[](int i) { return *val[i]; }

  DEVICE_SPACE var_multi(num_t* field, const size_t index, const size_t stride) :
      val{make_var_sequence<num_t, num_var>(field + index, stride)} {}
};

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

// Returns the index in a flattened array from a 4d index group
DEVICE_SPACE int
get_c_index(const int i, const int j, const int k, const int l, const int shape[4]) {
  return i * shape[1] * shape[2] * shape[3] + j * shape[2] * shape[3] + k * shape[3] + l;
}

// Return the cupy pointer
uintptr_t get_cupy_pointer(pybind11::object obj) {
  uintptr_t cp_ptr = obj.attr("data").attr("ptr").cast<uintptr_t>();
  return cp_ptr;
}

//! Extract raw pointer to given array's data and cast it to the requested type
//! \tparam num_t The type we wish to get from the input array
template <typename num_t>
num_t* get_c_ptr(const pybind11::array_t<num_t>& a) {
  return static_cast<num_t*>(a.request().ptr);
}

template <typename real_t, typename num_t>
struct forcing_params
{
  int index = -1;

  euler_state_3d<const num_t> q;
  var<const num_t>            pressure;
  var<const real_t>           sqrt_g;
  var_multi<const real_t, 9>  h;
  var_multi<const real_t, 27> christoffel;

  euler_state_3d<num_t> forcing;

  DEVICE_SPACE forcing_params(
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
};

#endif // PDE_INTERFACE_H
