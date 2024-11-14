#ifndef INTERFACE_H 
#define INTERFACE_H

#include "definitions.hpp"

template<typename num_t> 
struct euler_state_2d
{
  num_t* rho = nullptr;
  num_t* rho_u = nullptr;
  num_t* rho_w = nullptr;
  num_t* rho_theta = nullptr;

  DEVICE_SPACE euler_state_2d(num_t* q, const int ind, const int stride) 
  {
    if(q!=nullptr)
    {
      rho = &q[ind];
      rho_u = &q[ind + stride];
      rho_w = &q[ind + 2*stride];
      rho_theta = &q[ind + 3*stride];
    }
  }
};

template<typename num_t> 
struct euler_state_3d
{
  num_t rho = nullptr;
  num_t rho_u = nullptr;
  num_t rho_v = nullptr;
  num_t rho_w = nullptr;
  num_t rho_theta = nullptr;

  // Constructor
  DEVICE_SPACE euler_state_3d(num_t* q, const int ind, const int stride) :
    rho(&q[ind]),
    rho_u(&q[ind + stride]),
    rho_v(&q[ind + 2*stride]),
    rho_w(&q[ind + 3*stride]),
    rho_theta(&q[ind + 4*stride]) {}
};

template<typename num_t> 
struct shallow_water_state_2d
{
  num_t h = nullptr;
  num_t hu = nullptr;
  num_t hv = nullptr;

  // Constructor
  DEVICE_SPACE shallow_water_state_2d(num_t* q, const int ind, const int stride)
  {
    if(q!=nullptr)
    {
      h = &q[ind];
      hu = &q[ind + stride];
      hv = &q[ind + 2*stride];
    }
  }
};

// Stores the parameters of type <num_t> for equations <state> required to call a given kernel
template <typename num_t, template <typename> class state>
struct kernel_params
{
  state<const num_t> q;
  state<num_t> flux[3] = {};
  
  // Constructor
  DEVICE_SPACE kernel_params(const num_t *q, num_t *f_x1, num_t *f_x2, num_t *f_x3, const int ind, const int stride) :
  q(q, ind, stride), 
  flux{state<num_t>(f_x1, ind, stride), 
       state<num_t>(f_x2, ind, stride), 
       state<num_t>(f_x3, ind, stride)} {}
};


// Returns the index in a flattened array from a 4d index group
DEVICE_SPACE int get_c_index(const int i, const int j, const int k, const int l, const int shape[4])
{
  return i*shape[1]*shape[2]*shape[3] + j*shape[2]*shape[3] + k*shape[3] + l;
}

// Return the cupy pointer
uintptr_t get_cupy_pointer(pybind11::object obj)
{
  uintptr_t cp_ptr = obj.attr("data").attr("ptr").cast<uintptr_t>();
  return cp_ptr;
}

#endif // INTERFACE_H