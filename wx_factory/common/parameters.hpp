#ifndef COMMON_PARAMETERS_H
#define COMMON_PARAMETERS_H

//! \file
//! A set of constants, helper functions and classes used to manage kernel parameters

#include "functions.hpp"

//!{ \name Metric array indices
const int h11 = 0;
const int h12 = 1;
const int h13 = 2;
const int h21 = 3;
const int h22 = 4;
const int h23 = 5;
const int h31 = 6;
const int h32 = 7;
const int h33 = 8;

const int c101 = 0;
const int c102 = 1;
const int c103 = 2;
const int c111 = 3;
const int c112 = 4;
const int c113 = 5;
const int c122 = 6;
const int c123 = 7;
const int c133 = 8;
const int c201 = 9;
const int c202 = 10;
const int c203 = 11;
const int c211 = 12;
const int c212 = 13;
const int c213 = 14;
const int c222 = 15;
const int c223 = 16;
const int c233 = 17;
const int c301 = 18;
const int c302 = 19;
const int c303 = 20;
const int c311 = 21;
const int c312 = 22;
const int c313 = 23;
const int c322 = 24;
const int c323 = 25;
const int c333 = 26;
//!}

template <typename num_t>
struct var
{
  num_t* value = nullptr;

  HOST_DEVICE_SPACE var() {}
  HOST_DEVICE_SPACE var(num_t* field, const size_t index) : value(field + index) {}
  HOST_DEVICE_SPACE var(num_t* field) : value(field) {}

  HOST_DEVICE_SPACE operator num_t() const { return *value; }

  HOST_DEVICE_SPACE num_t  operator*() const { return *value; }
  HOST_DEVICE_SPACE num_t& operator*() { return *value; }

  HOST_DEVICE_SPACE void move_index(const int64_t index_change) { value += index_change; }
};

template <typename num_t, int size>
DEVICE_SPACE array<var<num_t>, size>
             make_var_sequence(const num_t* offset, const size_t stride) {
  array<var<num_t>, size> result;
  for (int i = 0; i < size; i++)
  {
    result[i] = i * stride + offset;
  }
  return result;
}

template <typename num_t, int num_var>
struct var_multi
{
  var<num_t> val[num_var];

  HOST_DEVICE_SPACE num_t  operator[](int i) const { return *val[i]; }
  HOST_DEVICE_SPACE num_t& operator[](int i) { return *val[i]; }

  HOST_DEVICE_SPACE var_multi(num_t* field, const size_t index, const size_t stride) {
    for (int i = 0; i < num_var; ++i)
    {
      val[i] = field + index + i * stride;
    }
  }

  HOST_DEVICE_SPACE void move_index(const int64_t index_change) {
    for (int i = 0; i < num_var; ++i)
    {
      val[i].move_index(index_change);
    }
  }
};

#endif // COMMON_PARAMETERS_H
