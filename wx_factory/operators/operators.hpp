#ifndef OPERATORS_HPP_
#define OPERATORS_HPP_

#include "definitions/definitions.hpp"

template <typename num_t, int order>
struct element
{
  static constexpr size_t elem_size = order * order * order;

  num_t* values;

  HOST_DEVICE_SPACE element(num_t* values, const size_t index) :
      values(values + index * elem_size) {}

  HOST_DEVICE_SPACE num_t  operator[](int i) const { return values[i]; }
  HOST_DEVICE_SPACE num_t& operator[](int i) { return values[i]; }
  HOST_DEVICE_SPACE void   move_index(const int64_t index_change) {
    values += index_change * elem_size;
  }
};

inline HOST_DEVICE_SPACE size_t
compute_itf_offset(const size_t index, const size_t order) {
  const size_t rem      = index % (order * order);
  const size_t elem_pos = (index - rem) * 2;
  //   printf(
  //       "itf offset for index %3zu = %zu (%zu, %zu)\n",
  //       index,
  //       elem_pos + rem,
  //       elem_pos,
  //       rem);
  return elem_pos + rem;
  //   return index * 2;
}

template <typename num_t, int order>
struct extrap_params_cubedsphere
{
  static constexpr size_t itf_size  = order * order;
  static constexpr size_t elem_size = element<num_t, order>::elem_size;

  size_t index   = std::numeric_limits<size_t>::max();
  size_t elem_id = std::numeric_limits<size_t>::max();

  element<const num_t, order> elem;
  var<num_t>                  side_x1;
  var<num_t>                  side_x2;
  var<num_t>                  side_y1;
  var<num_t>                  side_y2;
  var<num_t>                  side_z1;
  var<num_t>                  side_z2;

  template <typename ArrayType>
  extrap_params_cubedsphere(
      const size_t     index,
      const ArrayType& q,
      ArrayType&       result_x,
      ArrayType&       result_y,
      ArrayType&       result_z) :
      index(index),
      elem_id(index / itf_size),
      elem(get_raw_pointer<const num_t>(q), elem_id),
      side_x1(get_raw_pointer<num_t>(result_x), compute_itf_offset(index, order)),
      side_x2(
          get_raw_pointer<num_t>(result_x),
          compute_itf_offset(index, order) + itf_size),
      side_y1(get_raw_pointer<num_t>(result_y), compute_itf_offset(index, order)),
      side_y2(
          get_raw_pointer<num_t>(result_y),
          compute_itf_offset(index, order) + itf_size),
      side_z1(get_raw_pointer<num_t>(result_z), compute_itf_offset(index, order)),
      side_z2(
          get_raw_pointer<num_t>(result_z),
          compute_itf_offset(index, order) + itf_size) {}

  DEVICE_SPACE void set_index(const size_t new_index) {
    const size_t old_index   = index;
    const size_t old_elem_id = elem_id;
    const size_t new_elem_id = new_index / itf_size;

    const int64_t elem_diff = static_cast<int64_t>(new_elem_id) - old_elem_id;
    const int64_t offset_diff =
        static_cast<int64_t>(new_index % itf_size) - old_index % itf_size;

    const int64_t diff = elem_diff * itf_size * 2 + offset_diff;

    index   = new_index;
    elem_id = new_elem_id;
    elem.move_index(elem_diff);
    side_x1.move_index(diff);
    side_x2.move_index(diff);
    side_y1.move_index(diff);
    side_y2.move_index(diff);
    side_z1.move_index(diff);
    side_z2.move_index(diff);
  }
};
#endif
