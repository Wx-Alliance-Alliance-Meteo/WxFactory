#ifndef PAIRS_H_
#define PAIRS_H_
#include "common/parameters.hpp"
#include "common/transform_rule.hpp"


template <typename T, typename U>
HOST_DEVICE_SPACE void convert_pair_kernel_shared(const T* a1, const T* a2, const U* coord, T* o1, T* o2, size_t idx, size_t n_coord, const TransformRule& rule) {

    U x = coord[idx % n_coord];
    U c = (2.0 * x) / (1.0 + x*x);

    T p11 = static_cast<T>(rule.s11 + c * rule.s13);
    T p12 = static_cast<T>(rule.s12 + c * rule.s14);
    T p21 = static_cast<T>(rule.s21 + c * rule.s23);
    T p22 = static_cast<T>(rule.s22 + c * rule.s24);

    T A1 = a1[idx];
    T A2 = a2[idx];

    o1[idx] = p11 * A1 + p12 * A2;
    o2[idx] = p21 * A1 + p22 * A2;
}


#endif