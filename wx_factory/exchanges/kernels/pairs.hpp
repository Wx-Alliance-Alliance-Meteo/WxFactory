#ifndef PAIRS_H_
#define PAIRS_H_
#include "common/parameters.hpp"
#include "exchanges/transform_rule.hpp"


template <typename num_t, typename real_t>
HOST_DEVICE_SPACE void convert_pair_kernel_shared(const num_t* a1, const num_t* a2, const real_t* coord, num_t* o1, num_t* o2, size_t idx, size_t n_coord, const TransformRule& rule) {

    real_t x = coord[idx % n_coord];
    real_t c = (2.0 * x) / (1.0 + x*x);
    // real_t c = (real_t(2) * x) / (real_t(1) + x*x); // this explodes

    num_t p11 = static_cast<num_t>(rule.s11 + c * rule.s13);
    num_t p12 = static_cast<num_t>(rule.s12 + c * rule.s14);
    num_t p21 = static_cast<num_t>(rule.s21 + c * rule.s23);
    num_t p22 = static_cast<num_t>(rule.s22 + c * rule.s24);

    num_t A1 = a1[idx];
    num_t A2 = a2[idx];

    o1[idx] = p11 * A1 + p12 * A2;
    o2[idx] = p21 * A1 + p22 * A2;
}

#endif