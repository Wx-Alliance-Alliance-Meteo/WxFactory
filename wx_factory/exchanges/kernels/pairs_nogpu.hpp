#ifndef PAIRS__NOGPU_H_
#define PAIRS__NOGPU_H_

#include <iostream>
#include <vector>

struct TransformRule {
    int s11, s12, s13, s14;
    int s21, s22, s23, s24;
};

extern const TransformRule rules_cpu[6][4];

template <typename T, typename U>
void convert_pair_nogpu(const T* a1, const T* a2, const U* coord, T* o1, T* o2, int panel, int neighbour, int n_coord, int var_size)
{


    const TransformRule& rule = rules_cpu[panel][neighbour];
    const T s11 = rule.s11, s12 = rule.s12, s13 = rule.s13, s14 = rule.s14;
    const T s21 = rule.s21, s22 = rule.s22, s23 = rule.s23, s24 = rule.s24;

    // Precomputed c and scaling coefficients
    // Further optimization: thread local buffer
    // std::vector<T> cs13(n_coord), cs14(n_coord), cs23(n_coord), cs24(n_coord);
    
    std::vector<T> p11(n_coord), p12(n_coord), p21(n_coord), p22(n_coord);

    for (int j = 0; j < n_coord; ++j) {
        U x = coord[j];
        U c = (2.0 * x) / (1.0 + x*x);

        p11[j] = s11 + static_cast<T>(c * s13);
        p12[j] = s12 + static_cast<T>(c * s14);
        p21[j] = s21 + static_cast<T>(c * s23);
        p22[j] = s22 + static_cast<T>(c * s24);
    }

    int idx = 0;
    for (int i = 0; i < var_size; ++i) {
        // U c = c_vals[i % n_coord];

        T A1 = a1[i];
        T A2 = a2[i];

        o1[i] = p11[idx] * A1 + p12[idx] * A2;
        o2[i] = p21[idx] * A1 + p22[idx] * A2;

        if (++idx == n_coord) { idx = 0; } // Removes modulo overhead to replaces U c = c_vals[i % n_coord];
    }

}


inline constexpr TransformRule rules_cpu[6][4] = {
    // Panel 0
    {
        { 1, 0, 0, 1,   0, 1, 0, 0 },  // South
        { 1, 0, 0, -1,  0, 1, 0, 0 },  // North
        { 1, 0, 0, 0,   0, 1, 1, 0 },  // West
        { 1, 0, 0, 0,   0, 1, -1, 0 }, // East
    },
    // Panel 1
    {
        { 0, 1, 0, 0,  -1, 0, 0, -1 }, // South
        { 0,-1, 0, 0,   1, 0, 0, -1 }, // North
        { 1, 0, 0, 0,   0, 1, 1, 0 },  // West
        { 1, 0, 0, 0,   0, 1,-1, 0 },  // East
    },
    // Panel 2
    {
        {-1, 0, 0, -1,  0,-1, 0, 0 },  // South
        {-1, 0, 0, 1,   0,-1, 0, 0 },  // North
        { 1, 0, 0, 0,   0, 1, 1, 0 },  // West
        { 1, 0, 0, 0,   0, 1,-1, 0 },  // East
    },
    // Panel 3
    {
        { 0,-1, 0, 0,   1, 0, 0, 1 },  // South
        { 0, 1, 0, 0,  -1, 0, 0, 1 },  // North
        { 1, 0, 0, 0,   0, 1, 1, 0 },  // West
        { 1, 0, 0, 0,   0, 1,-1, 0 },  // East
    },
    // Panel 4
    {
        { 1, 0, 0, 1,   0, 1, 0, 0 },  // South
        {-1, 0, 0, 1,   0,-1, 0, 0 },  // North
        { 0,-1, -1, 0,   1, 0, 0, 0 }, // West
        { 0, 1, -1, 0,  -1, 0, 0, 0 }, // East
    },
    // Panel 5
    {
        {-1, 0, 0,-1,   0,-1, 0, 0 },  // South
        { 1, 0, 0,-1,   0, 1, 0, 0 },  // North
        { 0, 1, 1, 0,  -1, 0, 0, 0 },  // West
        { 0,-1, 1, 0,   1, 0, 0, 0 },  // East
    }
};


#endif