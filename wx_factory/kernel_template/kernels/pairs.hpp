#ifndef PAIRS_H_
#define PAIRS_H_

#include <iostream>
#include <vector>

struct TransformRule {
    int s11, s12, s13, s14;
    int s21, s22, s23, s24;
};

extern const TransformRule rules[6][4];

template <typename T, typename U>
void convert_pair(const T* a1, const T* a2, const U* coord, T* o1, T* o2, int panel, int neighbour, int n_coord, int var_size)
{


    const TransformRule& rule = rules[panel][neighbour];
    const T s11 = rule.s11, s12 = rule.s12, s13 = rule.s13, s14 = rule.s14;
    const T s21 = rule.s21, s22 = rule.s22, s23 = rule.s23, s24 = rule.s24;

    // Precomputed c and scaling coefficients
    // Further optimization: thread local buffer
    // std::vector<T> cs13(n_coord), cs14(n_coord), cs23(n_coord), cs24(n_coord);
    
    std::vector<T> p11(n_coord), p12(n_coord), p21(n_coord), p22(n_coord);

    for (int j = 0; j < n_coord; ++j) {
        U x = coord[j];
        U c = (2.0 * x) / (1.0 + x*x);
        // cs13[j] = c * s13;
        // cs14[j] = c * s14;
        // cs23[j] = c * s23;
        // cs24[j] = c * s24;
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

        // o1[i] = s11*A1 + s12*A2 + c*(s13*A1 + s14*A2);
        // o2[i] = s21*A1 + s22*A2 + c*(s23*A1 + s24*A2);

        // T p11 = s11 + cs13[idx];
        // T p12 = s12 + cs14[idx];
        // T p21 = s21 + cs23[idx];
        // T p22 = s22 + cs24[idx];

        // o1[i] = p11*A1 + p12*A2;
        // o2[i] = p21*A1 + p22*A2;
        o1[i] = p11[idx] * A1 + p12[idx] * A2;
        o2[i] = p21[idx] * A1 + p22[idx] * A2;

        if (++idx == n_coord) { idx = 0; } // Removes modulo overhead to replaces U c = c_vals[i % n_coord];
    }

}


constexpr TransformRule rules[6][4] = {
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