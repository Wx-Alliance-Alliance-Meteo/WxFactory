#ifndef PAIRS_H_
#define PAIRS_H_

enum Neighbour : int {South = 0, North = 1, West = 2, East = 3};


/*
Generalized form:
    r1 = s11*a1 + s12*a2 +c*(s13*a1 + s14*a2)
    r2 = s21*a1 + s22*a2 +c*(s23*a1 + s24*a2)
    c(x, y) = 2*x /(y + x**2)

    n: nbr of poitns converted
*/
template <typename T>
void convert_pair(T* a1, T* a2, T* coord, T* o1, T* o2, int panel, int neighbour, int n) {

    TransformRule rule = rules[panel][neighbour];
    for (int i = 0; i < n; ++i) {
        T x = coord[i];
        T c = static_cast<T>(2) * x / (static_cast<T>(1) + x*x);
        T A1 = a1[i];
        T A2 = a2[i];

        o1[i] = rule.s11*A1 + rule.s12*A2 + c*(rule.s13*A1 + rule.s14*A2);
        o2[i] = rule.s21*A1 + rule.s22*A2 + c*(rule.s23*A1 + rule.s24*A2);
    }

}

struct TransformRule {
    int s11, s12, s13, s14; // r1 = s11*a1 + s12*a2 +c*(s13*a1 + s14*a2)
    int s21, s22, s23, s24; // r2 = s21*a1 + s22*a2 +c*(s23*a1 + s24*a2)
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