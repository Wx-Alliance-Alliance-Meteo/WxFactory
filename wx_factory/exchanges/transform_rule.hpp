#ifndef TRANSFORM_RULE_H_
#define TRANSFORM_RULE_H_

struct TransformRule {
    int s11, s12, s13, s14;
    int s21, s22, s23, s24;
};

constexpr int N_PANELS = 6;
constexpr int N_FACES  = 4;

// host
extern const TransformRule kRulesHost[N_PANELS][N_FACES];

// device
#ifdef __CUDACC__
    __constant__ TransformRule kRulesDevice[N_PANELS][N_FACES];
    #define RULES kRulesDevice

#else
    #define RULES kRulesHost
#endif


void init_transform_rules_cuda(); // copy rules host to device

#endif // TRANSFORM_RULE_H_
