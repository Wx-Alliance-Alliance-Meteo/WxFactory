#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <cmath>
#include <complex>

#ifdef __CUDACC__
  #define DEVICE_SPACE __device__
#else 
  #define DEVICE_SPACE
  using namespace std;
#endif

// These declarations are temporary and must be done globally
DEVICE_SPACE const double p0 = 100000.;
DEVICE_SPACE const double Rd = 287.05;
DEVICE_SPACE const double cpd = 1005.46;
DEVICE_SPACE const double cvd = (cpd - Rd);
DEVICE_SPACE const double kappa = Rd / cpd;
DEVICE_SPACE const double heat_capacity_ratio = cpd / cvd;
DEVICE_SPACE const double inp0 = 1.0 / p0;
DEVICE_SPACE const double Rdinp0 = Rd * inp0;

const int BLOCK_SIZE = 256;

#endif // DEFINITIONS_H