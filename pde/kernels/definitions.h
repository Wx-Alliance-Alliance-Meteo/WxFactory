#ifndef DEFINITIONS_H
#define DEFINTIONS_H

#include <cmath>
#include <complex>

// These declarations are temporary and must be done globally
const double p0 = 100000.;
const double Rd = 287.05;
const double cpd = 1005.46;
const double cvd = (cpd - Rd);
const double kappa = Rd / cpd;
const double heat_capacity_ratio = cpd / cvd;
const double inp0 = 1.0 / p0;
const double Rdinp0 = Rd * inp0;

// Define non-standard operations for complex numbers
inline std::complex<double> fabs(const std::complex<double> &z)
{
  return std::sqrt(z.real()*z.real() + z.imag()*z.imag());
}

inline std::complex<double> fmax(const std::complex<double> &z1, const std::complex<double> &z2)
{   
  if (std::abs(z1)>std::abs(z2)) return z1;
  else return z2;
}

inline std::complex<double> fmin(const std::complex<double> &z1, const std::complex<double> &z2)
{   
  if (std::abs(z1)<std::abs(z2)) return z1;
  else return z2;
}

#endif // DEFINITIONS_H