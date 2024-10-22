#ifndef DEFINITIONS_H
#define DEFINTIONS_H

// These declarations are temporary and must be done globally
const double p0 = 100000.;
const double Rd = 287.05;
const double cpd = 1005.46;
const double cvd = (cpd - Rd);
const double kappa = Rd / cpd;
const double heat_capacity_ratio = cpd / cvd;
const double inp0 = 1.0 / p0;
const double Rdinp0 = Rd * inp0;

#endif // DEFINITIONS_H