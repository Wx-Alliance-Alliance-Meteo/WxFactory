#ifndef DEFINITIONS_H
#define DEFINITIONS_H

// Declarations
// DEVICE_SPACE const double p0 = 100000.0;
// DEVICE_SPACE const double Rd = 287.05;
// DEVICE_SPACE const double cpd                 = 1005.46;
// DEVICE_SPACE const double cvd                 = (cpd - Rd);
// DEVICE_SPACE const double kappa               = Rd / cpd;
// DEVICE_SPACE const double heat_capacity_ratio = cpd / cvd;
// DEVICE_SPACE const double inp0                = 1.0 / p0;
// DEVICE_SPACE const double Rdinp0              = Rd * inp0;
#define p0                  100000.0
#define Rd                  287.05
#define cpd                 1005.46
#define cvd                 (cpd - Rd)
#define kappa               (Rd / cpd)
#define heat_capacity_ratio (cpd / cvd)
#define inp0                (1.0 / p0)
#define Rdinp0              (Rd * inp0)

#endif // DEFINITIONS_H
