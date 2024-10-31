extern "C" __global__ void pointwise_eulercartesian_2d(const double *q, double *flux_x, double *flux_z, const int stride)
{
  // These constants are here just temporarily
  double p0 = 100000.;
  double Rd = 287.05;
  double cpd = 1005.46;
  double cvd = (cpd - Rd);
  double kappa = Rd / cpd;
  double heat_capacity_ratio = cpd / cvd;
  double inp0 = 1.0 / p0;
  double Rdinp0 = Rd * inp0;
  const int idt = blockIdx.x * blockDim.x + threadIdx.x;

  if (idt < stride)
  {
    // Compute the strides to access state variables
    const int  idx_rho = idt + 0;
    const int idx_rhou = idt + stride;
    const int idx_rhow = idt + 2 * stride;
    const int idx_rhot = idt + 3 * stride;

    const double rho = q[idx_rho];
    const double rhou = q[idx_rhou];
    const double rhow = q[idx_rhow];
    const double rho_theta = q[idx_rhot];

    const double invrho = 1.0 / rho;

    const double u = rhou * invrho;
    const double w = rhow * invrho;

    const double p = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_theta));

    flux_x[idx_rho] = rhou;
    flux_x[idx_rhou] = rhou * u + p;
    flux_x[idx_rhow] = rhou * w;
    flux_x[idx_rhot] = rho_theta * u;

    flux_z[idx_rho] = rhow;
    flux_z[idx_rhou] = rhow * u;
    flux_z[idx_rhow] = rhow * w + p;
    flux_z[idx_rhot] = rho_theta * w;
  }
}