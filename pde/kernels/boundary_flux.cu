
extern "C" __global__ void boundary_eulercartesian_2d(const double *q, double *f, const int *indb, const int direction, const int stride, const int nmax)
{
  // These constants are here just temporarily
  double p0 = 100000.;
  double Rd = 287.05;
  double cpd = 1005.46;
  double cvd = (cpd - Rd);
  double kappa = Rd / cpd;
  double heat_capacity_ratio = cpd / cvd;
  double inp0 = 1.0 / p0;
  double Rdinp0 = Rd / p0;
  const int idt = blockIdx.x * blockDim.x + threadIdx.x;

  if (idt < nmax)
  {
    const double *qb = &q[indb[idt]];
    double *flux = &f[indb[idt]];

    // Compute variable stride
    const int idx_rho = 0;
    const int idx_rhou = stride;
    const int idx_rhow = 2 * stride;
    const int idx_rhot = 3 * stride;
    
    if (direction == 0)
    {
      const double rho_theta = qb[idx_rhot];
      flux[idx_rho] = 0.0;
      flux[idx_rhou] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
      flux[idx_rhow] = 0.0;
      flux[idx_rhot] = 0.0;
    }
    else if (direction == 1)
    {
      const double rho_theta = qb[idx_rhot];
      flux[idx_rho] = 0.0;
      flux[idx_rhou] = 0.0;
      flux[idx_rhow] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
      flux[idx_rhot] = 0.0;
    }
  }
}