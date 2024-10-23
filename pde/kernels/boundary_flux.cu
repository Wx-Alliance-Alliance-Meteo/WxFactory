
extern "C" __global__ void boundary_flux(double *q, double *f, const int *indb, const int direction, const int stride, const int nmax)
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
  int idt = blockIdx.x * blockDim.x + threadIdx.x;

  if (idt < nmax)
  {
    double *qb, *fb, rho_theta;
    int idx_rho, idx_rhou, idx_rhow, idx_rhot;

    qb = &q[indb[idt]];
    fb = &f[indb[idt]];

    // Compute variable stride
    idx_rho = 0;
    idx_rhou = stride;
    idx_rhow = 2 * stride;
    idx_rhot = 3 * stride;

    rho_theta = qb[idx_rhot];

    fb[idx_rho] = 0.0;
    if (direction == 0)
    {
      fb[idx_rhou] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
      fb[idx_rhow] = 0.0;
    }
    if (direction == 1)
    {
      fb[idx_rhou] = 0.0;
      fb[idx_rhow] = p0 * pow(rho_theta * Rd * inp0, heat_capacity_ratio);
    }
    fb[idx_rhot] = 0.0;
  }
}