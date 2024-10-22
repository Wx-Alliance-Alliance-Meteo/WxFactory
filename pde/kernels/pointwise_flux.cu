extern "C" __global__ void euler_flux(const double *Q, double *flux_x, double *flux_z, const int stride)
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
  int idt = blockIdx.x * blockDim.x + threadIdx.x;

  if (idt < stride)
  {
    int idx_rho, idx_rhou, idx_rhow, idx_rhot;
    double rho, invrho, rhou, rhow, rho_theta;
    double u, w, p;

    // Compute the strides to access state variables
    idx_rho = idt + 0;
    idx_rhou = idt + stride;
    idx_rhow = idt + 2 * stride;
    idx_rhot = idt + 3 * stride;

    rho = Q[idx_rho];
    rhou = Q[idx_rhou];
    rhow = Q[idx_rhow];
    rho_theta = Q[idx_rhot];

    invrho = 1.0 / rho;

    u = rhou * invrho;
    w = rhow * invrho;

    p = p0 * exp(heat_capacity_ratio * log(Rdinp0 * rho_theta));

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