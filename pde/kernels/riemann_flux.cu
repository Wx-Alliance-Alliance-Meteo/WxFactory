extern "C" __global__ void riemann_eulercartesian_ausm_2d(const double *q,
  double *f, const int nvars, const int direction, const int *indlv,
  const int *indrv, const int stride, const int nmax)
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

  if (idt < nmax)
  {
    const int indl = indlv[idt];
    const int indr = indrv[idt];

    const double *ql = &q[indl];
    const double *qr = &q[indr];

    double *fl = &f[indl];
    double *fr = &f[indr];

    // Compute variable stride
    const int idx_rho = 0;
    const int idx_rhou = stride;
    const int idx_rhow = 2 * stride;
    const int idx_rhot = 3 * stride;

    const double rhol = ql[idx_rho];
    const double rhoul = ql[idx_rhou];
    const double rhowl = ql[idx_rhow];
    const double rho_thetal = ql[idx_rhot];

    const double rhor = qr[idx_rho];
    const double rhour = qr[idx_rhou];
    const double rhowr = qr[idx_rhow];
    const double rho_thetar = qr[idx_rhot];

    const double invrhol = 1.0 / rhol;
    const double ul = rhoul * invrhol;
    const double wl = rhowl * invrhol;

    const double invrhor = 1.0 / rhor;
    const double ur = rhour * invrhor;
    const double wr = rhowr * invrhor;

    const double pl = p0 * pow(rho_thetal * Rd * inp0, heat_capacity_ratio);
    const double pr = p0 * pow(rho_thetar * Rd * inp0, heat_capacity_ratio);

    const double al = sqrt(heat_capacity_ratio * pl * invrhol);
    const double ar = sqrt(heat_capacity_ratio * pr * invrhor);

    double vnr = 0.0;
    double vnl = 0.0;

    // Fx faces
    if (direction == 0)
    {
      vnl = ul;
      vnr = ur;
    }

    if (direction == 1)
    {
      vnl = wl;
      vnr = wr;
    }

    const double Ml = vnl / al + 1.0;
    const double Mr = vnr / ar - 1.0;

    const double M = 0.25 * (Ml * Ml - Mr * Mr);
    const double Mmax = max(0.0, M) * al;
    const double Mmin = min(0.0, M) * ar;

    // Set the interface fluxes
    fl[idx_rho] = rhol * Mmax + rhor * Mmin;
    if (direction == 0)
    {
      fl[idx_rhou] = 0.5 * (Ml * pl - Mr * pr);
      fl[idx_rhow] = rhowl * Mmax + rhowr * Mmin;
    }

    if (direction == 1)
    {
      fl[idx_rhou] = rhoul * Mmax + rhour * Mmin;
      fl[idx_rhow] = 0.5 * (Ml * pl - Mr * pr);
    }
    fl[idx_rhot] = rho_thetal * Mmax + rho_thetar * Mmin;

    for (int i = 0; i < nvars; i++)
    {
      fr[i * stride] = fl[i * stride];
    }
  }
}
