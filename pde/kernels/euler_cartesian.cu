extern "C" __global__ void ausm_solver(double *Q,
                                       double *f,
                                       const int nvars,
                                       const int direction,
                                       const int *indlv,
                                       const int *indrv,
                                       const int stride,
                                       const int nmax)
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

  if (idt < nmax)
  {
    double rhol, rhor, rho_thetal, rho_thetar;
    double rhoul, rhour, rhowl, rhowr;
    double ul, ur, wl, wr, invrhol, invrhor;
    double pl, pr, al, ar, vnl, vnr, M, Ml, Mr, Mmax, Mmin;
    int idx_rho, idx_rhou, idx_rhow, idx_rhot;
    int indl, indr;
    double *Ql, *Qr, *fl, *fr;

    indl = indlv[idt];
    indr = indrv[idt];

    Ql = &Q[indl];
    Qr = &Q[indr];

    fl = &f[indl];
    fr = &f[indr];

    // Compute variable stride
    idx_rho = 0;
    idx_rhou = stride;
    idx_rhow = 2 * stride;
    idx_rhot = 3 * stride;

    rhol = Ql[idx_rho];
    rhoul = Ql[idx_rhou];
    rhowl = Ql[idx_rhow];
    rho_thetal = Ql[idx_rhot];

    rhor = Qr[idx_rho];
    rhour = Qr[idx_rhou];
    rhowr = Qr[idx_rhow];
    rho_thetar = Qr[idx_rhot];

    invrhol = 1.0 / rhol;
    ul = rhoul * invrhol;
    wl = rhowl * invrhol;

    invrhor = 1.0 / rhor;
    ur = rhour * invrhor;
    wr = rhowr * invrhor;

    pl = p0 * pow(rho_thetal * Rd * inp0, heat_capacity_ratio);
    pr = p0 * pow(rho_thetar * Rd * inp0, heat_capacity_ratio);

    al = sqrt(heat_capacity_ratio * pl * invrhol);
    ar = sqrt(heat_capacity_ratio * pr * invrhor);

    vnr = 0.0;
    vnl = 0.0;

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

    Ml = vnl / al + 1.0;
    Mr = vnr / ar - 1.0;

    M = 0.25 * (Ml * Ml - Mr * Mr);
    Mmax = max(0.0, M) * al;
    Mmin = min(0.0, M) * ar;

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

extern "C" __global__ void boundary_flux(double *Q, double *f, const int *indb, const int direction, const int stride, const int nmax)
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
    double *Qb, *fb, rho_theta;
    int idx_rho, idx_rhou, idx_rhow, idx_rhot;

    Qb = &Q[indb[idt]];
    fb = &f[indb[idt]];

    // Compute variable stride
    idx_rho = 0;
    idx_rhou = stride;
    idx_rhow = 2 * stride;
    idx_rhot = 3 * stride;

    rho_theta = Qb[idx_rhot];

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