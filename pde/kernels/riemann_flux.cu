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
