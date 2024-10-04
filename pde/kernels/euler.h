// Euler equations
extern "C" void pointwise_flux_euler(double *q, double *flux_x, double *flux_y, double *flux_z, const double sqrt_g, const double *metrics, const int stride, const int num_dim);
extern "C" void riemann_ausm_euler(double *Ql, double *Qr, double *fl, double *fr, const int nvars, const int direction, const int stride);
extern "C" void boundary_flux_euler(double *Q, double *flux, const int direction, const int stride);

