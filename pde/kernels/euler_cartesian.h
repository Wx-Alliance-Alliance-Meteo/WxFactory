extern "C" void pointwise_euler_flux(double *Q, double *flux_x, double *flux_z, const int stride);
extern "C" void ausm_solver(double *Ql, double *Qr, double *fl, double *fr, const int nvars, const int direction, const int stride);
extern "C" void boundary_flux(double *Q, double *flux, const int direction, const int stride);
