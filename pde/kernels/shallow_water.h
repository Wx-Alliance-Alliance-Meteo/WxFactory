// Shallow water equations
extern "C" void pointwise_flux_sw(double *q, double *flux_x, double *flux_y, double *flux_z, const double sqrt_g, const double *metrics, const int stride, const int num_dim);
