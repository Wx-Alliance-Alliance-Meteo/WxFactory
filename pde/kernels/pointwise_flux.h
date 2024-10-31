template<typename num_t> 
void pointwise_eulercartesian_2d(const num_t *q, num_t *flux_x1, num_t *flux_x2, const int stride);

template<typename num_t> 
void pointwise_swcubedsphere_2d(const num_t *q, num_t *flux_x1, num_t *flux_x2, const double sqrt_g, const double *metrics, const int stride);
