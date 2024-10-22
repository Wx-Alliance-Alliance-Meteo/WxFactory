template<typename T> 
void pointwise_eulercartesian_2d(T *q, T *flux_x1, T *flux_x2, const int stride);

template<typename T> 
void pointwise_swcubedsphere_2d(T *q, T *flux_x1, T *flux_x2, const double sqrt_g, const double *metrics, const int stride);
