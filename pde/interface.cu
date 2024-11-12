#include "definitions.hpp"
#include "interface.hpp"

#include "kernels/boundary_flux.hpp"
#include "kernels/pointwise_flux.hpp"
#include "kernels/riemann_flux.hpp"

template <typename num_t>
__global__ void pointwise_eulercartesian_2d(const num_t *q, num_t *flux_x1, num_t *flux_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts_tot)
{
  const int ind = threadIdx.x + blockIdx.x * blockDim.x;
  const int nmax = nb_elem_x1 * nb_elem_x2 * nb_solpts_tot;
  const int stride = nmax;
  if(ind < nmax)
  {
    // Store variables and pointers to compute the fluxes 
    kernel_params<num_t,euler_state_2d> params(q, flux_x1, flux_x2, nullptr, ind, stride);

    // Call the pointwise flux kernel
    pointwise_eulercartesian_2d_kernel(params);
  }
}

template <typename num_t>
__global__ void riemann_eulercartesian_ausm_2d(const num_t *q_itf, num_t *flux_itf, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts, const int direction, const int nmax)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < nmax)
  {
    const int nb_solpts_riem = 2 * nb_solpts;
    const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts_riem;
    const int array_shape[4] = {4, nb_elem_x2, nb_elem_x1, nb_solpts_riem};

    // Get the thread index per solution point
    const int t = idx / nb_solpts;

    int ixl, jxl, ixr, jxr;

    if(direction==0)
    {
      // Get the global element index in the grid
      const int elem_ind = t + t / (nb_elem_x1 - 1);

      // Get the x, y indices of the element in the grid on the right and left
      jxl = elem_ind % nb_elem_x1;
      ixl = elem_ind / nb_elem_x1;

      jxr = jxl + 1;
      ixr = ixl;

      // Get the solution point index
      const int k = t % nb_solpts;

      // Initialize left-hand side parameters
      const int indl = get_c_index(0, ixl, jxl, nb_solpts + k, array_shape);
      kernel_params<num_t,euler_state_2d> params_l(q_itf, flux_itf, nullptr, nullptr, indl, stride);

      // Initialize right-hand-size parameters
      const int indr = get_c_index(0, ixr, jxr + 1, k, array_shape);
      kernel_params<num_t,euler_state_2d> params_r(q_itf, flux_itf, nullptr, nullptr, indr, stride);
    
      // Call Riemann kernel on the horizontal direction
      riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, direction);
      
    }
    else if(direction==1)
    {
      // Get the global element index in the grid
      const int elem_ind = t;

      // Get the x, y indices of the element in the grid on the right and left
      jxl = elem_ind % nb_elem_x1;
      ixl = elem_ind / nb_elem_x1;

      jxr = jxl;
      ixr = ixl + nb_elem_x1;

      // Get the solution point index
      const int k = t % nb_solpts;

      // Initialize left-hand side parameters
      const int indl = get_c_index(0, ixl, jxl, nb_solpts + k, array_shape);
      kernel_params<num_t,euler_state_2d> params_l(q_itf, nullptr, flux_itf, nullptr, indl, stride);

      // Initialize right-hand-size parameters
      const int indr = get_c_index(0, ixr, jxr + 1, k, array_shape);
      kernel_params<num_t,euler_state_2d> params_r(q_itf, nullptr, flux_itf, nullptr, indr, stride);
    
      // Call Riemann kernel on the horizontal direction
      riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, direction);
    }
  }
}


template <typename num_t>
__global__ void boundary_eulercartesian_2d(const num_t *q_itf, num_t *flux_itf, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts, const int direction, const int nmax)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
 
  if(idx < nmax)
  {
    const int nb_solpts_riem = 2 * nb_solpts;
    const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts_riem;
    const int array_shape[4] = {4, nb_elem_x2, nb_elem_x1, nb_solpts_riem}; 

    // Get the element and solution point indices
    const int i = idx / nb_solpts;
    const int k = idx % nb_solpts;

    if(direction==0) 
    {       
      // Left flux
      const int indl = get_c_index(0, i * nb_elem_x1, 0, k, array_shape);
      kernel_params<num_t,euler_state_2d> params_l(q_itf, flux_itf, nullptr, nullptr, indl, stride);
      boundary_eulercartesian_2d_kernel(params_l, 0);

      // Top flux
      const int indr = get_c_index(0, (i+1) * nb_elem_x1 , i, k + nb_solpts, array_shape);
      kernel_params<num_t,euler_state_2d> params_r(q_itf, flux_itf, nullptr, nullptr, indr, stride);
      boundary_eulercartesian_2d_kernel(params_r, 0);
    }

    if(direction==1) 
    {       
      // Bottom flux
      const int indb = get_c_index(0, 0, i, k, array_shape);
      kernel_params<num_t,euler_state_2d> params_b(q_itf, nullptr, flux_itf, nullptr, indb, stride);
      boundary_eulercartesian_2d_kernel(params_b, 1);

      // Top flux
      const int indt = get_c_index(0, nb_elem_x2-1, i, k + nb_solpts, array_shape);
      kernel_params<num_t,euler_state_2d> params_t(q_itf, nullptr, flux_itf, nullptr, indt, stride);
      boundary_eulercartesian_2d_kernel(params_t, 1);
    }
  }
}


// Explicit instantiations for each argument type (float, double or complex)
extern "C"
{
  void pointwise_eulercartesian_2d_double(const double *q, double *flux_x1, double *flux_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts_tot)
  {
    const int num_blocks = (nb_elem_x1 * nb_elem_x2 * nb_solpts_tot + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pointwise_eulercartesian_2d<double><<<num_blocks,BLOCK_SIZE>>>(q,flux_x1,flux_x2,nb_elem_x1,nb_elem_x2,nb_solpts_tot);
  }

  void riemann_eulercartesian_ausm_2d_double(const double *q_itf_x1, const double *q_itf_x2, double *f_itf_x1, double *f_itf_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts, const int direction)
  {
    int nmax, num_blocks;

    // Call Riemann solver on the horizontal direction
    nmax = (nb_elem_x1 - 1) * nb_elem_x2 * nb_solpts;
    num_blocks = (nmax + BLOCK_SIZE - 1) / BLOCK_SIZE;
    riemann_eulercartesian_ausm_2d<double><<<num_blocks,BLOCK_SIZE>>>(q_itf_x1,f_itf_x1,nb_elem_x1,nb_elem_x2,nb_solpts,0,nmax);

    // Call Riemann solver on the vertical direction
    nmax = nb_elem_x1 * (nb_elem_x2 - 1) * nb_solpts;
    num_blocks = (nmax + BLOCK_SIZE - 1) / BLOCK_SIZE;
    riemann_eulercartesian_ausm_2d<double><<<num_blocks,BLOCK_SIZE>>>(q_itf_x2,f_itf_x2,nb_elem_x1,nb_elem_x2,nb_solpts,1,nmax);
  }
  
  void boundary_eulercartesian_2d_double(const double *q_itf_x1, const double *q_itf_x2, double *f_itf_x1, double *f_itf_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts)
  {
    int nmax, num_blocks;

    // Call Riemann solver on the horizontal direction
    nmax = nb_elem_x1 * nb_solpts;
    num_blocks = (nmax + BLOCK_SIZE - 1) / BLOCK_SIZE;
    boundary_eulercartesian_2d<double><<<num_blocks,BLOCK_SIZE>>>(q_itf_x1,f_itf_x1,nb_elem_x1,nb_elem_x2,nb_solpts,0,nmax);

    // Call Riemann solver on the vertical direction
    nmax = nb_elem_x2 * nb_solpts;
    num_blocks = (nmax + BLOCK_SIZE - 1) / BLOCK_SIZE;
    boundary_eulercartesian_2d<double><<<num_blocks,BLOCK_SIZE>>>(q_itf_x2,f_itf_x2,nb_elem_x1,nb_elem_x2,nb_solpts,1,nmax);
  }
}

