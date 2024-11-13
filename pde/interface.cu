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
__global__ void riemann_eulercartesian_ausm_2d(const num_t *q_itf, num_t *flux_itf, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts, const int direction, const int nmax_x, const int nmax_y, const int nmax_z)
{
 const int ix = blockIdx.x * blockDim.x + threadIdx.x;
 const int iy = blockIdx.y * blockDim.y + threadIdx.y;
 const int iz = blockIdx.z * blockDim.z + threadIdx.z;

  if (ix<nmax_x && iy < nmax_y && iz < nmax_z)
  {
    const int nb_solpts_riem = 2 * nb_solpts;
    const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts_riem;
    const int array_shape[4] = {4, nb_elem_x2, nb_elem_x1, nb_solpts_riem};

    if (direction==0)
    {
      // Initialize left-hand side parameters
      const int indl = get_c_index(0, ix, iy, nb_solpts + iz, array_shape);
      kernel_params<num_t,euler_state_2d> params_l(q_itf, flux_itf, nullptr, nullptr, indl, stride);

      // Initialize right-hand-size parameters
      const int indr = get_c_index(0, ix, iy+1, iz, array_shape);
      kernel_params<num_t,euler_state_2d> params_r(q_itf, flux_itf, nullptr, nullptr, indr, stride);
    
      // Call Riemann kernel on the horizontal direction
      riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, direction);
      
    }
    else if (direction==1)
    {
      // Initialize left-hand side parameters
      const int indl = get_c_index(0, ix, iy, nb_solpts + iz, array_shape);
      kernel_params<num_t,euler_state_2d> params_l(q_itf, nullptr, flux_itf, nullptr, indl, stride);

      // Initialize right-hand-size parameters
      const int indr = get_c_index(0, ix+1, iy, iz, array_shape);
      kernel_params<num_t,euler_state_2d> params_r(q_itf, nullptr, flux_itf, nullptr, indr, stride);
    
      // Call Riemann kernel on the horizontal direction
      riemann_eulercartesian_ausm_2d_kernel(params_l, params_r, direction);
    }
  }
}


template <typename num_t>
__global__ void boundary_eulercartesian_2d(const num_t *q_itf, num_t *flux_itf, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts, const int direction, const int nmax_x, const int nmax_y)
{
 const int ix = blockIdx.x * blockDim.x + threadIdx.x;
 const int iy = blockIdx.y * blockDim.y + threadIdx.y;
 
  if(ix < nmax_x && iy < nmax_y)
  {
    const int nb_solpts_riem = 2 * nb_solpts;
    const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts_riem;
    const int array_shape[4] = {4, nb_elem_x2, nb_elem_x1, nb_solpts_riem}; 

    if (direction==0) 
    {       
      // Left flux
      const int indl = get_c_index(0, ix, 0, iy, array_shape);
      kernel_params<num_t,euler_state_2d> params_l(q_itf, flux_itf, nullptr, nullptr, indl, stride);
      boundary_eulercartesian_2d_kernel(params_l, 0);

      // Right flux
      const int indr = get_c_index(0, ix, nb_elem_x1-1, nb_solpts + iy, array_shape);
      kernel_params<num_t,euler_state_2d> params_r(q_itf, flux_itf, nullptr, nullptr, indr, stride);
      boundary_eulercartesian_2d_kernel(params_r, 0);
    }

    if (direction==1) 
    {       
      // Bottom flux
      const int indb = get_c_index(0, 0, ix, iy, array_shape);
      kernel_params<num_t,euler_state_2d> params_b(q_itf, nullptr, flux_itf, nullptr, indb, stride);
      boundary_eulercartesian_2d_kernel(params_b, 1);

      // Top flux
      const int indt = get_c_index(0, nb_elem_x2-1, ix, nb_solpts + iy, array_shape);
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
  void pointwise_eulercartesian_2d_complex(const cuda::std::complex<double> *q, cuda::std::complex<double>  *flux_x1, cuda::std::complex<double>  *flux_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts_tot)
  {
    const int num_blocks = (nb_elem_x1 * nb_elem_x2 * nb_solpts_tot + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pointwise_eulercartesian_2d<cuda::std::complex<double>><<<num_blocks,BLOCK_SIZE>>>(q,flux_x1,flux_x2,nb_elem_x1,nb_elem_x2,nb_solpts_tot);
  }


  void riemann_eulercartesian_ausm_2d_double(const double *q_itf_x1, const double *q_itf_x2, double *f_itf_x1, double *f_itf_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts)
  {
    int width, height, depth;

    // Call Riemann solver on the horizontal direction
    width = nb_elem_x2; 
    height = nb_elem_x1 - 1;
    depth = nb_solpts;

    dim3 threads_per_block (8, 8, 8);
    dim3 num_blocks1 ((width  + threads_per_block.x - 1) / threads_per_block.x,
                     (height + threads_per_block.y - 1) / threads_per_block.y,
                     (depth  + threads_per_block.z - 1) / threads_per_block.z);

    riemann_eulercartesian_ausm_2d<double><<<num_blocks1,threads_per_block>>>(q_itf_x1,f_itf_x1,nb_elem_x1,nb_elem_x2,nb_solpts,0,width,height,depth);


    // Call Riemann solver on the vertical direction
    width = nb_elem_x2 - 1; 
    height = nb_elem_x1;
    depth = nb_solpts;

    dim3 num_blocks2 ((width  + threads_per_block.x - 1) / threads_per_block.x,
                      (height + threads_per_block.y - 1) / threads_per_block.y,
                      (depth  + threads_per_block.z - 1) / threads_per_block.z);

    riemann_eulercartesian_ausm_2d<double><<<num_blocks2,threads_per_block>>>(q_itf_x2,f_itf_x2,nb_elem_x1,nb_elem_x2,nb_solpts,1,width,height,depth);


    // Set the boundary fluxes on the horizontal direction
    dim3 threads_per_block2 (16, 16);

    width = nb_elem_x2;
    height = nb_solpts;

    dim3 num_blocks3 ((width  + threads_per_block2.x - 1) / threads_per_block2.x,
                      (height + threads_per_block2.y - 1) / threads_per_block2.y);

    boundary_eulercartesian_2d<double><<<num_blocks3,threads_per_block2>>>(q_itf_x1,f_itf_x1,nb_elem_x1,nb_elem_x2,nb_solpts,0,width,height);

    width = nb_elem_x1;
    height = nb_solpts;

    dim3 num_blocks4 ((width  + threads_per_block2.x - 1) / threads_per_block2.x,
                      (height + threads_per_block2.y - 1) / threads_per_block2.y);

    boundary_eulercartesian_2d<double><<<num_blocks4,threads_per_block2>>>(q_itf_x2,f_itf_x2,nb_elem_x1,nb_elem_x2,nb_solpts,1,width,height);
  }

  void riemann_eulercartesian_ausm_2d_complex(const complex_t *q_itf_x1, const complex_t *q_itf_x2, complex_t *f_itf_x1, complex_t *f_itf_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts)
  {
    int width, height, depth;

    // Call Riemann solver on the horizontal direction
    width = nb_elem_x2; 
    height = nb_elem_x1 - 1;
    depth = nb_solpts;

    dim3 threads_per_block (8, 8, 8);
    dim3 num_blocks1 ((width  + threads_per_block.x - 1) / threads_per_block.x,
                     (height + threads_per_block.y - 1) / threads_per_block.y,
                     (depth  + threads_per_block.z - 1) / threads_per_block.z);

    riemann_eulercartesian_ausm_2d<complex_t><<<num_blocks1,threads_per_block>>>(q_itf_x1,f_itf_x1,nb_elem_x1,nb_elem_x2,nb_solpts,0,width,height,depth);

    // Call Riemann solver on the vertical direction
    width = nb_elem_x2 - 1; 
    height = nb_elem_x1;
    depth = nb_solpts;

    dim3 num_blocks2 ((width  + threads_per_block.x - 1) / threads_per_block.x,
                      (height + threads_per_block.y - 1) / threads_per_block.y,
                      (depth  + threads_per_block.z - 1) / threads_per_block.z);

    riemann_eulercartesian_ausm_2d<complex_t><<<num_blocks2,threads_per_block>>>(q_itf_x2,f_itf_x2,nb_elem_x1,nb_elem_x2,nb_solpts,1,width,height,depth);


    // Set the boundary fluxes on the horizontal direction
    dim3 threads_per_block2 (16, 16);

    width = nb_elem_x2;
    height = nb_solpts;

    dim3 num_blocks3 ((width  + threads_per_block2.x - 1) / threads_per_block2.x,
                      (height + threads_per_block2.y - 1) / threads_per_block2.y);

    boundary_eulercartesian_2d<complex_t><<<num_blocks3,threads_per_block2>>>(q_itf_x1,f_itf_x1,nb_elem_x1,nb_elem_x2,nb_solpts,0,width,height);

    width = nb_elem_x1;
    height = nb_solpts;

    dim3 num_blocks4 ((width  + threads_per_block2.x - 1) / threads_per_block2.x,
                      (height + threads_per_block2.y - 1) / threads_per_block2.y);

    boundary_eulercartesian_2d<complex_t><<<num_blocks4,threads_per_block2>>>(q_itf_x2,f_itf_x2,nb_elem_x1,nb_elem_x2,nb_solpts,1,width,height);
  }

  
}

