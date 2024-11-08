#include "definitions.hpp"

#include "kernels/boundary_flux.hpp"
#include "kernels/pointwise_flux.hpp"
#include "kernels/riemann_flux.hpp"

template <typename num_t>
__global__ void pointwise_eulercartesian_2d(const num_t *q, num_t *flux_x1, num_t *flux_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts_tot)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int nmax = nb_elem_x1 * nb_elem_x2 * nb_solpts_tot;
  if(idx < nmax)
  {
    // Kernel call
    pointwise_eulercartesian_2d_kernel(&q[idx],&flux_x1[idx],&flux_x2[idx],nmax);
  }
}

template <typename num_t>
__global__ void riemann_eulercartesian_ausm_2d(const num_t *q_itf, num_t *f_itf, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts, const int direction, const int nmax)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < nmax)
  {
    const int nb_solpts_riem = 2 * nb_solpts;
    const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts_riem;

    // Get the thread index per solution point
    const int t = idx / nb_solpts;

    int ixl, jxl, ixr, jxr;

    if(direction==0)
    {
      // Get the global element index in the grid
      const int elem_ind = t + t / (nb_elem_x1 - 1);

      // Get the x, y indices of the element in the grid on the right and left
      ixl = elem_ind % nb_elem_x1;
      jxl = elem_ind / nb_elem_x1;

      ixr = ixl + 1;
      jxr = jxl;
      
    }
    else if(direction==1)
    {
      // Get the global element index in the grid
      const int elem_ind = t;

      // Get the x, y indices of the element in the grid on the right and left
      ixl = elem_ind % nb_elem_x1;
      jxl = elem_ind / nb_elem_x1;

      ixr = ixl;
      jxr = jxl + nb_elem_x1;
    }

    // Compute the locations in the arrays
    const int solpt_ind = t % nb_solpts;
    const int indl = ixl * nb_elem_x1 + jxl * nb_solpts_riem + nb_solpts + solpt_ind;
    const int indr = ixr * nb_elem_x1 + jxr * nb_solpts_riem + solpt_ind;

    riemann_eulercartesian_ausm_2d_kernel(&q_itf[indl], &q_itf[indr], &f_itf[indl], &f_itf[indr], direction, stride);
  }
}


template <typename num_t>
__global__ void boundary_eulercartesian_2d(const num_t *q_itf_x1, const num_t *q_itf_x2, num_t *f_itf_x1, num_t *f_itf_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts, const int nmax)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(idx < nmax)
  {
    boundary_eulercartesian_2d_kernel(&q_itf_x2[indb], &f_itf_x2[indb], 1, stride);
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
}

