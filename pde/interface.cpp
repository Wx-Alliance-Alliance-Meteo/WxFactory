#include "definitions.hpp"

#include "kernels/boundary_flux.hpp"
#include "kernels/pointwise_flux.hpp"
#include "kernels/riemann_flux.hpp"



// -------------------------------------
// Pointwise fluxes
// -------------------------------------

template <typename num_t>
void pointwise_eulercartesian_2d(const num_t *q, num_t *flux_x1, num_t *flux_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts_tot)
{
  const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts_tot;
  for(int i=0; i<nb_elem_x2; i++)
  {
    for(int j=0; j<nb_elem_x1; j++)
    {
      for(int k=0; k<nb_solpts_tot; k++)
      {
        const int ind = i*nb_elem_x1 + j*nb_solpts_tot + k;
        pointwise_eulercartesian_2d_kernel(&q[ind],&flux_x1[ind],&flux_x2[ind],stride);
      }
    }
  }
}

// -------------------------------------
// Riemann fluxes
// -------------------------------------

template <typename num_t>
void riemann_eulercartesian_ausm_2d(const num_t *q_itf_x1, const num_t *q_itf_x2, num_t *f_itf_x1, num_t *f_itf_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts)
{
  const int nb_solpts_riem = 2 * nb_solpts;
  const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts_riem;

  for (int i=0; i<nb_elem_x2; i++)
  {
    for (int j=0; j<nb_elem_x1; j++)
    {
      // Solve along the x1-direction
      if (j + 1 < nb_elem_x1)
      {
        for(int k=0; k<nb_solpts; k++)
        {
          const int indl = i*nb_elem_x1 + j*nb_solpts_riem + k + nb_solpts;
          const int indr = i*nb_elem_x1 + (j+1)*nb_solpts_riem + k;

          riemann_eulercartesian_ausm_2d_kernel(&q_itf_x1[indl], &q_itf_x1[indr], &f_itf_x1[indl], &f_itf_x1[indr], 0, stride);
        }
      }

      if (i + 1 < nb_elem_x2)
      {
        for(int k=0; k<nb_solpts; k++)
        {
          const int indl = i*nb_elem_x1 + j*nb_solpts_riem + k + nb_solpts;
          const int indr = (i+1)*nb_elem_x1 + j*nb_solpts_riem + k;

          riemann_eulercartesian_ausm_2d_kernel(&q_itf_x1[indl], &q_itf_x1[indr], &f_itf_x1[indl], &f_itf_x1[indr], 1, stride);
        }
      }
    }
  }
}

// -------------------------------------
// Boundary fluxes
// -------------------------------------

template <typename num_t>
void boundary_eulercartesian_2d(const num_t *q_itf_x1, const num_t *q_itf_x2, num_t *f_itf_x1, num_t *f_itf_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts)
{
  int count = (nb_elem_x1 - 1) * nb_elem_x1 * nb_solpts * 2;
  const int stride = nb_elem_x1 * nb_elem_x2 * nb_solpts * 2;

  for(int i=0; i<nb_elem_x1; i++)
  {
    for(int j=0; j<nb_solpts; j++)
    { 
      const int indb = i * nb_solpts + j;
      const int indt = count + i * nb_solpts + j + nb_solpts;
      boundary_eulercartesian_2d_kernel(&q_itf_x2[indb], &f_itf_x2[indb], 1, stride);
      boundary_eulercartesian_2d_kernel(&q_itf_x2[indt], &f_itf_x2[indt], 1, stride);
    }
  }

  for(int i=0; i<nb_elem_x2; i++)
  {
    for(int j=0; j<nb_solpts; j++)
    { 
      const int indl = i * nb_elem_x1 * 2 * nb_solpts + j;
      const int indr = indl + nb_elem_x1 * 2 * nb_solpts + nb_solpts;
      boundary_eulercartesian_2d_kernel(&q_itf_x1[indl], &f_itf_x1[indl], 0, stride);
      boundary_eulercartesian_2d_kernel(&q_itf_x1[indr], &f_itf_x1[indr], 0, stride);
    }
  }
}

// -------------------------------------
// Template explicit instantiations
// -------------------------------------

extern "C"
{
  void pointwise_euler_cartesian_2d_double(const double *q, double *flux_x1, double *flux_x2, const int nb_elements_x1, const int nb_elements_x2, const int nb_solpts_total)
  {
    pointwise_eulercartesian_2d<double>(q,flux_x1,flux_x2,nb_elements_x1,nb_elements_x2,nb_solpts_total);
  }

  void riemann_eulercartesian_ausm_2d_double(const double *q_itf_x1, const double *q_itf_x2, double *f_itf_x1, double *f_itf_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts)
  {
    riemann_eulercartesian_ausm_2d<double>(q_itf_x1,q_itf_x2,f_itf_x1,f_itf_x2,nb_elem_x1,nb_elem_x2,nb_solpts);
  }

  void boundary_eulercartesian_2d_double(const double *q_itf_x1, const double *q_itf_x2, double *f_itf_x1, double *f_itf_x2, const int nb_elem_x1, const int nb_elem_x2, const int nb_solpts)
  {
    boundary_eulercartesian_2d<double>(q_itf_x1,q_itf_x2,f_itf_x1,f_itf_x2,nb_elem_x1,nb_elem_x2,nb_solpts);
  }

}

