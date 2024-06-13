import cupy as cp
from mpi4py import MPI

from cupy.cuda.nvtx import RangePush, RangePop

from .cuda_module import CudaModule, DimSpec, Dim, Requires, TemplateSpec, cuda_kernel

from common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio
from init.dcmip import dcmip_schar_damping

# For type hints
from typing import TypeVar
from numpy.typing import NDArray
from common.cuda_parallel import CudaDistributedWorld
from geometry import CubedSphere, DFROperators, Metric3DTopo


class RHSEuler(metaclass=CudaModule,
               path="rhs_euler.cu",
               defines=(("heat_capacity_ratio", heat_capacity_ratio),
                        ("idx_rho", idx_rho),
                        ("idx_rho_u1", idx_rho_u1),
                        ("idx_rho_u2", idx_rho_u2),
                        ("idx_rho_w", idx_rho_w))):

   @cuda_kernel[Number := TypeVar("Number", bound=cp.generic), Requires.DoubleOrComplex(Number),
                Real := TypeVar("Real", bound=cp.generic), Requires.ModulusOf(Real, Number)] \
   (DimSpec.groupby_first_x(lambda s: Dim(s[3], s[1], s[2] - 1)), TemplateSpec.array_dtype(0, 6))
   def compute_flux_i(
      flux_x1_itf_i: NDArray[Number], wflux_adv_x1_itf_i: NDArray[Number], wflux_pres_x1_itf_i: NDArray[Number],
      variables_itf_i: NDArray[Number], pressure_itf_i: NDArray[Number], u1_itf_i: NDArray[Number],
      sqrtG_itf_i: NDArray[Real], H_contra_1x_itf_i: NDArray[Real],
      nh: int, nk_nh: int, nk_nv: int, ne: int, advection_only: bool): ...

   @cuda_kernel[Number := TypeVar("Number", bound=cp.generic), Requires.DoubleOrComplex(Number),
                Real := TypeVar("Real", bound=cp.generic), Requires.ModulusOf(Real, Number)] \
   (DimSpec.groupby_first_x(lambda s: Dim(s[4], s[1], s[2] - 1)), TemplateSpec.array_dtype(0, 6))
   def compute_flux_j(
      flux_x2_itf_j: NDArray[Number], wflux_adv_x2_itf_j: NDArray[Number], wflux_pres_x2_itf_j: NDArray[Number],
      variables_itf_j: NDArray[Number], pressure_itf_j: NDArray[Number], u2_itf_j: NDArray[Number],
      sqrtG_itf_j: NDArray[Real], H_contra_2x_itf_j: NDArray[Real],
      nh: int, nk_nh: int, nk_nv: int, ne: int, advection_only: bool): ...

   @cuda_kernel[Number := TypeVar("Number", bound=cp.generic), Requires.DoubleOrComplex(Number),
                Real := TypeVar("Real", bound=cp.generic), Requires.ModulusOf(Real, Number)] \
   (DimSpec.groupby_first_x(lambda s: Dim(s[4], s[1], s[2] - 1)), TemplateSpec.array_dtype(0, 6))
   def compute_flux_k(
      flux_x3_itf_k: NDArray[Number], wflux_adv_x3_itf_k: NDArray[Number], wflux_pres_x3_itf_k: NDArray[Number],
      variables_itf_k: NDArray[Number], pressure_itf_k: NDArray[Number], w_itf_k: NDArray[Number],
      sqrtG_itf_k: NDArray[Real], H_contra_3x_itf_k: NDArray[Real],
      nv: int, nk_nh: int, ne: int, advection_only: bool): ...


@cp.fuse(kernel_name='forcing_hori')
def forcing_hori(f_, i_, r_, u1_, u2_, w_, p_, c01_, c02_, c03_, c11_, c12_, c13_, c22_, c23_, c33_, \
           h11_, h12_, h13_, h22_, h23_, h33_):
   #return  \
   f_[:] =  \
      2. * r_ * (c01_ * u1_ + c02_ * u2_ + c03_ * w_) \
         + c11_ * (r_ * u1_ * u1_ + h11_ * p_) \
         + 2. * c12_ * (r_ * u1_ * u2_ + h12_ * p_) \
         + 2. * c13_ * (r_ * u1_ * w_  + h13_ * p_) \
         +      c22_ * (r_ * u2_ * u2_ + h22_ * p_) \
         + 2. * c23_ * (r_ * u2_ * w_  + h23_ * p_) \
         +      c33_ * (r_ * w_  * w_  + h33_ * p_)

@cp.fuse(kernel_name='rho_w')
def forcing_vert(f_, i_, r_, u1_, u2_, w_, p_, c01_, c02_, c03_, c11_, c12_, c13_, c22_, c23_, c33_, \
          h11_, h12_, h13_, h22_, h23_, h33_, inv_dzdeta_, g_, inv_g_, filter_k_):
   f_[:] =  \
      2. * r_ * (c01_ * u1_ + c02_ * u2_ + c03_ * w_) \
         + c11_ * (r_ * u1_ * u1_ + h11_ * p_) \
         + 2. * c12_ * (r_ * u1_ * u2_ + h12_ * p_) \
         + 2. * c13_ * (r_ * u1_ * w_  + h13_ * p_) \
         +      c22_ * (r_ * u2_ * u2_ + h22_ * p_) \
         + 2. * c23_ * (r_ * u2_ * w_  + h23_ * p_) \
         +      c33_ * (r_ * w_  * w_  + h33_ * p_) \
         + inv_dzdeta_ * g_ * inv_g_ * filter_k_

@cp.fuse(kernel_name='rhs_assemble')
def rhs_assemble(df1_dx1_, df2_dx2_, df3_dx3_, inv_g_, f_):
   return -inv_g_ * (df1_dx1_ + df2_dx2_ + df3_dx3_) - f_

@cp.fuse(kernel_name='compute_pressure')
def compute_pressure(p0, cpd, cvd, Rd, theta):
   return p0 * cp.exp((cpd / cvd) * cp.log((Rd / p0) * theta))

@cp.fuse(kernel_name='initial_flux')
def initial_flux(rho_u, rho, q, sqrt_g):
   return sqrt_g * (rho_u / rho) * q

@cp.fuse(kernel_name='flux_mid')
def flux_mid(f1, f2, f3, f01, f02, f03, h11, h12, h13, sqrt_g, p):
   f1[:] = f01 + sqrt_g * h11 * p
   f2[:] = f02 + sqrt_g * h12 * p
   f3[:] = f03 + sqrt_g * h13 * p


def to_new_shape(q, nbsolpts):
   # Get correct axis sizes
   n_eq, n_ks, n_is, n_js = q.shape
   n_k = n_ks // nbsolpts
   n_i = n_is // nbsolpts
   n_j = n_js // nbsolpts

   q_reshaped   = q.reshape((n_eq, n_k, nbsolpts, n_i, nbsolpts, n_j, nbsolpts)) # Split axes
   q_transposed = q_reshaped.transpose((0, 1, 3, 5, 2, 4, 6))                    # Reorganize data
   q_new = q_transposed.reshape(n_eq, n_k * n_i * n_j, nbsolpts * nbsolpts * nbsolpts) # Merge axes

   return q_new



def rhs_euler_cuda(Q: NDArray[cp.float64],
                   geom: CubedSphere,
                   mtrx: DFROperators,
                   metric: Metric3DTopo,
                   ptopo: CudaDistributedWorld,
                   nbsolpts: int,
                   nb_elements_hori: int,
                   nb_elements_vert: int,
                   case_number: int) -> NDArray[cp.float64]:

   RangePush(f'rhs_euler_cuda')

   type_vec = Q.dtype
   nb_equations = Q.shape[0]
   nb_pts_hori = nb_elements_hori * nbsolpts
   nb_vertical_levels = nb_elements_vert * nbsolpts

   forcing = cp.zeros_like(Q, dtype=type_vec)

   variables_itf_i = cp.ones( (nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)
   flux_x1_itf_i   = cp.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, nb_pts_hori, 2), dtype=type_vec)

   variables_itf_j = cp.ones( (nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)
   flux_x2_itf_j   = cp.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)

   variables_itf_k = cp.ones( (nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)
   flux_x3_itf_k   = cp.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)

   wflux_adv_x1_itf_i  = cp.zeros_like(flux_x1_itf_i[0])
   wflux_pres_x1_itf_i = cp.zeros_like(flux_x1_itf_i[0])
   wflux_adv_x2_itf_j  = cp.zeros_like(flux_x2_itf_j[0])
   wflux_pres_x2_itf_j = cp.zeros_like(flux_x2_itf_j[0])
   wflux_adv_x3_itf_k  = cp.zeros_like(flux_x3_itf_k[0])
   wflux_pres_x3_itf_k = cp.zeros_like(flux_x3_itf_k[0])

   advection_only = case_number < 13

   ####################################################
   RangePush(f'Extrapolate')

   variables_itf_i[:, :, 1:-1, :, :] = mtrx.extrapolate_i(Q, geom).transpose((0, 1, 3, 4, 2))
   variables_itf_j[:, :, 1:-1, :, :] = mtrx.extrapolate_j(Q, geom)


   logrho      = cp.log(Q[idx_rho])
   logrhotheta = cp.log(Q[idx_rho_theta])

   variables_itf_i[idx_rho, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_i(logrho, geom)).transpose((0, 2, 3, 1))
   variables_itf_j[idx_rho, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_j(logrho, geom))

   variables_itf_i[idx_rho_theta, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_i(logrhotheta, geom)).transpose((0, 2, 3, 1))
   variables_itf_j[idx_rho_theta, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_j(logrhotheta, geom))

   RangePop()

   ############################
   # New arrangement
   #q_new = to_new_shape(Q, nbsolpts)
   #RangePush(f'Extrapolate_new')
   #
   #RangePop()


   RangePush('Start comm')
   all_request = ptopo.xchange_Euler_interfaces(geom, variables_itf_i, variables_itf_j, blocking=False)
   RangePop()

   RangePush('Vertical flux')

   rho = Q[idx_rho]
   u1  = Q[idx_rho_u1] / rho
   u2  = Q[idx_rho_u2] / rho
   w   = Q[idx_rho_w]  / rho

   flux_x1 = initial_flux(Q[idx_rho_u1], Q[idx_rho], Q, metric.sqrtG)
   flux_x2 = initial_flux(Q[idx_rho_u2], Q[idx_rho], Q, metric.sqrtG)
   flux_x3 = initial_flux(Q[idx_rho_w], Q[idx_rho], Q, metric.sqrtG)
   #f = metric.sqrtG * u1 * Q
   #diff = f - flux_x1
   #diff_norm = cp.linalg.norm(diff) / cp.linalg.norm(f)
   #if diff_norm > 1e-10:
   #   print(f'rel diff {diff_norm:.3e}')
   #   raise ValueError

   wflux_adv_x1 = flux_x1[idx_rho_w].copy()
   wflux_adv_x2 = flux_x2[idx_rho_w].copy()
   wflux_adv_x3 = flux_x3[idx_rho_w].copy()

   #diff = flux_x1[idx_rho_w] - wflux_adv_x1
   #diff_norm = cp.linalg.norm(diff) / cp.linalg.norm(wflux_adv_x1)
   #if diff_norm > 1e-10:
   #   print(f'rel diff {diff_norm:.3e}')
   #   raise ValueError

   pressure = compute_pressure(p0, cpd, cvd, Rd, Q[idx_rho_theta])
   #p = p0 * cp.exp((cpd / cvd) * cp.log((Rd / p0) * Q[idx_rho_theta]))

   #diff = p - pressure
   #diff_norm = cp.linalg.norm(diff) / cp.linalg.norm(p)
   #if diff_norm > 1e-10:
   #   print(f'rel diff {diff_norm:.3e}')
   #   raise ValueError


   #flux_x1[idx_rho_u1] += metric.sqrtG * metric.H_contra_11 * pressure
   #flux_x1[idx_rho_u2] += metric.sqrtG * metric.H_contra_12 * pressure
   #flux_x1[idx_rho_w]  += metric.sqrtG * metric.H_contra_13 * pressure
   #f[idx_rho_u1] += metric.sqrtG * metric.H_contra_11 * pressure
   #f[idx_rho_u2] += metric.sqrtG * metric.H_contra_12 * pressure
   #f[idx_rho_w]  += metric.sqrtG * metric.H_contra_13 * pressure
   flux_mid(flux_x1[idx_rho_u1], flux_x1[idx_rho_u2], flux_x1[idx_rho_w], \
            flux_x1[idx_rho_u1], flux_x1[idx_rho_u2], flux_x1[idx_rho_w], \
            metric.H_contra_11, metric.H_contra_12, metric.H_contra_13, \
            metric.sqrtG, pressure)

   #diff = f - flux_x1
   #diff_norm = cp.linalg.norm(diff) / cp.linalg.norm(f)
   #if diff_norm > 1e-10:
   #   print(f'rel diff {diff_norm:.3e}')
   #   raise ValueError

   wflux_pres_x1 = (metric.sqrtG * metric.H_contra_13).astype(type_vec)

   #flux_x2[idx_rho_u1] += metric.sqrtG * metric.H_contra_21 * pressure
   #flux_x2[idx_rho_u2] += metric.sqrtG * metric.H_contra_22 * pressure
   #flux_x2[idx_rho_w]  += metric.sqrtG * metric.H_contra_23 * pressure
   flux_mid(flux_x2[idx_rho_u1], flux_x2[idx_rho_u2], flux_x2[idx_rho_w], \
            flux_x2[idx_rho_u1], flux_x2[idx_rho_u2], flux_x2[idx_rho_w], \
            metric.H_contra_21, metric.H_contra_22, metric.H_contra_23, \
            metric.sqrtG, pressure)

   wflux_pres_x2 = (metric.sqrtG * metric.H_contra_23).astype(type_vec)

   #flux_x3[idx_rho_u1] += metric.sqrtG * metric.H_contra_31 * pressure
   #flux_x3[idx_rho_u2] += metric.sqrtG * metric.H_contra_32 * pressure
   #flux_x3[idx_rho_w]  += metric.sqrtG * metric.H_contra_33 * pressure
   flux_mid(flux_x3[idx_rho_u1], flux_x3[idx_rho_u2], flux_x3[idx_rho_w], \
            flux_x3[idx_rho_u1], flux_x3[idx_rho_u2], flux_x3[idx_rho_w], \
            metric.H_contra_31, metric.H_contra_32, metric.H_contra_33, \
            metric.sqrtG, pressure)

   wflux_pres_x3 = (metric.sqrtG * metric.H_contra_33).astype(type_vec)

   variables_itf_k[:, :, 1:-1, :, :] = mtrx.extrapolate_k(Q, geom).transpose((0, 3, 1, 2, 4))

   variables_itf_k[idx_rho, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_k(logrho, geom).transpose((2, 0, 1, 3)))
   variables_itf_k[idx_rho_theta, :, 1:-1, :, :] = cp.exp(mtrx.extrapolate_k(logrhotheta, geom).transpose((2, 0, 1, 3)))

   variables_itf_k[:, :,  0, 1, :] = variables_itf_k[:, :,  1, 0, :]
   variables_itf_k[:, :,  0, 0, :] = variables_itf_k[:, :,  0, 1, :]
   variables_itf_k[:, :, -1, 0, :] = variables_itf_k[:, :, -2, 1, :]
   variables_itf_k[:, :, -1, 1, :] = variables_itf_k[:, :, -1, 0, :]

   #pressure_itf_k = p0 * cp.exp((cpd / cvd) * cp.log(variables_itf_k[idx_rho_theta] * (Rd / p0)))
   pressure_itf_k = compute_pressure(p0, cpd, cvd, Rd, variables_itf_k[idx_rho_theta])

   w_itf_k = variables_itf_k[idx_rho_w] / variables_itf_k[idx_rho]

   w_itf_k[:,  0, 0, :] = 0.
   w_itf_k[:,  0, 1, :] = -w_itf_k[:,  1, 0, :]
   w_itf_k[:, -1, 1, :] = 0.
   w_itf_k[:, -1, 0, :] = -w_itf_k[:, -2, 1, :]

   RHSEuler.compute_flux_k(flux_x3_itf_k, wflux_adv_x3_itf_k, wflux_pres_x3_itf_k,
                           variables_itf_k, pressure_itf_k, w_itf_k,
                           metric.sqrtG_itf_k, metric.H_contra_itf_k[2],
                           nb_elements_vert, nb_pts_hori, nb_equations, advection_only)


   RangePop()

   RangePush('Wait comm')
   all_request.wait()
   RangePop()

   RangePush('Horizontal flux')

   u1_itf_i = variables_itf_i[idx_rho_u1] / variables_itf_i[idx_rho]
   u2_itf_j = variables_itf_j[idx_rho_u2] / variables_itf_j[idx_rho]

   pressure_itf_i = p0 * cp.exp((cpd / cvd) * cp.log(variables_itf_i[idx_rho_theta] * (Rd / p0)))
   pressure_itf_j = p0 * cp.exp((cpd / cvd) * cp.log(variables_itf_j[idx_rho_theta] * (Rd / p0)))

   RangePush('Flux kernels - 2')
   RHSEuler.compute_flux_i(flux_x1_itf_i, wflux_adv_x1_itf_i, wflux_pres_x1_itf_i,
                           variables_itf_i, pressure_itf_i, u1_itf_i,
                           metric.sqrtG_itf_i, metric.H_contra_itf_i[0],
                           nb_elements_hori, nb_pts_hori, nb_vertical_levels, nb_equations, advection_only)

   RHSEuler.compute_flux_j(flux_x2_itf_j, wflux_adv_x2_itf_j, wflux_pres_x2_itf_j,
                           variables_itf_j, pressure_itf_j, u2_itf_j,
                           metric.sqrtG_itf_j, metric.H_contra_itf_j[1],
                           nb_elements_hori, nb_pts_hori, nb_vertical_levels, nb_equations, advection_only)
   RangePop()

   flux_x1_bdy = flux_x1_itf_i.transpose((0, 1, 3, 2, 4))[:, :, :, 1:-1, :].copy()
   RangePush('Comma i (1)')
   df1_dx1 = mtrx.comma_i(flux_x1, flux_x1_bdy, geom)
   RangePop()
   flux_x2_bdy = flux_x2_itf_j[:, :, 1:-1, :, :].copy()
   RangePush('Comma j (1)')
   df2_dx2 = mtrx.comma_j(flux_x2, flux_x2_bdy, geom)
   RangePop()
   flux_x3_bdy = flux_x3_itf_k[:, :, 1:-1, :, :].transpose(0, 2, 3, 1, 4).copy()
   RangePush('Comma k (1)')
   df3_dx3 = mtrx.comma_k(flux_x3, flux_x3_bdy, geom)
   RangePop()

   RangePush('Pressure')
   logp_int = cp.log(pressure)

   #pressure_bdy_i = pressure_itf_i[:, 1:-1, :, :].transpose((0, 3, 1, 2)).copy()
   #pressure_bdy_j = pressure_itf_j[:, 1:-1, :, :].copy()
   #pressure_bdy_k = pressure_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()

   logp_bdy_i = cp.log(pressure_itf_i[:, 1:-1, :, :].transpose((0, 3, 1, 2)))
   logp_bdy_j = cp.log(pressure_itf_j[:, 1:-1, :, :])
   logp_bdy_k = cp.log(pressure_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)))
   RangePop()

   RangePush('Adv')
   wflux_adv_x1_bdy_i  = wflux_adv_x1_itf_i.transpose((0, 2, 1, 3))[:, :, 1:-1, :].copy()
   wflux_pres_x1_bdy_i = wflux_pres_x1_itf_i.transpose((0, 2, 1, 3))[:, :, 1:-1, :].copy()

   wflux_adv_x2_bdy_j  =  wflux_adv_x2_itf_j[:, 1:-1, :, :].copy()
   wflux_pres_x2_bdy_j = wflux_pres_x2_itf_j[:, 1:-1, :, :].copy()

   wflux_adv_x3_bdy_k  =  wflux_adv_x3_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()
   wflux_pres_x3_bdy_k = wflux_pres_x3_itf_k[:, 1:-1, :, :].transpose((1, 2, 0, 3)).copy()
   RangePop()

   RangePush('Comma ijk (2)')
   #w_df1_dx1_adv   = comma_i(wflux_adv_x1,  wflux_adv_x1_bdy_i, nbsolpts, mtrx.diff_solpt_tr, mtrx.correction_tr)
   w_df1_dx1_adv   = mtrx.comma_i(wflux_adv_x1,  wflux_adv_x1_bdy_i,  geom)

   #diff = a - w_df1_dx1_adv
   #diff_norm = cp.linalg.norm(diff) / cp.linalg.norm(a)
   #if diff_norm > 1e-10:
   #   print(f'rel diff {diff_norm:.3e}')
   #   raise ValueError

   w_df1_dx1_presa = mtrx.comma_i(wflux_pres_x1, wflux_pres_x1_bdy_i, geom) * pressure
   w_df1_dx1_presb = mtrx.comma_i(logp_int,      logp_bdy_i,          geom) * pressure * wflux_pres_x1
   w_df1_dx1       = w_df1_dx1_adv + w_df1_dx1_presa + w_df1_dx1_presb

   w_df2_dx2_adv   = mtrx.comma_j(wflux_adv_x2,  wflux_adv_x2_bdy_j,  geom)
   w_df2_dx2_presa = mtrx.comma_j(wflux_pres_x2, wflux_pres_x2_bdy_j, geom) * pressure
   w_df2_dx2_presb = mtrx.comma_j(logp_int,      logp_bdy_j,          geom) * pressure * wflux_pres_x2
   w_df2_dx2       = w_df2_dx2_adv + w_df2_dx2_presa + w_df2_dx2_presb

   w_df3_dx3_adv   = mtrx.comma_k(wflux_adv_x3,  wflux_adv_x3_bdy_k,  geom)
   w_df3_dx3_presa = mtrx.comma_k(wflux_pres_x3, wflux_pres_x3_bdy_k, geom) * pressure
   w_df3_dx3_presb = mtrx.comma_k(logp_int,      logp_bdy_k,          geom) * pressure * wflux_pres_x3
   w_df3_dx3       = w_df3_dx3_adv + w_df3_dx3_presa + w_df3_dx3_presb
   RangePop()

   RangePop()

   RangePush('Forcing')

   #forcing[idx_rho] = 0.

   RangePush('rho_u1')
   #forcing[idx_rho_u1] = \
   forcing_hori(forcing[idx_rho_u1], idx_rho, rho, u1, u2, w, pressure, \
          metric.christoffel_1_01, metric.christoffel_1_02, metric.christoffel_1_03, \
          metric.christoffel_1_11, metric.christoffel_1_12, metric.christoffel_1_13, \
          metric.christoffel_1_22, metric.christoffel_1_23, metric.christoffel_1_33, \
          metric.H_contra_11, metric.H_contra_12, metric.H_contra_13, \
          metric.H_contra_22, metric.H_contra_23, metric.H_contra_33)
   RangePop()

   ### Verify the result
   #a = \
   #        2. * rho * (metric.christoffel_1_01 * u1 + metric.christoffel_1_02 * u2 + metric.christoffel_1_03 * w) \
   #      +      metric.christoffel_1_11 * (rho * u1 * u1 + metric.H_contra_11 * pressure) \
   #      + 2. * metric.christoffel_1_12 * (rho * u1 * u2 + metric.H_contra_12 * pressure) \
   #      + 2. * metric.christoffel_1_13 * (rho * u1 * w  + metric.H_contra_13 * pressure) \
   #      +      metric.christoffel_1_22 * (rho * u2 * u2 + metric.H_contra_22 * pressure) \
   #      + 2. * metric.christoffel_1_23 * (rho * u2 * w  + metric.H_contra_23 * pressure) \
   #      +      metric.christoffel_1_33 * (rho * w  * w  + metric.H_contra_33 * pressure)

   #diff = a - forcing[idx_rho_u1]
   #diff_norm = cp.linalg.norm(diff) / cp.linalg.norm(a)
   #if diff_norm > 1e-10:
   #   print(f'rel diff {diff_norm:.3e}')
   #   raise ValueError

   RangePush('rho_u2')
   #forcing[idx_rho_u2] = \
   forcing_hori(forcing[idx_rho_u2], idx_rho_u2, rho, u1, u2, w, pressure, \
         metric.christoffel_2_01, metric.christoffel_2_02, metric.christoffel_2_03, \
         metric.christoffel_2_11, metric.christoffel_2_12, metric.christoffel_2_13, \
         metric.christoffel_2_22, metric.christoffel_2_23, metric.christoffel_2_33, \
         metric.H_contra_11, metric.H_contra_12, metric.H_contra_13, \
         metric.H_contra_22, metric.H_contra_23, metric.H_contra_33)
   RangePop()

   #b = \
   #        2. * rho * (metric.christoffel_2_01 * u1 + metric.christoffel_2_02 * u2 + metric.christoffel_2_03 * w) \
   #      +      metric.christoffel_2_11 * (rho * u1 * u1 + metric.H_contra_11 * pressure) \
   #      + 2. * metric.christoffel_2_12 * (rho * u1 * u2 + metric.H_contra_12 * pressure) \
   #      + 2. * metric.christoffel_2_13 * (rho * u1 * w  + metric.H_contra_13 * pressure) \
   #      +      metric.christoffel_2_22 * (rho * u2 * u2 + metric.H_contra_22 * pressure) \
   #      + 2. * metric.christoffel_2_23 * (rho * u2 * w  + metric.H_contra_23 * pressure) \
   #      +      metric.christoffel_2_33 * (rho * w  * w  + metric.H_contra_33 * pressure)
   #diff = b - forcing[idx_rho_u2]
   #diff_norm = cp.linalg.norm(diff) / cp.linalg.norm(b)
   #if diff_norm > 1e-10:
   #   print(f'rel diff {diff_norm:.3e}')
   #   raise ValueError

   RangePush('rho_u3')
   #forcing[idx_rho_w]  = \
   forcing_vert(forcing[idx_rho_w], idx_rho_w, rho, u1, u2, w, pressure, \
         metric.christoffel_3_01, metric.christoffel_3_02, metric.christoffel_3_03, \
         metric.christoffel_3_11, metric.christoffel_3_12, metric.christoffel_3_13, \
         metric.christoffel_3_22, metric.christoffel_3_23, metric.christoffel_3_33, \
         metric.H_contra_11, metric.H_contra_12, metric.H_contra_13, \
         metric.H_contra_22, metric.H_contra_23, metric.H_contra_33, \
         metric.inv_dzdeta, gravity, metric.inv_sqrtG, mtrx.filter_k(metric.sqrtG * rho, geom))
   RangePop()

   #c = \
   #        2. * rho * (metric.christoffel_3_01 * u1 + metric.christoffel_3_02 * u2 + metric.christoffel_3_03 * w) \
   #      +      metric.christoffel_3_11 * (rho * u1 * u1 + metric.H_contra_11 * pressure) \
   #      + 2. * metric.christoffel_3_12 * (rho * u1 * u2 + metric.H_contra_12 * pressure) \
   #      + 2. * metric.christoffel_3_13 * (rho * u1 * w  + metric.H_contra_13 * pressure) \
   #      +      metric.christoffel_3_22 * (rho * u2 * u2 + metric.H_contra_22 * pressure) \
   #      + 2. * metric.christoffel_3_23 * (rho * u2 * w  + metric.H_contra_23 * pressure) \
   #      +      metric.christoffel_3_33 * (rho * w  * w  + metric.H_contra_33 * pressure) \
   #      + metric.inv_dzdeta * gravity * metric.inv_sqrtG * mtrx.filter_k(metric.sqrtG * rho, geom)
   #diff = c - forcing[idx_rho_w]
   #diff_norm = cp.linalg.norm(diff) / cp.linalg.norm(c)
   #if diff_norm > 1e-10:
   #   print(f'rel diff {diff_norm:.3e}')
   #   raise ValueError


   #forcing[idx_rho_theta] = 0.


   if case_number == 21:
      dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=False)
   elif case_number == 22:
      dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=True)

   rhs            = rhs_assemble(df1_dx1, df2_dx2, df3_dx3, metric.inv_sqrtG, forcing)
   rhs[idx_rho_w] = rhs_assemble(w_df1_dx1, w_df2_dx2, w_df3_dx3, metric.inv_sqrtG, forcing[idx_rho_w])

   #a            = -metric.inv_sqrtG * (  df1_dx1 +   df2_dx2 +   df3_dx3) - forcing
   #a[idx_rho_w] = -metric.inv_sqrtG * (w_df1_dx1 + w_df2_dx2 + w_df3_dx3) - forcing[idx_rho_w]

   #diff = a - rhs
   #diff_norm = cp.linalg.norm(diff) / cp.linalg.norm(a)
   #if diff_norm > 1e-10:
   #   print(f'rel diff {diff_norm:.3e}')
   #   raise ValueError
   

   if advection_only:
      rhs[idx_rho]       = 0.
      rhs[idx_rho_u1]    = 0.
      rhs[idx_rho_u2]    = 0.
      rhs[idx_rho_w]     = 0.
      rhs[idx_rho_theta] = 0.

   RangePop()

   RangePop()

   return rhs
