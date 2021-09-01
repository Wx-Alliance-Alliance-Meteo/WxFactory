import numpy
import math

from definitions import cpd, day_in_secs, gravity, p0, Rd
from winds import *

def dcmip_T11_update_winds(geom, metric, mtrx, param, time=0):
   """
   Test 11 - Deformational Advection Test
   The velocities are time dependent and therefore must be updated in the dynamical core.
   """

   tau     = 12.0 * 86400.0                             # period of motion 12 days
   u0      = (2.0 * math.pi * geom.earth_radius) / tau  # 2 pi a / 12 days
   k0      = (10.0 * geom.earth_radius) / tau,          # Velocity Magnitude
   omega0  = (23000.0 * math.pi) / tau                  # Velocity Magnitude
   T0      = 300.0                                      # temperature
   H       = Rd * T0 / gravity                          # scale height

   p    = p0 * numpy.exp(-geom.height / H)
   ptop = p0 * math.exp(-12000.0 / H)

   lonp = geom.lon - 2.0 * math.pi * time / tau

   # Shape function
   bs = 0.2
   s = 1.0 + math.exp( (ptop-p0)/(bs*ptop) ) - numpy.exp( (p-p0)/(bs*ptop)) - numpy.exp( (ptop-p)/(bs*ptop))

   # Zonal Velocity

   ud = (omega0 * geom.earth_radius)/(bs*ptop) * numpy.cos(lonp) * (numpy.cos(geom.lat)**2.0) * math.cos(2.0 * math.pi * time / tau) * ( - numpy.exp( (p-p0)/(bs*ptop)) + numpy.exp( (ptop-p)/(bs*ptop))  )

   u = k0 * numpy.sin(lonp) * numpy.sin(lonp) * numpy.sin(2.0 * geom.lat) * math.cos(math.pi * time / tau) + u0 * numpy.cos(geom.lat) + ud

   # Meridional Velocity

   v = k0 * numpy.sin(2.0 * lonp) * numpy.cos(geom.lat) * math.cos(math.pi * time / tau)

   # Vertical Velocity

   w = -((Rd * T0) / (gravity * p)) * omega0 * numpy.sin(lonp) * numpy.cos(geom.lat) * math.cos(2.0 * math.pi * time / tau) * s

   # Contravariant components

   u1_contra, u2_contra = wind2contra(u, v, geom)
   u3_contra = w

   return u1_contra, u2_contra, u3_contra

def dcmip_advection_deformation(geom, metric, mtrx, param):
   """ Test 11 - Deformational Advection Test """
   tau     = 12.0 * 86400.0                             # period of motion 12 days
   T0      = 300.0                                      # temperature
   H       = Rd * T0 / gravity                          # scale height
   RR      = 1.0/2.0                                    # horizontal half width divided by 'a'
   ZZ      = 1000.0                                     # vertical half width
   z0      = 5000.0                                     # center point in z
   lambda0 = 5.0 * math.pi / 6.0                        # center point in longitudes
   lambda1 = 7.0 * math.pi / 6.0                        # center point in longitudes
   phi0    = 0.0                                        # center point in latitudes
   phi1    = 0.0

   #-----------------------------------------------------------------------
   #    HEIGHT AND PRESSURE
   #-----------------------------------------------------------------------

   p    = p0 * numpy.exp(-geom.height / H)

   #-----------------------------------------------------------------------
   #    WINDS
   #-----------------------------------------------------------------------
   u1_contra, u2_contra, u3_contra = dcmip_T11_update_winds(geom, metric, mtrx, param, time=0)

   #-----------------------------------------------------------------------
   #    TEMPERATURE IS CONSTANT 300 K
   #-----------------------------------------------------------------------

   t = T0

   #-----------------------------------------------------------------------
   #    RHO (density)
   #-----------------------------------------------------------------------

   rho = p/(Rd*t)


   #-----------------------------------------------------------------------
   #     Initialize theta (potential virtual temperature)
   #-----------------------------------------------------------------------

   tv = t
   theta = tv * (p0 / p)**(Rd/cpd)

   #-----------------------------------------------------------------------
   #     initialize tracers
   #-----------------------------------------------------------------------

   # Tracer 1 - Cosine Bells

   # To calculate great circle distance
   sin_tmp = numpy.empty_like(p)
   cos_tmp = numpy.empty_like(p)
   sin_tmp2 = numpy.empty_like(p)
   cos_tmp2 = numpy.empty_like(p)

   sin_tmp[:,:,:] = numpy.sin(geom.lat) * math.sin(phi0)
   cos_tmp[:,:,:]  = numpy.cos(geom.lat) * math.cos(phi0)
   sin_tmp2[:,:,:] = numpy.sin(geom.lat) * math.sin(phi1)
   cos_tmp2[:,:,:] = numpy.cos(geom.lat) * math.cos(phi1)

   # great circle distance without 'a'

   r  = numpy.arccos(sin_tmp + cos_tmp * numpy.cos(geom.lon-lambda0))
   r2  = numpy.arccos(sin_tmp2 + cos_tmp2 * numpy.cos(geom.lon-lambda1))
   d1 = numpy.minimum( 1.0, (r / RR)**2 + ((geom.height - z0) / ZZ)**2 )
   d2 = numpy.minimum( 1.0, (r2 / RR)**2 + ((geom.height - z0) / ZZ)**2 )

   q1 = 0.5 * (1.0 + numpy.cos(math.pi * d1)) + 0.5 * (1.0 + numpy.cos(math.pi * d2))

   # Tracer 2 - Correlated Cosine Bells

   q2 = 0.9 - 0.8 * q1**2

   # Tracer 3 - Slotted Ellipse

   # Make the ellipse
   q3 = numpy.zeros_like(q1)
   nk, ni, nj = q3.shape
   for k in range(nk):
      for i in range(ni):
         for j in range(nj):
            # Make the ellipse
            if d1[k,i,j] <= RR:
                q3[k,i,j] = 1.0
            elif d2[k,i,j] <= RR:
                q3[k,i,j] = 1.0
            else:
                q3[k,i,j] = 0.1

            # Put in the slot
            if geom.height[k,i,j] > z0 and abs(geom.lat[i,j]) < 0.125:
                q3[k,i,j] = 0.1

   # Tracer 4: q4 is chosen so that, in combination with the other three tracer
   #           fields with weight (3/10), the sum is equal to one

   q4 = 1.0 - 0.3 * (q1 + q2 + q3)

   return rho, u1_contra, u2_contra, u3_contra, theta, q1, q2, q3, q4

def dcmip_mountain(geom, metric, mtrx, param):

   lon_m = 3.0 * numpy.pi / 2.0
   # lon_m = 0.0
   lat_m = 0.0
   radius_m   = 3.0 * numpy.pi / 4.0 * 0.5
   height_max = 2000.0
   oscillation_half_width = numpy.pi / 16.0

   def compute_distance_radian(lon, lat):
      """ Compute the angular distance (in radians) of the given lon/lat coordinates from the center of the mountain """
      return numpy.minimum(radius_m, numpy.sqrt((lon - lon_m)**2 + (lat - lat_m)**2))

   def compute_height_from_dist(dist):
      """ Compute the height of the surface that corresponds to the given distance(s) from the mountain center.
       Based on the DCMIP case 1-3 description """
      return height_max / 2.0 * \
             (1.0 + numpy.cos(numpy.pi * dist / radius_m)) * \
             numpy.cos(numpy.pi * dist / oscillation_half_width)

   # Distances from the mountain on all grid and interface points
   distance = compute_distance_radian(geom.lon[0, :, :], geom.lat[0, :, :])
   distance_itf_i = compute_distance_radian(geom.lon_itf_i, geom.lat_itf_i)
   distance_itf_j = compute_distance_radian(geom.lon_itf_j, geom.lat_itf_j)

   # Height at every grid and interface point
   h_surf = compute_height_from_dist(distance)

   nb_interfaces_horiz = param.nb_elements_horizontal + 1
   h_surf_itf_i = numpy.zeros((param.nb_elements_horizontal+2, param.nbsolpts*param.nb_elements_horizontal, 2))
   h_surf_itf_j = numpy.zeros((param.nb_elements_horizontal+2, 2, param.nbsolpts*param.nb_elements_horizontal))

   h_surf_itf_i[0:nb_interfaces_horiz,   :, 1] = compute_height_from_dist(distance_itf_i.T)
   h_surf_itf_i[1:nb_interfaces_horiz+1, :, 0] = h_surf_itf_i[0:nb_interfaces_horiz, :, 1]

   h_surf_itf_j[0:nb_interfaces_horiz,   1, :] = compute_height_from_dist(distance_itf_j)
   h_surf_itf_j[1:nb_interfaces_horiz+1, 0, :] = h_surf_itf_j[0:nb_interfaces_horiz, 1, :]

   # Height derivative along x and y at every grid point
   _, ni, nj = geom.lon.shape
   dhdx1 = numpy.zeros((ni, nj))
   dhdx2 = numpy.zeros((ni, nj))

   offset = 1 # Offset due to the halo
   for elem in range(param.nb_elements_horizontal):
      epais = elem * param.nbsolpts + numpy.arange(param.nbsolpts)

      # --- Direction x1
      dhdx1[:, epais] = h_surf[:,epais] @ mtrx.diff_solpt_tr + h_surf_itf_i[elem+offset,:,:] @ mtrx.correction_tr

      # --- Direction x2
      dhdx2[epais,:] = mtrx.diff_solpt @ h_surf[epais,:] + mtrx.correction @ h_surf_itf_j[elem+offset,:,:]

   return h_surf, h_surf_itf_i, h_surf_itf_j, dhdx1, dhdx2


def dcmip_gravity_wave(geom, metric, mtrx, param):

   u0      = 20.0                 # Reference Velocity
   Teq     = 300.0                # Temperature at Equator
   Peq     = 100000.0             # Reference PS at Equator
   lambdac = 2.0 * math.pi / 3.0  # Lon of Pert Center
   d       = 5000.0               # Width for Pert
   phic    = 0.0                  # Lat of Pert Center
   delta_theta = 1.0              # Max Amplitude of Pert
   Lz      = 20000.0              # Vertical Wavelength of Pert
   N       = 0.010                 # Brunt-Vaisala frequency
   N2      = N**2                  # Brunt-Vaisala frequency Squared
   bigG    = (gravity**2)/(N2*cpd) # Constant

   inv_kappa = cpd/Rd

   p0 = 100000.0 # reference pressure (Pa)

   fld_shape = geom.height.shape

   # Zonal Velocity
   u = u0 * numpy.cos(geom.lat)

   # Meridional Velocity
   v = numpy.zeros(fld_shape)

   # Contravariant components
   u1_contra, u2_contra = wind2contra(u, v, geom)

   # Vertical Velocity
   u3_contra = numpy.zeros(fld_shape)

   # Surface temperature
   Ts = bigG + (Teq - bigG) * numpy.exp( -(u0 * N2 / (4.0 * gravity**2)) * (u0 + 2.0 * geom.rotation_speed * geom.earth_radius) * (numpy.cos(2.0 * geom.lat) - 1.0) )

   # Pressure
   ps = Peq * numpy.exp( (u0 / (4.0 * bigG * Rd)) * (u0 + 2.0 * geom.rotation_speed * geom.earth_radius) * (numpy.cos(2.0 * geom.lat) - 1.0) ) * (Ts/Teq)**inv_kappa

   p = ps * ( (bigG / Ts) * numpy.exp(-N2 * geom.height / gravity) + 1.0 - (bigG / Ts) )**inv_kappa

   # Background potential temperature
   theta_base = Ts * (p0/ps)**kappa * numpy.exp(N2 * geom.height / gravity)

   # Background temperature
   Tb = bigG * (1.0 - numpy.exp(N2 * geom.height / gravity)) + Ts * numpy.exp(N2 * geom.height / gravity)

   # density is initialized with unperturbed background temperature, temperature perturbation is added afterwards
   rho = p / (Rd * Tb)

   # Potential temperature perturbation
   sin_tmp = numpy.sin(geom.lat) * math.sin(phic)
   cos_tmp = numpy.cos(geom.lat) * math.cos(phic)

   r = geom.earth_radius * numpy.cos(sin_tmp + cos_tmp * numpy.cos(geom.lon - lambdac))
   s = (d**2)/(d**2 + r**2)

   theta_pert = delta_theta * s * numpy.sin(2.0 * math.pi * geom.height / Lz)

   # Potential temperature
   theta = theta_base + theta_pert

   return rho, u1_contra, u2_contra, u3_contra, theta
