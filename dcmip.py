import numpy
import math

from definitions import *
from winds import *

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

   X       = 125.0                # Reduced Earth reduction factor
   rotation_speed      = 0.0                  # Rotation Rate of Earth (om)
   radius      = earth_radius/X              # New Radius of small Earth (as)
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

   ni, nj, nk = geom.height.shape

   # Zonal Velocity
   u = u0 * numpy.cos(geom.lat)
#   u = numpy.zeros((ni, nj, nk))
#   for k in range(nk):
#      u[:,:,k] = u0 * numpy.cos(geom.lat)
   
   # Meridional Velocity
   v = numpy.zeros((ni,nj,nk))

   # Contravariant components
   u1_contra, u2_contra = wind2contra(u, v, geom)

   # Vertical Velocity
   u3_contra = numpy.zeros((ni,nj,nk))

   # Surface temperature
   Ts = bigG + (Teq - bigG) * numpy.exp( -(u0 * N2 / (4.0 * gravity**2)) * (u0 + 2.0 * rotation_speed * radius) * (numpy.cos(2.0 * geom.lat) - 1.0) )
   
   # Pressure
   ps = Peq * numpy.exp( (u0 / (4.0 * bigG * Rd)) * (u0 + 2.0 * rotation_speed * radius) * (numpy.cos(2.0 * geom.lat) - 1.0) ) * (Ts/Teq)**inv_kappa

   p = ps * ( (bigG / Ts) * numpy.exp(-N2 * geom.height / gravity) + 1.0 - (bigG / Ts) )**inv_kappa

   # Background potential temperature
   theta_base = Ts * (p0/ps)**kappa * numpy.exp(N2 * geom.height / gravity)

   # Background temperature
   Tb = bigG * (1.0 - numpy.exp(N2 * geom.height / gravity)) + Ts * numpy.exp(N2 * geom.height / gravity)

   # density is initialized with unperturbed background temperature, temperature perturbation is added afterwards
   density = p / (Rd * Tb)

   # Potential temperature perturbation
   sin_tmp = numpy.sin(geom.lat) * math.sin(phic)
   cos_tmp = numpy.cos(geom.lat) * math.cos(phic)

   r = radius * numpy.cos(sin_tmp + cos_tmp * numpy.cos(geom.lon - lambdac))
   s = (d**2)/(d**2 + r**2)

   theta_pert = delta_theta * s * numpy.sin(2.0 * math.pi * geom.height / Lz)

   # Potential temperature
   theta = theta_base + theta_pert
  
   return density, u1_contra, u2_contra, u3_contra, theta
