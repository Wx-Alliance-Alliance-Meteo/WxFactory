import numpy
import math

from definitions import *
from winds import *

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
