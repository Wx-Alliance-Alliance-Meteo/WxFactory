import numpy
from numpy.typing import NDArray

from common.definitions import (
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_w,
    idx_rho_theta,
    idx_h,
    idx_u1,
    idx_u2,
    idx_hu1,
    idx_hu2,
    idx_2d_rho,
    idx_2d_rho_u,
    idx_2d_rho_w,
    idx_2d_rho_theta,
    gravity,
    cpd,
    cvd,
    Rd,
    p0,
)
from common import Configuration
from common.graphx import plot_array
from geometry import Cartesian2D, CubedSphere3D, CubedSphere2D, DFROperators, Metric2D, Metric3DTopo


def conservative_to_entropy(Q: NDArray, geom: Cartesian2D, param: Configuration) -> NDArray[numpy.float64]:
    """ Computes entropy variables V(Q) = dS/dQ from the conservative variables
    Shape: (num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2))"""
    # Possible issues: diviosion by zero
    num_equations = 4
    xp = geom.device.xp
    
    #V = xp.zeros((num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2))
    V = xp.zeros_like(Q)
    
    ρ, ρ_uu, ρ_ww, ρ_θ, uu , ww, θ = conservative_to_prim(Q)
    
    gamma = cpd/cvd
    p = p0 * ((Rd * ρ_θ)/p0)**gamma # pressure
    
    # # Compute conservative variables in terms of total energy E
    ρ_e, ρ_E, _ = Theta_to_E(ρ,ρ_uu,ρ_ww,ρ_θ)
    
    s = xp.log(p/ρ**gamma)
    
    v1 = (ρ_e * (gamma + 1 - s) - ρ_E) / ρ_e
    v2 = ρ_uu/ρ_e
    v3 = ρ_ww/ρ_e
    v4 = - ρ/ρ_e
    
    V[idx_2d_rho, :, :] = v1
    V[idx_2d_rho_u, :, :] = v2
    V[idx_2d_rho_w, :, :] = v3
    V[idx_2d_rho_theta, :, :] = v4

    return V

def conservative_to_prim(Q: NDArray):
    ρ = Q[idx_2d_rho, :, :]
    ρ_uu = Q[idx_2d_rho_u, :, :]
    ρ_ww = Q[idx_2d_rho_w, :, :] 
    ρ_θ = Q[idx_2d_rho_theta, :, :] 
    
    uu = ρ_uu / ρ
    ww = ρ_ww / ρ
    θ = ρ_θ / ρ
    
    return ρ, ρ_uu, ρ_ww, ρ_θ, uu , ww, θ

def Theta_to_E(ρ,ρ_uu,ρ_ww,ρ_θ):
    """Transform from Potential temperature to Energy"""
    uu = ρ_uu / ρ
    ww = ρ_ww / ρ
    θ = ρ_θ / ρ
    
    gamma = cpd/cvd

    ρ_e = cvd * θ * ((Rd * ρ_θ)/p0)**(gamma-1) 
    ρ_E = ρ_e + 0.5 * (ρ_uu * uu + ρ_ww * ww)
    E = ρ_E/ρ
    
    return ρ_e, ρ_E, E

def entropy_to_conservative(V: NDArray, geom: Cartesian2D, param: Configuration) -> NDArray[numpy.float64]:
    """ Compute conservative variable from entropy variables"""
    # TODO: Check this function!!!!!!!!!
    # Is it used????
    num_equations = 4
    xp = geom.device.xp
    
    v1 = V[idx_2d_rho, :, :] 
    v2 = V[idx_2d_rho_u, :, :] 
    v3 = V[idx_2d_rho_w, :, :] 
    v4 = V[idx_2d_rho_theta, :, :]
    
    gamma = cpd/cvd
    
    s = gamma - v1 + (v2**2 + v3**2)/(2*v4)
    # Check this formula!!!!
    ρ_e = ((gamma - 1)/ (-v4)**gamma)**(1/(gamma-1)) * xp.exp(-s/(gamma - 1) )
    
    ρ = - ρ_e * v4
    ρ_uu = ρ_e * v2
    ρ_ww = ρ_e * v3
    ρ_E = ρ_e * (1 - (v2**2 + v3**2)/(2*v4))
    
    # To do: Energy to potential
    
    Q = xp.zeros((num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2))
    
    # Q[idx_2d_rho, :, :] = ρ
    # Q[idx_2d_rho_u, :, :] = ρ_uu
    # Q[idx_2d_rho_w, :, :] = ρ_ww
    # Q[idx_2d_rho_theta, :, :] = ρ_θ
    
    return Q

def du_dv(Q: NDArray, geom: Cartesian2D, param: Configuration):
    """Computes matrix du_dv. 
    Shape : (num_equations, num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2)"""
    # Possible issues: diviosion by zero
    num_equations = 4
    xp = geom.device.xp
    
    K = xp.zeros((num_equations, num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2))
    
    ρ, ρ_uu, ρ_ww, ρ_θ, uu , ww, θ = conservative_to_prim(Q)
    
    gamma = cpd/cvd
    p = p0 * ((Rd * ρ_θ)/p0)**gamma # pressure
    
    # # Compute conservative variables in terms of total energy E
    # ρ_e = cvd * θ * ((Rd * ρ_θ)/p0)**(gamma-1) 
    # ρ_E = ρ_e + 0.5 * (ρ_uu * uu + ρ_ww * ww)
    
    _, _, E = Theta_to_E(ρ,ρ_uu,ρ_ww,ρ_θ)
    
    k00 = ρ
    k01 = ρ_uu
    k02 = ρ_ww
    k03 = E
    
    k11 = ρ_uu * uu + p
    k12 = ρ_uu * ww 
    k13 = uu* (E + p)
    
    k22 = ρ_ww * ww + p
    k23 = ww*(E+p)
    
    a = xp.sqrt(gamma * (p/ρ))
    H = a**2/ (gamma-1) + 0.5* (uu**2 + ww**2)
    k33 = ρ*H**2 - a**2 * (p/(gamma - 1))
    
    K[0,0,:,:,:] = k00
    K[0,1,:,:,:] = k01
    K[0,2,:,:,:] = k02
    K[0,2,:,:,:] = k03
    
    K[1,0,:,:,:] = k01
    K[1,1,:,:,:] = k11
    K[1,2,:,:,:] = k12
    K[1,2,:,:,:] = k13
    
    K[2,0,:,:,:] = k02
    K[2,1,:,:,:] = k12
    K[2,2,:,:,:] = k22
    K[2,2,:,:,:] = k23
    
    K[2,0,:,:,:] = k02
    K[2,1,:,:,:] = k12
    K[2,2,:,:,:] = k22
    K[2,2,:,:,:] = k23
    
    K[3,0,:,:,:] = k03
    K[3,1,:,:,:] = k13
    K[3,2,:,:,:] = k23
    K[3,2,:,:,:] = k33
    
    return K