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
    xp = geom.device.xp
    
    #V = xp.zeros((num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2))
    V = xp.zeros_like(Q)
    
    #_________________________________________________
    # ρ, ρ_uu, ρ_ww, ρ_θ, uu , ww, θ = conservative_to_prim(Q)
    # ρ_e, ρ_E, _ = Theta_to_E(ρ,ρ_uu,ρ_ww,ρ_θ)
    # gamma = cpd/cvd
    
    # v1 = ρ_uu / ρ
    # v2 = ρ_ww / ρ
    
    # v_square = v1**2 + v2**2
    
    # p = (gamma - 1) * (ρ_E - 0.5 * ρ * v_square )
    # s = xp.log(p) - gamma * xp.log(ρ)
    # rho_p = ρ / p
    
    # w1 = (gamma - s)/(gamma-1) - 0.5* rho_p * v_square
    # w2 = rho_p * v1
    # w3 = rho_p * v2
    # w4 = - rho_p
    
    # print("\n\nw1\n",w1[0,36,:])
    # print("w2\n",w2[0,36,:])
    # print("w3\n",w3[0,36,:])
    # print("w4\n",w4[0,36,:])
    #_________________________________________________
    
    ρ, ρ_uu, ρ_ww, ρ_θ, uu , ww, θ = conservative_to_prim(Q)
    
    gamma = cpd/cvd
    p = p0 * (((Rd * ρ_θ)/p0)**gamma) # pressure
    
    # # Compute conservative variables in terms of total energy E
    ρ_e, ρ_E, _ = Theta_to_E(ρ,ρ_uu,ρ_ww,ρ_θ)
    
    
    s = xp.log(p) - gamma * xp.log(ρ)
    
    v1 = (ρ_e * (gamma + 1 - s) - ρ_E) / ρ_e
    v2 = ρ_uu/ρ_e
    v3 = ρ_ww/ρ_e
    v4 = - ρ/ρ_e
    
    # print("rho e",xp.any(xp.isclose(ρ_e,0)))
    
    # V[idx_2d_rho, :, :] = w1
    # V[idx_2d_rho_u, :, :] = w2
    # V[idx_2d_rho_w, :, :] = w3
    # V[idx_2d_rho_theta, :, :] = w4
    
    # print("\nv1\n",1 / (gamma-1) * v1[0,36,:])
    # print("v2\n",1 / (gamma-1) * v2[0,36,:])
    # print("v3\n",1 / (gamma-1) * v3[0,36,:])
    # print("v4\n",1 / (gamma-1) * v4[0,36,:])
    
    V[idx_2d_rho, :, :] = v1
    V[idx_2d_rho_u, :, :] = v2
    V[idx_2d_rho_w, :, :] = v3
    V[idx_2d_rho_theta, :, :] = v4
    
    V = 1 / (gamma-1) * V
    return V

def conservative_to_prim(Q: NDArray):
    if len(Q.shape) == 4:
        ρ = Q[idx_2d_rho, :, :, :]
        ρ_uu = Q[idx_2d_rho_u, :, :, :]
        ρ_ww = Q[idx_2d_rho_w, :, :,  :] 
        ρ_θ = Q[idx_2d_rho_theta, :, :,:] 
    elif len(Q.shape) == 3:
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

    ρ_e = ρ * cvd * θ * ((Rd * ρ_θ)/p0)**(gamma-1) 
    ρ_E = ρ_e + 0.5 * (ρ_uu * uu + ρ_ww * ww)
    E = ρ_E/ρ
    
    return ρ_e, ρ_E, E

def E_to_Theta(e,ρ):
    gamma = cpd/cvd
    θ = (e/cvd)**(1/(gamma)) * (p0/(Rd*ρ))**((gamma-1)/(gamma))
    return θ

def entropy_to_conservative(V: NDArray, geom: Cartesian2D, param: Configuration) -> NDArray[numpy.float64]:
    """ Compute conservative variable from entropy variables"""
    # TODO: Check this function!!!!!!!!!
    
    num_equations = 4
    xp = geom.device.xp
    
    gamma = cpd/cvd
    
    V = (gamma - 1) * V
    
    v1 = V[idx_2d_rho, :, :] 
    v2 = V[idx_2d_rho_u, :, :] 
    v3 = V[idx_2d_rho_w, :, :] 
    v4 = V[idx_2d_rho_theta, :, :]
    
    #_________________________________________________
    # beta = -v4
    
    # uu = v2 / beta
    # ww = v3 / beta
    
    # s = gamma - (gamma - 1) * v1 - 0.5 * (gamma - 1) * beta (uu**2 + ww**2)
    
    # rho = xp.exp(-s/(gamma-1)) * xp.power(beta,-1/(gamma-1))
    
    # p = rho/beta
    # rho_u = rho * uu
    # rho_w = rho * ww
    # rho_E = p/(gamma-1) + 0.5 * rho * (uu**2 + ww**2)
    #_________________________________________________
    
    s = gamma - v1 + (v2**2 + v3**2)/(2*v4)
    # Check this formula!!!!
    ρ_e = ((gamma - 1)/ (-v4)**gamma)**(1/(gamma-1)) * xp.exp(-s/(gamma - 1) )
    # print("gamma-1",xp.any(xp.close(gamma-1,0)))
    
    ρ = - ρ_e * v4
    ρ_uu = ρ_e * v2
    ρ_ww = ρ_e * v3
    ρ_E = ρ_e * (1 - (v2**2 + v3**2)/(2*v4))
    # print("2*v4",xp.any(xp.close(2*v4,0)))
    
    # TODO: Energy to potential
    e = ρ_e/ρ
    
    θ = E_to_Theta(e,ρ)
    ρ_θ = ρ*θ
    
    
    #Q = xp.zeros((num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2))
    Q = xp.zeros_like(V)
    
    Q[idx_2d_rho, :, :] = ρ
    Q[idx_2d_rho_u, :, :] = ρ_uu
    Q[idx_2d_rho_w, :, :] = ρ_ww
    Q[idx_2d_rho_theta, :, :] = ρ_θ
    
    return Q

def du_dv(Q: NDArray, geom: Cartesian2D, param: Configuration):
    """Computes matrix du_dv. 
    Shape : (num_equations, num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2)"""
    # Possible issues: diviosion by zero
    num_equations = 4
    xp = geom.device.xp
    
    if len(Q.shape) == 4:
        K = xp.zeros((num_equations, num_equations, param.num_elements_vertical, param.num_elements_horizontal, geom.num_solpts**2))
    elif len(Q.shape) == 3:
        K = xp.zeros((num_equations, num_equations, param.num_elements_vertical, param.num_elements_horizontal))
    
    ρ, ρ_uu, ρ_ww, ρ_θ, uu , ww, θ = conservative_to_prim(Q)
    
    gamma = cpd/cvd
    p = p0 * ((Rd * ρ_θ)/p0)**gamma # pressure
    
    # # Compute conservative variables in terms of total energy E
    # ρ_e = cvd * θ * ((Rd * ρ_θ)/p0)**(gamma-1) 
    # ρ_E = ρ_e + 0.5 * (ρ_uu * uu + ρ_ww * ww)
    
    _, ρ_E, E = Theta_to_E(ρ,ρ_uu,ρ_ww,ρ_θ)
    
    k00 = ρ
    k01 = ρ_uu
    k02 = ρ_ww
    # k03 = E
    k03 = ρ_E
    
    k11 = ρ_uu * uu + p
    k12 = ρ_uu * ww 
    # k13 = uu* (E + p)
    k13 = uu* (ρ_E + p)
    
    k22 = ρ_ww * ww + p
    # k23 = ww*(E+p)
    k23 = ww*(ρ_E+p)
    
    a2 = gamma * (p/ρ)
    H = a2 / (gamma-1) + 0.5 * (uu**2 + ww**2)
    k33 = ρ*(H**2) - a2 * (p/(gamma - 1))
    
    # print("1/gamma-1: ", 1/(gamma-1))
    # print("gamma: ", gamma)
    
    if len(Q.shape) == 4:
        K[0,0,:,:,:] = k00
        K[0,1,:,:,:] = k01
        K[0,2,:,:,:] = k02
        K[0,3,:,:,:] = k03
        
        K[1,0,:,:,:] = k01
        K[1,1,:,:,:] = k11
        K[1,2,:,:,:] = k12
        K[1,3,:,:,:] = k13
        
        K[2,0,:,:,:] = k02
        K[2,1,:,:,:] = k12
        K[2,2,:,:,:] = k22
        K[2,3,:,:,:] = k23
        
        K[3,0,:,:,:] = k03
        K[3,1,:,:,:] = k13
        K[3,2,:,:,:] = k23
        K[3,3,:,:,:] = k33
    elif len(Q.shape) == 3:
        K[0,0,:,:] = k00
        K[0,1,:,:] = k01
        K[0,2,:,:] = k02
        K[0,3,:,:] = k03
        
        K[1,0,:,:] = k01
        K[1,1,:,:] = k11
        K[1,2,:,:] = k12
        K[1,3,:,:] = k13
        
        K[2,0,:,:] = k02
        K[2,1,:,:] = k12
        K[2,2,:,:] = k22
        K[2,3,:,:] = k23
        
        K[3,0,:,:] = k03
        K[3,1,:,:] = k13
        K[3,2,:,:] = k23
        K[3,3,:,:] = k33
        
    # K = (gamma-1) * K
        
    return K

def jacobian_complex_field(func, Q, geom: Cartesian2D, param: Configuration,h=1e-20):
    """
    Compute Jacobian J = dfunc/dQ using complex-step,
    for field-shaped inputs/outputs.

    Parameters
    ----------
    func : callable
        Maps Q -> array of same leading dimension (e.g. (4,...))
    Q : ndarray
        Shape (nvar, ...)
    h : float
        Complex step size

    Returns
    -------
    J : ndarray
        Shape (nvar, nvar, ...)
    """
    # print("\n jacobian_complex_field \n")
    xp = geom.device.xp  # or pass xp if needed

    Q = xp.asarray(Q, dtype=float)
    nvar = Q.shape[0]
    spatial_shape = Q.shape[1:]

    # Evaluate once
    f0 = func(Q,geom,param)
    
    J = xp.zeros((nvar, nvar) + spatial_shape, dtype=float)

    Qc = Q.astype(complex)

    # print("nvar",nvar)
    for i in range(nvar):
        Q_step = Qc.copy()
        Q_step[i, ...] += 1j * h

        f_step = func(Q_step,geom,param)
        # print("\nf_step\n",f_step[:,0,36,0])
        # print("jac\n",(xp.imag(f_step) / h)[:,0,36,0])
        J[:, i, ...] = xp.imag(f_step) / h
        
    q = func(Q,geom,param) # conservative variables
    
    # dρE_dρθ_q = dρE_dρθ(q)
    # # print("dρ_E_ρ_θ",dρE_dρθ[0,36,:])
    # print("J[nvar-1,...] before:\n",J[nvar-1,:,0,36,0])
    # print("dρE_dρθ_q:\n",dρE_dρθ_q.shape)
    # print("dρE_dρθ_q:\n",dρE_dρθ_q[0,36,0])
    # J[nvar-1,...] *= dρE_dρθ_q
    # print("J[nvar-1,...] after:\n",J[nvar-1,:,0,36,0])

    return J

def jacobian_fd_field(func, Q, geom, param, eps=1e-6):
    xp = geom.device.xp
    nvar = Q.shape[0]
    spatial_shape = Q.shape[1:]

    J = xp.zeros((nvar, nvar) + spatial_shape)

    for i in range(nvar):
        Qp = Q.copy()
        Qm = Q.copy()
        Qp[i,...] += eps
        Qm[i,...] -= eps

        fp = func(Qp, geom, param)
        fm = func(Qm, geom, param)

        J[:, i, ...] = (fp - fm) / (2*eps)
    
    q = func(Q,geom,param) # conservative variables
    
    # dρE_dρθ_q = dρE_dρθ(q)
    # ρ, ρ_uu, ρ_ww, ρ_θ, uu , ww, θ = conservative_to_prim(q)
    # print("ρ_θ",ρ_θ[0,36,:])
    # ρ_e, ρ_E, _ = Theta_to_E(ρ,ρ_uu,ρ_ww,ρ_θ)
    # e = ρ_e / ρ
    
    # gamma = cpd/cvd
    # dρE_dρθ = gamma * e/θ
    
    # print("q new\n",q[:,0,36,:])
    # print("ρ_θ",ρ_θ[0,36,:])
    # print("ρ_e",ρ_e[0,36,:])
    # print("rho",ρ[0,36,:])
    # print("e",e[0,36,:])
    # print("θ",θ[0,36,:])
    # print("dρ_E_ρ_θ",dρE_dρθ[0,36,:])
    # print("J[nvar-1,...] before:\n",J[nvar-1,:,0,36,0])
    # print("dρE_dρθ_q:\n",dρE_dρθ_q.shape)
    # print("dρE_dρθ_q:\n",dρE_dρθ_q[0,36,0])
    # J[nvar-1,...] *= dρE_dρθ_q
    # print("J[nvar-1,...] after:\n",J[nvar-1,:,0,36,0])

    return J

def dρE_dρθ(Q):
    ρ, ρ_uu, ρ_ww, ρ_θ, uu , ww, θ = conservative_to_prim(Q)
    ρ_e, ρ_E, _ = Theta_to_E(ρ,ρ_uu,ρ_ww,ρ_θ)
    e = ρ_e / ρ
    
    gamma = cpd/cvd
    dρE_dρθ = gamma * e/θ
    return dρE_dρθ

def entropy_potential(Q: NDArray)-> NDArray[numpy.float64]:
    psi_x1 = Q[idx_2d_rho_u, :, :, :]
    psi_x2 = Q[idx_2d_rho_w, :, :,  :] 
    return psi_x1, psi_x2

def entropy(Q: NDArray,geom:Cartesian2D)-> NDArray[numpy.float64]:
    "Computes physical entropy s = log(p / rho**gamma)"
    xp = geom.device.xp
    
    ρ, _, _, ρ_θ, _ , _, _ = conservative_to_prim(Q)
    
    gamma = cpd/cvd
    p = p0 * (((Rd * ρ_θ)/p0)**gamma) # pressure
    
    s = xp.log(p) - gamma * xp.log(ρ) # physical entropy
    return s

def entropy_function(Q: NDArray,geom:Cartesian2D)-> NDArray[numpy.float64]:
    "Computes mathematical entropy function  S(u) = -rho*s"

    ρ, _, _, ρ_θ, _ , _, _ = conservative_to_prim(Q)
    s = entropy(Q,geom)
    return - ρ*s
    