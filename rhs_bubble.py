import sys
import pickle
import numpy as np
import cupy as cp
from cupyx import jit
from numpy.typing import NDArray

p0: float
Rd: float
cpd: float
cvd: float
heat_capacity_ratio: float

Q: NDArray[cp.floating]
nbsolpts: int
nb_elements_x: int
nb_elements_z: int

dx1: float
dx3: float

flux_x1: NDArray[cp.floating]
flux_x3: NDArray[cp.floating]

kfaces_var: NDArray[cp.floating]
ifaces_var: NDArray[cp.floating]

kfaces_pres: NDArray[cp.floating]
ifaces_pres: NDArray[cp.floating]

kfaces_flux: NDArray[cp.floating]
ifaces_flux: NDArray[cp.floating]

extrap_down: NDArray[cp.floating]
extrap_up: NDArray[cp.floating]
extrap_east: NDArray[cp.floating]
extrap_west: NDArray[cp.floating]
diff_solpt: NDArray[cp.floating]
correction: NDArray[cp.floating]

df1_dx1: NDArray[cp.floating]
df3_dx3: NDArray[cp.floating]

def kfaces_var_full_loop():
    for elem in range(nb_elements_z):
        epais = elem * nbsolpts + cp.arange(nbsolpts)

        kfaces_var[:, elem, 0, :] = extrap_down @ Q[:, epais, :]
        kfaces_var[:, elem, 1, :] = extrap_up @ Q[:, epais, :]

def ifaces_var_full_loop():
    for elem in range(nb_elements_x):
        epais = elem * nbsolpts + cp.arange(nbsolpts)

        ifaces_var[:, elem, :, 0] = Q[:, :, epais] @ extrap_west
        ifaces_var[:, elem, :, 1] = Q[:, :, epais] @ extrap_east

def kfaces_flux_preamble():
    kfaces_flux[:, 0, 0, :] = 0.0
    kfaces_flux[:, -1, 1, :] = 0.0
    kfaces_flux[2, 0, 0, :] = kfaces_pres[0, 0, :]
    kfaces_flux[2, -1, 1, :] = kfaces_pres[-1, 1, :]

def kfaces_flux_full_loop():
    for itf in range(1, nb_elements_z):

        left = itf - 1
        right = itf

        a_L = cp.sqrt(heat_capacity_ratio * kfaces_pres[left, 1, :] / kfaces_var[0, left, 1, :])
        M_L = kfaces_var[2, left, 1, :] / (kfaces_var[0, left, 1, :] * a_L)

        a_R = cp.sqrt(heat_capacity_ratio * kfaces_pres[right, 0, :] / kfaces_var[0, right, 0, :])
        M_R = kfaces_var[2, right, 0, :] / (kfaces_var[0, right, 0, :] * a_R)

        M = 0.25 * ((M_L + 1.) ** 2 - (M_R - 1.) ** 2)

        kfaces_flux[:, right, 0, :] = (kfaces_var[:, left, 1, :] * cp.maximum(0., M) * a_L) + (kfaces_var[:, right, 0, :] * cp.minimum(0., M) * a_R)
        kfaces_flux[2, right, 0, :] += 0.5 * ((1. + M_L) * kfaces_pres[left, 1, :] + (1. - M_R) * kfaces_pres[right, 0, :])

        kfaces_flux[:, left, 1, :] = kfaces_flux[:, right, 0, :]

def ifaces_flux_preamble():
    # included: xperiodic
    ifaces_flux[:, 0, :, 0] = 0.0
    ifaces_flux[:, -1, :, 1] = 0.0
    ifaces_flux[1, 0, :, 0] = ifaces_pres[0, :, 0]
    ifaces_flux[1, -1, :, 1] = ifaces_pres[-1, :, 1]

def ifaces_flux_full_loop():
    for itf in range(1, nb_elements_x):
        left = itf - 1
        right = itf

        # omitted: xperiodic

        a_L = cp.sqrt(heat_capacity_ratio * ifaces_pres[left, :, 1] / ifaces_var[0, left, :, 1])
        M_L = ifaces_var[1, left, :, 1] / (ifaces_var[0, left, :, 1] * a_L)

        a_R = cp.sqrt(heat_capacity_ratio * ifaces_pres[right, :, 0] / ifaces_var[0, right, :, 0])
        M_R = ifaces_var[1, right, :, 0] / (ifaces_var[0, right, :, 0] * a_R)

        M = 0.25 * ((M_L + 1. ** 2) - (M_R - 1.) ** 2)

        ifaces_flux[:, right, :, 0] = (ifaces_var[:, left, :, 1] * cp.maximum(0., M) * a_L) + (ifaces_var[:, right, :, 0] * cp.minimum(0., M) * a_R)
        ifaces_flux[1, right, :, 0] += 0.5 * ((1. + M_L) * ifaces_pres[left, :, 1] + (1. - M_R) * ifaces_pres[right, :, 0])

        ifaces_flux[:, left, :, 1] = ifaces_flux[:, right, :, 0]

@jit.rawkernel()
def set_ifaces_flux(ifaces_flux, ifaces_pres, ifaces_var, nb_equations, max_j, xperiodic):
    """
    Run with `nb_interfaces_x - 2` x `nbsolpts * nb_elements_z` x `1` threads,
    with block size `(1, ?, 1)`
    and `max_j = nbsolpts * nb_elements_z`
    """
    j = jit.blockIdx.y * jit.blockDim.y + jit.threadIdx.y
    if j < max_j:
        i = jit.blockIdx.x
        left = i
        right = i + 1
        hcr, rho, rho_u = heat_capacity_ratio, 0, 1

        # Left state
        if xperiodic and left == 0:
            left = -1
        
        a_L = cp.sqrt(hcr * ifaces_pres[left, j, 1] / ifaces_var[rho, left, j, 1])
        M_L = ifaces_var[rho_u, left, j, 1] / (ifaces_var[rho, left, j, 1] * a_L)

        # Right state
        a_R = cp.sqrt(hcr * ifaces_pres[right, j, 0] / ifaces_var[rho, right, j, 0])
        M_R = ifaces_var[rho_u, right, j, 0] / (ifaces_var[rho, right, j, 0] * a_R)

        M = 0.25 * ((M_L + 1.) ** 2 - (M_R - 1.) ** 2)
        rho_u_extra = 0.5 * ((1. + M_L) * ifaces_pres[left, j, 1] + (1. - M_R) * ifaces_pres[right, j, 0])

        if M > 0:
            for k in jit.range(nb_equations):
                r = ifaces_var[k, left, j, 1] * M * a_L
                if k == rho_u:
                    r += rho_u_extra
                ifaces_flux[k, right, j, 0] = r
                ifaces_flux[k, left, j, 1] = r
        else:
            for k in jit.range(nb_equations):
                r = ifaces_var[k, right, j, 0] * M * a_R
                if k == rho_u:
                    r += rho_u_extra
                ifaces_flux[k, right, j, 0] = r
                ifaces_flux[k, left, j, 1] = r

def df3_dx3_full_loop():
    for elem in range(nb_elements_z):
        epais = elem * nbsolpts + cp.arange(nbsolpts)

        df3_dx3[:, epais, :] = (diff_solpt @ flux_x3[:, epais, :] + correction @ kfaces_flux[:, elem, :, :]) * 2.0 / dx3

def df1_dx1_full_loop():
    for elem in range(nb_elements_x):
        epais = elem * nbsolpts + cp.arange(nbsolpts)

        df1_dx1[:, :, epais] = (flux_x1[:, :, epais] @ diff_solpt.T + ifaces_flux[:, elem, :, :] @ correction.T) * 2.0 / dx1

@jit.rawkernel()
def set_df1_dx1(df1_dx1, flux_x1, ifaces_flux, diff_solpt, correction, dx1, nbsolpts, nb_elements_x):
    """
    Run with `Q.shape[0]` x `Q.shape[1]` x `Q.shape[2] // nbsolpts` total threads,
    with block size `(1, 1, ?)`
    note that diff_solpt and correction are automatically transposed
    """
    i, j, start, r = 0, 0, 0, 0.0
    k = jit.blockIdx.z * jit.blockDim.z + jit.threadIdx.z
    if k < nb_elements_x:
        i = jit.blockIdx.x
        j = jit.blockIdx.y

        start = nbsolpts * k
        for l in jit.range(nbsolpts):
            r = 0.0

            for m in jit.range(nbsolpts):
                r += flux_x1[i, j, start + m] * diff_solpt[l, m]
            for m in jit.range(2):
                r += ifaces_flux[i, k, j, m] * correction[l, m]
            
            df1_dx1[i, j, start + l] = 2.0 * r / dx1

with open(r"rhs\rhs_bubble.cu") as file:
    _rhs_bubble_cu = cp.RawModule(code=file.read())

def new_df1_dx1():
    B = 64
    blocks, extra = divmod(nb_elements_x, B)
    gridspec = (Q.shape[0], Q.shape[1], blocks + (1 if extra else 0))
    blockspec = (1, 1, B)
    params = (df1_dx1, flux_x1, ifaces_flux, diff_solpt, correction,
              dx1, nbsolpts, nb_elements_z, nb_elements_x)
    _rhs_bubble_cu.get_function("new_df1_dx1")(gridspec, blockspec, params)

if __name__ == "__main__":

    with open(sys.argv[1], "rb") as file:
        ex = pickle.load(file)
    ex = {k: (cp.asarray(v) if type(v) is np.ndarray else v) for k, v in ex.items()}
    globals().update(ex)

    globals()["dx1"] = 1000. / nb_elements_x
    globals()["dx3"] = 1500. / nb_elements_z

    rho = Q[0, :, :]
    uu = Q[1, :, :] / rho
    ww = Q[2, :, :] / rho
    pressure = p0 * (Q[3, :, :] * Rd / p0) ** (cpd / cvd)

    globals()["flux_x1"] = flux_x1 = cp.empty_like(Q)
    flux_x1[0, :, :] = Q[1, :, :]
    flux_x1[1, :, :] = Q[1, :, :] * uu + pressure
    flux_x1[2, :, :] = Q[1, :, :] * ww
    flux_x1[3, :, :] = Q[3, :, :] * uu

    globals()["flux_x3"] = flux_x3 = cp.empty_like(Q)
    flux_x3[0, :, :] = Q[2, :, :]
    flux_x3[1, :, :] = Q[2, :, :] * uu
    flux_x3[2, :, :] = Q[2, :, :] * ww + pressure
    flux_x3[3, :, :] = Q[3, :, :] * ww

    globals()["kfaces_pres"] = p0 * (kfaces_var[3, :, :, :] * Rd / p0) ** (cpd / cvd)
    globals()["ifaces_pres"] = p0 * (ifaces_var[3, :, :, :] * Rd / p0) ** (cpd / cvd)

    breakpoint()
