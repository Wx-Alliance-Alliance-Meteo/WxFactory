"""
  This script has the matrix-free Jacobian functions
  and the rhs of each pde
 
  At the moment there are 4 PDES (with their RHS and Jtv functions)

  1. Allen-Cahn
  2. Advection-Diffusion-Reaction
  3. Porous Medium
  4. Inviscid Burger's

"""

import numpy as np

from mpi4py import MPI
from stiff_pdes import JTV


# -------------------------ALLEN-CAHN--------------------------------


# RHS
def allencahn_rhs(u, epsilon, world):

    # 1. laplacian with Neumann boundary conditions
    out_lap = JTV.laplacianNeumann(u, epsilon, world)

    # 2. other terms
    out_other = u - u**3

    # 3. output
    out = out_lap + out_other

    return out


# JAC-MATVEC
# 1. vec = vector multiplying laplacian with
# 2. u   = previous solution for non-linear jacobian
def allencahn_jtv(vec, u, dt, epsilon, alpha, gamma, world):

    # 1. laplacian with neumann bcs
    out_lap = JTV.laplacianNeumann(vec, epsilon, world)

    # 2. other terms: derivative of u - u^3
    out_other = vec - (3.0 * u**2) * vec

    # 3. total
    out = out_lap + out_other

    return out * dt


# -------------------------------------------------------------------


# -------------------------ADV-DIFF-REACTION--------------------------------


# RHS
def adr_rhs(vec, epsilon, alpha, gamma, world):

    # 1. laplacian with homogeneous bc's
    out_lap = JTV.laplacianNeumann(vec, epsilon, world)

    # 2. advection with homogeneous bc's
    out_advec = JTV.advectionNeumann(vec, alpha, world)

    # 3. reaction term
    out_reaction = gamma * vec * (vec - 0.5) * (1.0 - vec)

    # 4. total
    out = out_lap - out_advec + out_reaction

    return out


# JAC-MATVEC
# 1. vec = vector multiplying laplacian with
# 2. u   = previous solution for non-linear jacobian
def adr_jtv(vec, u, dt, epsilon, alpha, gamma, world):

    # 1. laplacian with neumann bc's
    out_lap = JTV.laplacianNeumann(vec, epsilon, world)

    # 2. advection with neumann bc's
    out_advec = JTV.advectionNeumann(vec, alpha, world)

    # 3. derivative of reaction term
    out_reaction = gamma * (3.0 * u - 3.0 * u**2 - 0.5) * vec

    # 4. total
    out = out_lap - out_advec + out_reaction

    return out * dt


# --------------------------------------------------------------------------


# -------------------------POROUS MEDIUM--------------------------------


# RHS
def porous_rhs(vec, alpha, world):

    # 1. advection with perioidic bc's
    out_advec = JTV.advectionPeriodic(vec, alpha, world)

    # 2. nonlinear diffusive term, this is (u^2)_xx
    out_lap = JTV.laplacianUsquared(vec, world)

    # 3. total
    out = out_lap + out_advec

    return out


# JAC-MATVEC
# 1. vec = vector multiplying laplacian with
# 2. u   = previous solution for non-linear jacobian
def porous_jtv(vec, u, dt, epsilon, alpha, gamma, world):

    # 1. advection with periodic boundary conditions
    out_advec = JTV.advectionPeriodic(vec, alpha, world)

    # 2. laplacian with periodic bc's
    # note this is not constant, need the vector
    # multiplying jac and previous solution
    out_lap = JTV.nonLinLapPeriodic(vec, u, 1.0, world)

    # 3. total
    out = out_lap + out_advec

    return out * dt


# ----------------------------------------------------------------------


# -------------------------BURGER'S----------------------------------


# RHS
# NOTE: this is with Dirichlet bc's as opposed to Pranab's
# periodic example because I was encountering division by 0
def burgers_rhs(vec, epsilon, alpha, world):

    # 1. laplacian with periodic bc's
    out_lap = JTV.laplacianDirichlet(vec, epsilon, world)

    # 2. nonlinear advection (0.5u^2)_x
    out_advec = JTV.advectionUsquaredDir(vec, world)

    # 3. total
    # NOTE: double check sign for advection because it's pluss in pranab's paper
    # but minus in Mayya's JCP
    out = out_lap - alpha * out_advec

    return out


# JAC-MATVEC
# 1. vec = vector multiplying laplacian with
# 2. u   = previous solution for non-linear jacobian
def burgers_jtv(vec, u, dt, epsilon, alpha, gamma, world):

    # 1. laplacian with periodic bcs
    out_lap = JTV.laplacianDirichlet(vec, epsilon, world)

    # 2. nonconstant advection for burgers with periodic bcs
    # (u*v)_x
    out_advec = JTV.nonLinAdvecDirichlet(vec, u, alpha, world)

    # 3. total
    out = out_lap - out_advec

    return out * dt


# -------------------------------------------------------------------
