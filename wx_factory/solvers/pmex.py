import math
from collections.abc import Callable

import torch
from mpi4py import MPI
from torch import Tensor

from ..context import Context
from .dense import expm, solve_triangular


def pmex(
    tau_out: Tensor,
    A: Callable[[Tensor], Tensor],
    u: Tensor,
    tol: float = 1e-7,
    delta: float = 1.2,
    m_init: int = 10,
    mmin: int = 10,
    mmax: int = 128,
    reuse_info: bool = True,
    task1: bool = False,
    context: Context | None = None,
):
    """
    :param tau_out: Vector of `tau_out`
    :param A: The matrix argument of the ``φ`` functions
    :param u: The matrix with rows representing the vectors to be multiplied by the ``φ`` functions

    :param tol: Tolerance of the computation. Optional
    :param delta: ?. Optional
    :param m_init: ?. Optional
    :param mmax: Max size of the krylov space. Optional
    :param reuse_info: ?. Optional
    :param task1: If true, divide the result by 1/tau_out. Optional

    :param context: Context to use for computing

    :return: `w` - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``
    :return: `stats[0]` - number of substeps
    :return: `stats[1]` - number of rejected steps
    :return: `stats[2]` - number of Krylov steps
    :return: `stats[3]` - number of matrix exponentials
    :return: `stats[4]` - Error estimate
    :return: `stats[5]` - the Krylov size of the last substep
    :return: `stats[6]` - number of explicit residual-norm reductions
    """
    if context is None:
        context = Context.get_default()

    if mmax < mmin:
        raise ValueError(f"mmax ({mmax}) must be greater than or equal to mmin ({mmin})")

    comm = context.comm

    ppo, n = u.shape
    p = ppo - 1

    if p == 0:
        p = 1
        u = torch.row_stack((u, torch.zeros(len(u), dtype=u.dtype)))

    step = 0
    krystep = 0
    ireject = 0
    reject = 0
    exps = 0
    sgn = math.copysign(1, tau_out[-1])
    tau_now = 0.0
    tau_end = abs(tau_out[-1])
    happy = False
    j = 0
    conv = 0.0
    reg_comm_nrm = 0
    numSteps = len(tau_out)

    m = max(mmin, min(m_init, mmax))

    # Keep the large basis in working precision; float64 bookkeeping protects error estimates.
    acc = torch.float64 if u.dtype == torch.float32 else u.dtype

    V = torch.zeros((mmax + 1, n + p), dtype=u.dtype)
    H = torch.zeros((mmax + 1, mmax + 1), dtype=acc)
    Minv = torch.eye(mmax, dtype=acc)
    M = torch.eye(mmax, dtype=acc)
    N = torch.zeros([mmax, mmax], dtype=acc)

    # The MPI datatype for the reductions must match the precision of the buffer being reduced.
    mpi_real = MPI.FLOAT if u.dtype == torch.float32 else MPI.DOUBLE
    mpi_acc = MPI.DOUBLE if acc == torch.float64 else MPI.FLOAT

    w = torch.zeros((numSteps, n), dtype=u.dtype)
    w[0, :] = u[0, :].clone()

    local_nrmU = torch.sum(abs(u[1:, :]), dim=1)
    global_normU = torch.empty_like(local_nrmU)

    context.synchronize()
    comm.Allreduce([local_nrmU, mpi_real], [global_normU, mpi_real])

    normU = torch.amax(global_normU)

    if ppo > 1 and normU > 0:
        ex = math.ceil(math.log2(normU))
        nu = 2 ** (-ex)
        mu = 2 ** (ex)
    else:
        nu = 1.0
        mu = 1.0

    u_flip = nu * torch.flipud(u[1:, :])

    tau = tau_end

    if tau_end > 1:
        gamma = 0.2
        gamma_mmax = 0.1
    else:
        gamma = 0.9
        gamma_mmax = 0.6

    old_m = -1
    old_tau = math.nan
    ohm = math.nan
    kestold = True
    same_tau = None

    tiny_err = float(torch.finfo(acc).tiny)

    l = 0

    while tau_now < tau_end:
        if j == 0:
            H[:, :] = 0.0

            V[0, 0:n] = w[l, :]

            for k in range(p - 1):
                i = p - k + 1
                V[j, n + k] = (tau_now**i) / math.factorial(i) * mu
            V[j, n + p - 1] = mu

            # Float64 accumulation avoids full-vector copies that can exhaust GPU memory.
            local_sum = torch.sum(V[0, 0:n] * V[0, 0:n], dtype=acc)
            global_sum_nrm = torch.empty_like(local_sum)
            context.synchronize()
            comm.Allreduce([local_sum, mpi_acc], [global_sum_nrm, mpi_acc])
            beta = math.sqrt(global_sum_nrm + V[j, n : n + p].to(acc) @ V[j, n : n + p].to(acc))

            V[j, :] /= beta

        while j < m:
            j = j + 1

            V[j, 0:n] = A(V[j - 1, 0:n]) + V[j - 1, n : n + p] @ u_flip
            V[j, n : n + p - 1] = V[j - 1, n + 1 : n + p]
            V[j, -1] = 0.0

            local_vec = (V[0 : j + 1, 0:n] @ V[j - 1 : j + 1, 0:n].T).to(acc)
            global_vec = torch.empty_like(local_vec)

            context.synchronize()
            comm.Allreduce([local_vec, mpi_acc], [global_vec, mpi_acc])

            global_vec += (V[0 : j + 1, n : n + p] @ V[j - 1 : j + 1, n : n + p].T).to(acc)

            # Two Gauss-Seidel projections reduce loss of orthogonality.
            if j > 1:
                M[j - 1, 0 : j - 1] = global_vec[0 : j - 1, 0]
                N[0 : j - 1, j - 1] = -global_vec[0 : j - 1, 0]
                Minv[j - 1, 0 : j - 1] = -global_vec[0 : j - 1, 0] @ Minv[0 : j - 1, 0 : j - 1]

            rhs = (torch.eye(j, dtype=acc) + torch.matmul(N[0:j, 0:j], Minv[0:j, 0:j])) @ global_vec[0:j, 1]

            if hasattr(torch.linalg, "solve_triangular"):
                sol = torch.linalg.solve_triangular(
                    M[0:j, 0:j].contiguous(), rhs.reshape(-1, 1), upper=False, unitriangular=True
                )[:, 0]
            else:
                sol = solve_triangular(M[0:j, 0:j], rhs, lower=True, unit_diagonal=True, check_finite=False)

            V[j, :] -= sol.to(u.dtype) @ V[0:j, :]

            # Direct residual norms preserve basis normalization in working precision.
            local_sum = torch.sum(V[j, 0:n] * V[j, 0:n], dtype=acc)
            global_sum_nrm = torch.empty_like(local_sum)
            context.synchronize()
            comm.Allreduce([local_sum, mpi_acc], [global_sum_nrm, mpi_acc])
            curr_nrm = math.sqrt(global_sum_nrm + V[j, n : n + p].to(acc) @ V[j, n : n + p].to(acc))
            reg_comm_nrm += 1

            H[0:j, j - 1] = sol
            if curr_nrm < tol:
                happy = True
                break

            V[j, :] /= float(curr_nrm)
            H[j, j - 1] = curr_nrm

            krystep += 1

        H[0, j] = 1.0

        # Save h_j+1,j and remove it temporarily to compute the exponential of H
        nrm = H[j, j - 1].clone()
        H[j, j - 1] = 0.0

        F_half = expm(sgn * 0.5 * tau * H[0 : j + 1, 0 : j + 1])
        F = F_half @ F_half

        exps += 1

        H[j, j - 1] = nrm

        if happy is True:
            ohm = 0.0
            err = 0.0
            tau_new = min(tau_end - (tau_now + tau), tau)
            m_new = m
            happy = False

        else:
            err_half = abs(beta * nrm * F_half[j - 1, j])
            err = abs(beta * nrm * F[j - 1, j])
            if not (math.isfinite(err) and math.isfinite(err_half)):
                raise FloatingPointError("PMEX matrix exponential produced a non-finite error estimate")

            # Resolved float32 steps can underflow both errors to zero; accept them
            # to avoid 0/0 or zero order in the adaptive controller.
            if not (err > tiny_err and err_half > tiny_err and err != err_half):
                ohm = 0.0
                err = 0.0
                tau_new = min(tau_end - (tau_now + tau), tau)
                m_new = m

            else:
                old_ohm = ohm
                ohm = tau_end * err / (tau * tol)

                order = math.log(err / err_half) / math.log(2)

                if m != old_m and tau == old_tau and ireject >= 1:
                    kest = max(1.1, (ohm / old_ohm) ** (1 / (old_m - m)))
                    kestold = False
                elif kestold is True or ireject == 0:
                    kest = 2
                    kestold = True
                else:
                    kestold = True

                if ohm > delta:
                    remaining_time = tau_end - tau_now
                else:
                    remaining_time = tau_end - (tau_now + tau)

                same_tau = min(remaining_time, tau)

                tau_opt = tau * (gamma / ohm) ** (1 / order)
                tau_opt = min(remaining_time, max(tau / 5, min(5 * tau, tau_opt)))

                m_opt = math.ceil(j + math.log(ohm / gamma) / math.log(kest))
                m_opt = max(mmin, min(mmax, max(math.floor(3 / 4 * m), min(m_opt, math.ceil(4 / 3 * m)))))

                if j == mmax:
                    if ohm > delta:
                        m_new = j
                        tau_new = tau * (gamma_mmax / ohm) ** (1 / order)
                        tau_new = min(tau_end - tau_now, max(tau / 5, tau_new))
                    else:
                        tau_new = tau_opt
                        m_new = m
                else:
                    if same_tau < tau:
                        m_new = m  # We reduced tau to avoid small step size. Then keep m constant.
                    else:
                        m_new = m_opt
                    tau_new = same_tau

        if ohm <= delta:
            reject += ireject
            step += 1

            """
         if first_accepted:
            pmex.suggested_step = min(pmex.suggested_step, tau)
            pmex.suggested_m    = min(pmex.suggested_m, m_opt)
            first_accepted = False
         """

            blownTs = 0
            nextT = tau_now + tau
            for k in range(l, numSteps):
                if abs(tau_out[k]) < abs(nextT):
                    blownTs += 1

            if blownTs != 0:
                w[l + blownTs, :] = w[l, :].clone()

                for k in range(blownTs):
                    tau_phantom = tau_out[l + k] - tau_now
                    F2 = expm(sgn * tau_phantom * H[0:j, :j])
                    w[l + k, :] = (beta * F2[:j, 0]).to(u.dtype) @ V[:j, :n]

                l += blownTs

            w[l, :] = (beta * F[:j, 0]).to(u.dtype) @ V[:j, :n]

            tau_now += tau

            j = 0
            ireject = 0

            conv += err

        else:
            ireject += 1

            H[0, j] = 0.0

        old_tau = tau
        tau = tau_new

        old_m = m
        m = m_new

    if task1 is True:
        for k in range(numSteps):
            w[k, :] = w[k, :] / tau_out[k]

    m_ret = m

    stats = (step, reject, krystep, exps, conv, m_ret, reg_comm_nrm)

    return w, stats
