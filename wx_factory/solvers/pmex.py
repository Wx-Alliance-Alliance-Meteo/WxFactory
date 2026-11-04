import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from mpi4py import MPI
from torch import Tensor

from ..context import Context
from .dense import expm, solve_triangular
from .global_operations import host_allreduce, to_device


@dataclass
class PmexStartup:
    """First accepted substep and Krylov dimension from the preceding successful solve."""

    first_accepted: float | None = None
    first_dimension: int | None = None


def pmex(
    tau_out: Tensor,
    A: Callable[[Tensor], Tensor],
    u: Tensor,
    tol: float = 1e-7,
    delta: float = 1.2,
    m_init: int = 10,
    mmin: int = 10,
    mmax: int = 128,
    task1: bool = False,
    context: Context | None = None,
    startup: PmexStartup | None = None,
):
    """
    :param tau_out: Vector of `tau_out`
    :param A: The matrix argument of the ``φ`` functions
    :param u: The matrix with rows representing the vectors to be multiplied by the ``φ`` functions

    :param tol: Tolerance of the computation. Optional
    :param delta: ?. Optional
    :param m_init: ?. Optional
    :param mmax: Max size of the krylov space. Optional
    :param task1: If true, divide the result by 1/tau_out. Optional

    :param context: Context to use for computing
    :param startup: Optional per-solver history for the first substep and Krylov dimension

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
    if isinstance(tau_out, Tensor):
        tau_out = tau_out.detach().cpu().tolist()

    ppo, n = u.shape
    p = ppo - 1

    if p == 0:
        p = 1
        u = torch.row_stack((u, u.new_zeros((1, n))))

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

    if startup is not None:
        cached_dimension = startup.first_dimension
        m_init = comm.allreduce(m_init if cached_dimension is None else cached_dimension, op=MPI.MAX)
    m = max(mmin, min(m_init, mmax))

    # Keep the large basis in working precision; float64 bookkeeping protects error estimates.
    acc = torch.float64 if u.dtype == torch.float32 else u.dtype

    # The augmented rows and the Hessenberg bookkeeping are tiny; keeping them on the host avoids
    # device synchronizations for scalars that the controller reads anyway.
    device, host = u.device, torch.device("cpu")
    V = torch.zeros((mmax + 1, n), dtype=u.dtype, device=device)
    V_aug = torch.zeros((mmax + 1, p), dtype=u.dtype, device=host)
    H = torch.zeros((mmax + 1, mmax + 1), dtype=acc, device=host)
    Minv = torch.eye(mmax, dtype=acc, device=host)
    M = torch.eye(mmax, dtype=acc, device=host)
    N = torch.zeros([mmax, mmax], dtype=acc, device=host)

    w = torch.zeros((numSteps, n), dtype=u.dtype, device=device)
    w[0, :] = u[0, :].clone()

    global_normU = host_allreduce(torch.sum(abs(u[1:, :]), dim=1), comm)

    normU = float(torch.amax(global_normU))

    if ppo > 1 and normU > 0:
        ex = math.ceil(math.log2(normU))
        nu = 2 ** (-ex)
        mu = 2 ** (ex)
    else:
        nu = 1.0
        mu = 1.0

    u_flip = nu * torch.flipud(u[1:, :])

    tau = tau_end
    if startup is not None:
        first_step = startup.first_accepted
        if first_step is not None and (not math.isfinite(first_step) or first_step <= 0):
            raise ValueError("PMEX initial step must be positive and finite")
        tau = comm.allreduce(min(float(tau_end), first_step) if first_step is not None else float(tau_end), op=MPI.MIN)
    first_accepted = None
    first_dimension = None

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

            V[0] = w[l, :]

            for k in range(p - 1):
                i = p - k - 1
                V_aug[j, k] = ((sgn * tau_now) ** i) / math.factorial(i) * mu
            V_aug[j, p - 1] = mu

            global_sum_nrm = host_allreduce(torch.sum(V[0] * V[0], dtype=acc), comm)
            beta = math.sqrt(global_sum_nrm + V_aug[j].to(acc) @ V_aug[j].to(acc))

            V[j] /= beta
            V_aug[j] /= beta

        while j < m:
            j = j + 1

            augmented = to_device(V_aug[j - 1], device)
            V[j] = A(V[j - 1]) + augmented @ u_flip
            V_aug[j, : p - 1] = V_aug[j - 1, 1:p]
            V_aug[j, -1] = 0.0

            # Two GEMVs avoid a slow two-column float64 GEMM on GPUs.
            if V.is_cuda and V.dtype == torch.float64:
                basis = V[0 : j + 1]
                local_vec = torch.stack((basis @ V[j - 1], basis @ V[j]), dim=1).to(acc)
            else:
                local_vec = (V[0 : j + 1] @ V[j - 1 : j + 1].T).to(acc)
            global_vec = host_allreduce(local_vec, comm)

            global_vec += (V_aug[0 : j + 1] @ V_aug[j - 1 : j + 1].T).to(acc)

            # Two Gauss-Seidel projections reduce loss of orthogonality.
            if j > 1:
                M[j - 1, 0 : j - 1] = global_vec[0 : j - 1, 0]
                N[0 : j - 1, j - 1] = -global_vec[0 : j - 1, 0]
                Minv[j - 1, 0 : j - 1] = -global_vec[0 : j - 1, 0] @ Minv[0 : j - 1, 0 : j - 1]

            correction = torch.eye(j, dtype=acc, device=host) + torch.matmul(N[0:j, 0:j], Minv[0:j, 0:j])
            rhs = correction @ global_vec[0:j, 1]

            if hasattr(torch.linalg, "solve_triangular"):
                sol = torch.linalg.solve_triangular(
                    M[0:j, 0:j].contiguous(), rhs.reshape(-1, 1), upper=False, unitriangular=True
                )[:, 0]
            else:
                sol = solve_triangular(M[0:j, 0:j], rhs, lower=True, unit_diagonal=True, check_finite=False)

            projection = sol.to(u.dtype)
            V[j] -= to_device(projection, device) @ V[0:j]
            V_aug[j] -= projection @ V_aug[0:j]

            # Direct residual norms preserve basis normalization in working precision.
            global_sum_nrm = host_allreduce(torch.sum(V[j] * V[j], dtype=acc), comm)
            curr_nrm = math.sqrt(global_sum_nrm + V_aug[j].to(acc) @ V_aug[j].to(acc))
            reg_comm_nrm += 1

            H[0:j, j - 1] = sol
            if curr_nrm < tol:
                happy = True
                break

            V[j] /= float(curr_nrm)
            V_aug[j] /= float(curr_nrm)
            H[j, j - 1] = curr_nrm

            krystep += 1

        H[0, j] = 1.0

        # Save h_j+1,j and remove it temporarily to compute the exponential of H
        nrm = float(H[j, j - 1])
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
            # Read both estimates once for the host-side controller.
            err_half, err = torch.stack((err_half, err)).detach().cpu().tolist()
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

                # Before the error is asymptotic, halving tau barely changes it and the estimated order can
                # approach zero or turn negative, so tau ** (1 / order) overflows. The order is at least one.
                order = max(1.0, math.log(err / err_half) / math.log(2))

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
                if ohm <= delta:
                    # Large reductions often need another basis expansion on the next substep.
                    m_opt = max(m_opt, math.ceil(0.9 * m))

                if j == mmax:
                    if ohm > delta:
                        m_new = j
                        tau_new = tau * (gamma_mmax / ohm) ** (1 / order)
                        tau_new = min(tau_end - tau_now, max(tau / 5, tau_new))
                    elif m_opt < m and same_tau == tau:
                        # The dimension prediction assumes an unchanged substep length.
                        tau_new = same_tau
                        m_new = m_opt
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

            if first_accepted is None:
                first_accepted = float(tau)
                first_dimension = j

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
                    w[l + k, :] = to_device((beta * F2[:j, 0]).to(u.dtype), device) @ V[:j]

                l += blownTs

            w[l, :] = to_device((beta * F[:j, 0]).to(u.dtype), device) @ V[:j]

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

    if startup is not None:
        startup.first_accepted = first_accepted
        startup.first_dimension = first_dimension

    m_ret = m

    stats = (step, reject, krystep, exps, conv, m_ret, reg_comm_nrm)

    return w, stats
