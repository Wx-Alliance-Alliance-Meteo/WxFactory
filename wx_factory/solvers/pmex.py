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
    :return: `stats[6]` - number of communicated norm recomputations
    """
    if context is None:
        context = Context.get_default()

    if mmax < mmin:
        raise ValueError(f"mmax ({mmax}) must be greater than or equal to mmin ({mmin})")

    comm = context.comm

    # Reject unreachable tolerance
    tol_floor = 100.0 * float(torch.finfo(u.dtype).eps)
    if tol < tol_floor:
        raise ValueError(
            f"PMEX tolerance {tol:.1e} is unreachable in {u.dtype} precision; " f"use at least {tol_floor:.1e}."
        )

    ppo, n = u.shape
    p = ppo - 1

    if p == 0:
        p = 1
        # Add extra column of zeros
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

    # We only allow m to vary between mmin and mmax
    m = max(mmin, min(m_init, mmax))

    # Mixed precision: the basis V stays in the working precision, but the bookkeeping (H, M, Minv, N),
    # the matrix exponential and the error estimates use acc (float64 for a float32 basis).
    acc = torch.float64 if u.dtype == torch.float32 else u.dtype

    # Preallocate matrix
    V = torch.zeros((mmax + 1, n + p), dtype=u.dtype)
    H = torch.zeros((mmax + 1, mmax + 1), dtype=acc)
    Minv = torch.eye(mmax, dtype=acc)
    M = torch.eye(mmax, dtype=acc)
    N = torch.zeros([mmax, mmax], dtype=acc)

    # The MPI datatype for the reductions must match the precision of the buffer being reduced.
    mpi_real = MPI.FLOAT if u.dtype == torch.float32 else MPI.DOUBLE
    mpi_acc = MPI.DOUBLE if acc == torch.float64 else MPI.FLOAT

    # Initial condition
    w = torch.zeros((numSteps, n), dtype=u.dtype)
    w[0, :] = u[0, :].clone()

    # compute the 1-norm of u
    local_nrmU = torch.sum(abs(u[1:, :]), dim=1)
    global_normU = torch.empty_like(local_nrmU)

    context.synchronize()
    comm.Allreduce([local_nrmU, mpi_real], [global_normU, mpi_real])

    normU = torch.amax(global_normU)

    # Normalization factors
    if ppo > 1 and normU > 0:
        ex = math.ceil(math.log2(normU))
        nu = 2 ** (-ex)
        mu = 2 ** (ex)
    else:
        nu = 1.0
        mu = 1.0

    # Flip the rest of the u matrix
    u_flip = nu * torch.flipud(u[1:, :])

    # Compute and initial starting approximation for the step size

    # follow same as kiops
    tau = tau_end

    # Setting the safety factors and tolerance requirements
    if tau_end > 1:
        gamma = 0.2
        gamma_mmax = 0.1
    else:
        gamma = 0.9
        gamma_mmax = 0.6

    # Used in the adaptive selection
    old_m = -1
    old_tau = math.nan
    ohm = math.nan
    kestold = True
    same_tau = None

    tiny_err = float(torch.finfo(acc).tiny)

    l = 0

    while tau_now < tau_end:

        # Compute necessary starting information
        if j == 0:

            H[:, :] = 0.0

            V[0, 0:n] = w[l, :]

            # Update the last part of w
            for k in range(p - 1):
                i = p - k + 1
                V[j, n + k] = (tau_now**i) / math.factorial(i) * mu
            V[j, n + p - 1] = mu

            # Normalize initial vector (this norm is nonzero). Accumulate the sum of squares in `acc`
            # (float64 for a float32 basis) via the reduction's dtype rather than upcasting the whole
            # vector first: `x.astype(acc) @ x.astype(acc)` would materialize two full-size float64
            # copies of V[0, 0:n] (~68 MiB/rank on the 1 deg/L60 case), which overflows the GPU.
            local_sum = torch.sum(V[0, 0:n] * V[0, 0:n], dtype=acc)
            global_sum_nrm = torch.empty_like(local_sum)
            context.synchronize()
            comm.Allreduce([local_sum, mpi_acc], [global_sum_nrm, mpi_acc])
            beta = math.sqrt(global_sum_nrm + V[j, n : n + p].to(acc) @ V[j, n : n + p].to(acc))

            # The first Krylov basis vector
            V[j, :] /= beta

        # Incomplete orthogonalization process
        while j < m:

            j = j + 1

            # 1. Augmented matrix - vector product
            V[j, 0:n] = A(V[j - 1, 0:n]) + V[j - 1, n : n + p] @ u_flip
            V[j, n : n + p - 1] = V[j - 1, n + 1 : n + p]
            V[j, -1] = 0.0

            # 2. compute terms needed for R and T
            local_vec = (V[0 : j + 1, 0:n] @ V[j - 1 : j + 1, 0:n].T).to(acc)
            global_vec = torch.empty_like(local_vec)

            context.synchronize()
            comm.Allreduce([local_vec, mpi_acc], [global_vec, mpi_acc])

            global_vec += (V[0 : j + 1, n : n + p] @ V[j - 1 : j + 1, n : n + p].T).to(acc)

            # 3. Projection with 2-step Gauss-Seidel to the orthogonal complement
            # Note: this is done in two steps. (1) matvec and (2) a lower
            # triangular solve
            # 3a. here we set the values for matrix M, Minv, N
            if j > 1:
                M[j - 1, 0 : j - 1] = global_vec[0 : j - 1, 0]
                N[0 : j - 1, j - 1] = -global_vec[0 : j - 1, 0]
                Minv[j - 1, 0 : j - 1] = -global_vec[0 : j - 1, 0] @ Minv[0 : j - 1, 0 : j - 1]

            # 3b. part 1: the mat-vec
            rhs = (torch.eye(j, dtype=acc) + torch.matmul(N[0:j, 0:j], Minv[0:j, 0:j])) @ global_vec[0:j, 1]

            # 3c. part 2: the LOWER triangular solve
            if hasattr(torch.linalg, "solve_triangular"):
                sol = torch.linalg.solve_triangular(
                    M[0:j, 0:j].contiguous(), rhs.reshape(-1, 1), upper=False, unitriangular=True
                )[:, 0]
            else:
                sol = solve_triangular(M[0:j, 0:j], rhs, lower=True, unit_diagonal=True, check_finite=False)

            # 4. Orthogonalize
            V[j, :] -= sol.to(u.dtype) @ V[0:j, :]

            # 5. Norm of the freshly orthogonalized vector V[j], estimated by Pythagoras. Near a happy
            #    breakdown these two terms nearly cancel, so the cheap difference loses accuracy there.
            #    We trust the cheap difference while it is a healthy fraction of ||Av||^2, and fall back
            #    to an exact, communicated norm only in the cancellation regime (rare, near breakdown),
            #    where the direct sum of squares of the small residual has no cancellation.
            raw_nrm_sq = global_vec[-1, 1]
            sum_sqrd = (global_vec[0:j, 1] ** 2).sum()
            diff = raw_nrm_sq - sum_sqrd
            cancel_floor = 100.0 * float(torch.finfo(u.dtype).eps)

            if diff <= cancel_floor * raw_nrm_sq:
                # Severe cancellation: recompute the norm directly (one reduction, no cancellation).
                # float64 accumulation via `dtype=acc`, without full-size float64 temporaries (see above).
                local_sum = torch.sum(V[j, 0:n] * V[j, 0:n], dtype=acc)
                global_sum_nrm = torch.empty_like(local_sum)
                context.synchronize()
                comm.Allreduce([local_sum, mpi_acc], [global_sum_nrm, mpi_acc])
                curr_nrm = math.sqrt(global_sum_nrm + V[j, n : n + p].to(acc) @ V[j, n : n + p].to(acc))
                reg_comm_nrm += 1
            else:
                curr_nrm = torch.sqrt(diff)

            # Happy breakdown
            if curr_nrm < tol:
                happy = True
                break

            # Normalize vector and set norm to H matrix
            # acc scalar)
            V[j, :] /= float(curr_nrm)
            H[j, j - 1] = curr_nrm
            H[0:j, j - 1] = sol

            krystep += 1

        # To obtain the phi_1 function which is needed for error estimate
        H[0, j] = 1.0

        # Save h_j+1,j and remove it temporarily to compute the exponential of H
        nrm = H[j, j - 1].clone()
        H[j, j - 1] = 0.0

        # Compute the exponential of the augmented matrix
        F_half = expm(sgn * 0.5 * tau * H[0 : j + 1, 0 : j + 1])
        F = F_half @ F_half

        exps += 1

        # Restore the value of H_{m+1,m}
        H[j, j - 1] = nrm

        if happy is True:
            # Happy breakdown wrap up
            ohm = 0.0
            err = 0.0
            tau_new = min(tau_end - (tau_now + tau), tau)
            m_new = m
            happy = False

        else:

            # Local truncation error estimation
            err_half = abs(beta * nrm * F_half[j - 1, j])
            err = abs(beta * nrm * F[j - 1, j])

            # In single precision err and err_half can underflow to (near) zero once the current
            # Krylov space resolves the substep to machine accuracy. The controller below would then
            # form err / err_half = 0 / 0 = NaN (or order = log(1) = 0, dividing by zero in tau_opt)
            # and crash. Such a step is fully resolved, so accept it and hold the step size / Krylov
            # size, exactly as for a happy breakdown. In double precision these underflows do not
            # occur, so this branch never triggers there.
            if not (err > tiny_err and err_half > tiny_err and err != err_half):
                ohm = 0.0
                err = 0.0
                tau_new = min(tau_end - (tau_now + tau), tau)
                m_new = m

            else:

                # Error for this step
                old_ohm = ohm
                ohm = tau_end * err / (tau * tol)

                # Estimate order
                order = math.log(err / err_half) / math.log(2)

                # Estimate k
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

                # Krylov adaptivity
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

        # Check error against target
        if ohm <= delta:
            # Yep, got the required tolerance; update
            reject += ireject
            step += 1

            """
         if first_accepted:
            pmex.suggested_step = min(pmex.suggested_step, tau)
            pmex.suggested_m    = min(pmex.suggested_m, m_opt)
            first_accepted = False
         """

            # Udate for tau_out in the interval (tau_now, tau_now + tau)
            blownTs = 0
            nextT = tau_now + tau
            for k in range(l, numSteps):
                if abs(tau_out[k]) < abs(nextT):
                    blownTs += 1

            if blownTs != 0:
                # Copy current w to w we continue with.
                w[l + blownTs, :] = w[l, :].clone()

                for k in range(blownTs):
                    tau_phantom = tau_out[l + k] - tau_now
                    F2 = expm(sgn * tau_phantom * H[0:j, :j])
                    w[l + k, :] = (beta * F2[:j, 0]).to(u.dtype) @ V[:j, :n]

                # Advance l.
                l += blownTs

            # Using the standard scheme
            w[l, :] = (beta * F[:j, 0]).to(u.dtype) @ V[:j, :n]

            # Update tau_out
            tau_now += tau

            j = 0
            ireject = 0

            conv += err

        else:
            # Nope, try again
            ireject += 1

            # Restore the original matrix
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
