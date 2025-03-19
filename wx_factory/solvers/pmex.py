import math

from mpi4py import MPI

from device import Device, CpuDevice


def pmex(
    tau_out,
    A,
    u,
    tol=1e-7,
    delta=1.2,
    m_init=10,
    mmin=10,
    mmax=128,
    reuse_info=True,
    task1=False,
    device: Device = None,
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

    :param device: Device to use for the computing
    :param comm: Communicator to use for MPI (only relevant for testing)

    :return: `w` - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``
    :return: `stats[0]` - number of substeps
    :return: `stats[1]` - number of rejected steps
    :return: `stats[2]` - number of Krylov steps
    :return: `stats[3]` - number of matrix exponentials
    :return: `stats[4]` - Error estimate
    :return: `stats[5]` - the Krylov size of the last substep
    :return: `stats[6]` = ?
    """
    if device is None:
        device = Device.get_default()

    comm = device.comm

    ppo, n = u.shape
    p = ppo - 1

    if p == 0:
        p = 1
        # Add extra column of zeros
        u = device.xp.row_stack((u, device.xp.zeros(len(u))))

    step = 0
    krystep = 0
    ireject = 0
    reject = 0
    exps = 0
    sgn = device.xp.sign(tau_out[-1])
    tau_now = 0.0
    tau_end = abs(tau_out[-1])
    happy = False
    j = 0
    conv = 0.0
    reg_comm_nrm = 0
    numSteps = len(tau_out)

    first_accepted = True

    # We only allow m to vary between mmin and mmax
    # mmin = 1
    m = max(mmin, min(m_init, mmax))

    # Preallocate matrix
    V = device.xp.zeros((mmax + 1, n + p))
    H = device.xp.zeros((mmax + 1, mmax + 1))
    Minv = device.xp.eye(mmax)
    M = device.xp.eye(mmax)
    N = device.xp.zeros([mmax, mmax])

    # Initial condition
    w = device.xp.zeros((numSteps, n))
    w[0, :] = u[0, :].copy()

    # compute the 1-norm of u
    local_nrmU = device.xp.sum(abs(u[1:, :]), axis=1)
    global_normU = device.xp.empty_like(local_nrmU)

    device.synchronize()
    comm.Allreduce([local_nrmU, MPI.DOUBLE], [global_normU, MPI.DOUBLE])

    normU = device.xp.amax(global_normU)

    # Normalization factors
    if ppo > 1 and normU > 0:
        ex = math.ceil(math.log2(normU))
        nu = 2 ** (-ex)
        mu = 2 ** (ex)
    else:
        nu = 1.0
        mu = 1.0

    # Flip the rest of the u matrix
    u_flip = nu * device.xp.flipud(u[1:, :])

    # Compute and initial starting approximation for the step size
    # tau = min(pmex.suggested_step, tau_end)

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

            # Normalize initial vector (this norm is nonzero)
            local_sum = V[0, 0:n] @ V[0, 0:n]
            global_sum_nrm = device.xp.empty_like(local_sum)
            device.synchronize()
            comm.Allreduce([local_sum, MPI.DOUBLE], [global_sum_nrm, MPI.DOUBLE])
            beta = math.sqrt(global_sum_nrm + V[j, n : n + p] @ V[j, n : n + p])

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
            local_vec = V[0 : j + 1, 0:n] @ V[j - 1 : j + 1, 0:n].T
            global_vec = device.xp.empty_like(local_vec)

            device.synchronize()
            comm.Allreduce([local_vec, MPI.DOUBLE], [global_vec, MPI.DOUBLE])

            global_vec += V[0 : j + 1, n : n + p] @ V[j - 1 : j + 1, n : n + p].T

            # 3. Projection with 2-step Gauss-Seidel to the orthogonal complement
            # Note: this is done in two steps. (1) matvec and (2) a lower
            # triangular solve
            # 3a. here we set the values for matrix M, Minv, N
            if j > 1:
                M[j - 1, 0 : j - 1] = global_vec[0 : j - 1, 0]
                N[0 : j - 1, j - 1] = -global_vec[0 : j - 1, 0]
                Minv[j - 1, 0 : j - 1] = -global_vec[0 : j - 1, 0].T @ Minv[0 : j - 1, 0 : j - 1]

            # 3b. part 1: the mat-vec
            rhs = (device.xp.eye(j) + device.xp.matmul(N[0:j, 0:j], Minv[0:j, 0:j])) @ global_vec[0:j, 1]

            # 3c. part 2: the lower triangular solve
            sol = device.xalg.linalg.solve_triangular(
                M[0:j, 0:j], rhs, unit_diagonal=True, check_finite=False, overwrite_b=True
            )

            # 4. Orthogonalize
            V[j, :] -= sol @ V[0:j, :]

            # 5. compute norm estimate with quad precision
            if device.has_128_bits_float():
                sum_vec = device.xp.array(global_vec[0:j, 1], device.xp.float128) ** 2
                sum_sqrd = device.xp.sum(sum_vec)
            else:
                device.synchronize()
                default_device = CpuDevice.get_default()
                sum_vec = default_device.xp.array(global_vec[0:j, 1].get(), default_device.xp.float128) ** 2
                sum_sqrd = device.array(
                    default_device.xp.asarray(default_device.xp.sum(sum_vec), dtype=default_device.xp.float64)
                )

            # sum_sqrd = sum(global_vec[0:j,1]**2)
            if global_vec[-1, 1] < sum_sqrd:
                # use communication to compute norm estimate
                local_sum = V[j, 0:n] @ V[j, 0:n]
                global_sum_nrm = device.xp.empty_like(local_sum)
                device.synchronize()
                comm.Allreduce([local_sum, MPI.DOUBLE], [global_sum_nrm, MPI.DOUBLE])
                curr_nrm = math.sqrt(global_sum_nrm + V[j, n : n + p] @ V[j, n : n + p])
                reg_comm_nrm += 1
            else:
                # compute norm estimate in quad precision
                curr_nrm = device.xp.array(device.xp.sqrt(global_vec[-1, 1] - sum_sqrd), device.xp.float64)

            # Happy breakdown
            if curr_nrm < tol:
                happy = True
                break

            # Normalize vector and set norm to H matrix
            V[j, :] /= curr_nrm
            H[j, j - 1] = curr_nrm
            H[0:j, j - 1] = sol

            krystep += 1

        # To obtain the phi_1 function which is needed for error estimate
        H[0, j] = 1.0

        # Save h_j+1,j and remove it temporarily to compute the exponential of H
        nrm = H[j, j - 1].copy()
        H[j, j - 1] = 0.0

        # Compute the exponential of the augmented matrix
        F_half = device.xalg.linalg.expm(sgn * 0.5 * tau * H[0 : j + 1, 0 : j + 1])
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
                w[l + blownTs, :] = w[l, :].copy()

                for k in range(blownTs):
                    tau_phantom = tau_out[l + k] - tau_now
                    F2 = device.xalg.linalg.expm(sgn * tau_phantom * H[0:j, :j])
                    w[l + k, :] = beta * F2[:j, 0] @ V[:j, :n]

                # Advance l.
                l += blownTs

            # Using the standard scheme
            w[l, :] = beta * F[:j, 0] @ V[:j, :n]

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
