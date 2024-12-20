import math
from typing import Callable

from mpi4py import MPI
from numpy import ndarray

from device import Device, default_device


def kiops(
    tau_out: ndarray,
    A: Callable[[ndarray], ndarray],
    u: ndarray,
    tol: float = 1e-7,
    m_init: int = 10,
    mmin: int = 10,
    mmax: int = 128,
    iop: int = 2,
    task1: bool = False,
    device: Device = default_device,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> tuple[ndarray, tuple]:
    """kiops(tstops, A, u; kwargs...) -> (w, stats)

    Evaluate a linear combinaton of the ``φ`` functions evaluated at ``tA`` acting on
    vectors from ``u``, that is

    ```math
    w(i) = φ_0(t[i] A) u[0, :] + φ_1(t[i] A) u[1, :] + φ_2(t[i] A) u[2, :] + ...
    ```

    The size of the Krylov subspace is changed dynamically during the integration.
    The Krylov subspace is computed using the incomplete orthogonalization method.

    `n` is the size of the original problem
    `p` is the highest index of the ``φ`` functions

    References:
    * Gaudreault, S., Rainwater, G. and Tokman, M., 2018. KIOPS: A fast adaptive Krylov subspace solver for exponential
      integrators. Journal of Computational Physics. Based on the PHIPM and EXPMVP codes
      (http://www1.maths.leeds.ac.uk/~jitse/software.html). https://gitlab.com/stephane.gaudreault/kiops.
    * Niesen, J. and Wright, W.M., 2011. A Krylov subspace method for option pricing. SSRN 1799124
    * Niesen, J. and Wright, W.M., 2012. Algorithm 919: A Krylov subspace algorithm for evaluating the ``φ``-functions
      appearing in exponential integrators. ACM Transactions on Mathematical Software (TOMS), 38(3), p.22

    :param tau_out: Array of `tau_out`
    :param A: the matrix argument of the ``φ`` functions
    :param u: the matrix with rows representing the vectors to be multiplied by the ``φ`` functions

    :param tol: the convergence tolerance required (default: 1e-7). Optional
    :param mmin: let the Krylov size vary between mmin and mmax (default: 10, 128). Optional
    :param mmax: let the Krylov size vary between mmin and mmax (default: 10, 128). Optional
    :param m: an estimate of the appropriate Krylov size (default: mmin). Optional
    :param iop: length of incomplete orthogonalization procedure (default: 2). Optional
    :param task1: if true, divide the result by 1/T**p. Optional

    :param device: Device to use for the computing
    :param comm: Communicator to use for MPI (only relevant for testing)

    :return: `w` - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``
    :return: `stats[0]` - number of substeps
    :return: `stats[1]` - number of rejected steps
    :return: `stats[2]` - number of Krylov steps
    :return: `stats[3]` - number of matrix exponentials
    :return: `stats[4]` - Error estimate
    :return: `stats[5]` - the Krylov size of the last substep
    """

    xp = device.xp

    tau_out = device.array(tau_out)
    u = device.array(u)
    # tol     = device.array(tol)     # That's not an array...

    ppo, n = u.shape
    p = ppo - 1

    if p == 0:
        p = 1
        # Add extra column of zeros
        u = xp.row_stack((u, xp.zeros(len(u))))

    # We only allow m to vary between mmin and mmax
    m = max(mmin, min(m_init, mmax))

    # Preallocate matrix
    V = xp.zeros((mmax + 1, n + p))
    H = xp.zeros((mmax + 1, mmax + 1))

    step = 0
    krystep = 0
    ireject = 0
    reject = 0
    exps = 0
    sgn = xp.sign(tau_out[-1])
    tau_now = 0.0
    tau_end = xp.abs(tau_out[-1])
    happy = False
    j = 0

    conv = 0.0

    numSteps = len(tau_out)

    # Initial condition
    w = xp.zeros((numSteps, n))
    w[0, :] = u[0, :].copy()

    # compute 1-norm of u
    local_normU = xp.sum(xp.abs(u[1:, :]), axis=1)
    global_normU = xp.empty_like(local_normU)
    device.synchronize()
    comm.Allreduce([local_normU, MPI.DOUBLE], [global_normU, MPI.DOUBLE])
    normU = xp.amax(global_normU)

    # Normalization factors
    if ppo > 1 and normU > 0:
        ex = xp.ceil(xp.log2(normU))
        nu = 2 ** (-ex)
        mu = 2**ex
    else:
        nu = 1.0
        mu = 1.0

    # Flip the rest of the u matrix
    u_flip = nu * xp.flipud(u[1:, :])

    # Compute an initial starting approximation for the step size
    tau = tau_end

    # Setting the safety factors and tolerance requirements
    if tau_end > 1:
        gamma = 0.2
        gamma_mmax = 0.1
    else:
        gamma = 0.9
        gamma_mmax = 0.6

    delta = 1.4

    # Used in the adaptive selection
    oldm = -1
    oldtau = math.nan
    omega = math.nan
    orderold = True
    kestold = True

    l = 0

    while tau_now < tau_end:

        # Compute necessary starting information
        if j == 0:

            V[0, :n] = w[l, :]

            # Update the last part of w
            for k in range(p - 1):
                i = p - k + 1
                V[0, n + k] = (tau_now**i) / math.factorial(i) * mu
            V[0, n + p - 1] = mu

            # Normalize initial vector (this norm is nonzero)
            local_sum = V[0, :n] @ V[0, :n]
            global_sum = xp.empty_like(local_sum)
            device.synchronize()
            comm.Allreduce([local_sum, MPI.DOUBLE], [global_sum, MPI.DOUBLE])
            beta = xp.sqrt(global_sum + V[0, n : n + p] @ V[0, n : n + p])

            # The first Krylov basis vector
            V[0, :] /= beta

        # Incomplete orthogonalization process
        while j < m:

            j = j + 1

            # Augmented matrix - vector product
            V[j, :n] = A(V[j - 1, :n]) + V[j - 1, n : n + p] @ u_flip
            V[j, n : n + p - 1] = V[j - 1, n + 1 : n + p]
            V[j, n + p - 1] = 0.0

            # Classical Gram-Schmidt
            ilow = max(0, j - iop)
            local_sum = V[ilow:j, :n] @ V[j, :n]
            global_sum = xp.empty_like(local_sum)
            device.synchronize()
            comm.Allreduce([local_sum, MPI.DOUBLE], [global_sum, MPI.DOUBLE])

            H[ilow:j, j - 1] = global_sum + V[ilow:j, n : n + p] @ V[j, n : n + p]

            V[j, :] = V[j, :] - V[ilow:j, :].T @ H[ilow:j, j - 1]

            local_sum = V[j, :n] @ V[j, :n]
            global_sum = xp.empty_like(local_sum)
            device.synchronize()
            comm.Allreduce([local_sum, MPI.DOUBLE], [global_sum, MPI.DOUBLE])
            nrm = xp.sqrt(global_sum + V[j, n : n + p] @ V[j, n : n + p])

            # Happy breakdown
            if nrm < tol:
                happy = True
                break

            H[j, j - 1] = nrm
            V[j, :] = V[j, :] / nrm
            device.synchronize()

            krystep += 1

        # To obtain the phi_1 function which is needed for error estimate
        H[0, j] = 1.0

        # Save h_j+1,j and remove it temporarily to compute the exponential of H
        nrm = H[j, j - 1].copy()
        H[j, j - 1] = 0.0

        # Compute the exponential of the augmented matrix
        F = device.xalg.linalg.expm(sgn * tau * H[: j + 1, : j + 1])
        exps += 1

        # Restore the value of H_{m+1,m}
        H[j, j - 1] = nrm

        if happy:
            # Happy breakdown wrap up
            omega = 0.0
            err = 0.0
            tau_new = min(tau_end - (tau_now + tau), tau)
            m_new = m
            happy = False

        else:
            # Local truncation error estimation
            err = xp.abs(beta * nrm * F[j - 1, j])

            # Error for this step
            oldomega = omega
            omega = tau_end * err / (tau * tol)
            omega = device.to_host(omega)

            # Estimate order
            if m == oldm and tau != oldtau and ireject >= 1:
                order = max(1.0, math.log(omega / oldomega) / device.to_host(xp.log(tau / oldtau)))
                orderold = False
            elif orderold or ireject == 0:
                orderold = True
                order = j / 4
            else:
                orderold = True

            # Estimate k
            if m != oldm and tau == oldtau and ireject >= 1:
                kest = max(1.1, (omega / oldomega) ** (1 / (oldm - m)))
                kestold = False
            elif kestold or ireject == 0:
                kestold = True
                kest = 2
            else:
                kestold = True

            if omega > delta:
                remaining_time = tau_end - tau_now
            else:
                remaining_time = tau_end - (tau_now + tau)

            # Krylov adaptivity

            same_tau = min(remaining_time, tau)
            tau_opt = tau * (gamma / omega) ** (1 / order)
            tau_opt = min(remaining_time, max(tau / 5, min(5 * tau, tau_opt)))

            m_opt = xp.ceil(j + xp.log(omega / gamma) / xp.log(kest))
            m_opt = max(mmin, min(mmax, max(xp.floor(3 / 4 * m), min(m_opt, xp.ceil(4 / 3 * m)))))

            if j == mmax:
                if omega > delta:
                    m_new = j
                    tau_new = tau * (gamma_mmax / omega) ** (1 / order)
                    tau_new = min(tau_end - tau_now, max(tau / 5, tau_new))
                else:
                    tau_new = tau_opt
                    m_new = m
            else:
                m_new = m_opt
                tau_new = same_tau

        # Check error against target
        if omega <= delta:

            # Yep, got the required tolerance; update
            reject += ireject
            step += 1

            # Update for tau_out in the interval (tau_now, tau_now + tau)
            blownTs = 0
            nextT = tau_now + tau
            for k in range(l, numSteps):
                if xp.abs(tau_out[k]) < xp.abs(nextT):
                    blownTs += 1

            if blownTs != 0:
                # Copy current w to w we continue with
                w[l + blownTs, :] = w[l, :]

                for k in range(blownTs):
                    tauPhantom = tau_out[l + k] - tau_now
                    F2 = device.xalg.linalg.expm(sgn * tauPhantom * H[:j, :j])
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

        oldtau = tau
        tau = tau_new

        oldm = m
        m = m_new

    if task1 is True:
        for k in range(numSteps):
            w[k, :] = w[k, :] / tau_out[k]

    m_ret = m
    stats = (step, reject, krystep, exps, conv, m_ret)

    return w, stats
