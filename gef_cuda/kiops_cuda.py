import math
import numpy as np
import cupy as cp
import cupyx as cx
from mpi4py import MPI
import cu_utils.linalg

from typing import Callable
from numpy.typing import NDArray

# 
# In the original kiops function, the array "V" is absolutely massive,
# usually too large to reside on GPU for any moderately-sized problem.
# This version of kiops stores the "V" array on the CPU,
# but stores just the most recently computed rows of "V" on the GPU
# and manually transfers them to the right place on a separate CUDA stream
# when profiling has indicated that the GPU is busy.
# This is sufficient to keep most computations on the GPU,
# although some must by necessity be done on the CPU.
# 

def kiops_cuda(tau_out: NDArray[cp.float64], A: Callable[[NDArray[cp.float64]], NDArray[cp.float64]], u: NDArray[cp.float64],
               tol: float = 1e-7, m_init: int = 10, mmin: int = 10, mmax: int = 128,
               iop: int = 2, task1: bool = False) -> tuple[NDArray[np.float64], tuple]:
    """ kiops(tstops, A, u; kwargs...) -> (w, stats)

    Evaluate a linear combinaton of the ``φ`` functions evaluated at ``tA`` acting on
    vectors from ``u``, that is

    ```math
    w(i) = φ_0(t[i] A) u[0, :] + φ_1(t[i] A) u[1, :] + φ_2(t[i] A) u[2, :] + ...
    ```

    The size of the Krylov subspace is changed dynamically during the integration.
    The Krylov subspace is computed using the incomplete orthogonalization method.

    Arguments:
    - `τ_out`    - Array of `τ_out`
    - `A`        - the matrix argument of the ``φ`` functions
    - `u`        - the matrix with rows representing the vectors to be multiplied by the ``φ`` functions

    Optional arguments:
    - `tol`      - the convergence tolerance required (default: 1e-7)
    - `mmin`, `mmax` - let the Krylov size vary between mmin and mmax (default: 10, 128)
    - `m`        - an estimate of the appropriate Krylov size (default: mmin)
    - `iop`      - length of incomplete orthogonalization procedure (default: 2)
    - `task1`     - if true, divide the result by 1/T**p

    Returns:
    - `w`      - the linear combination of the ``φ`` functions evaluated at ``tA`` acting on the vectors from ``u``
    - `stats[0]` - number of substeps
    - `stats[1]` - number of rejected steps
    - `stats[2]` - number of Krylov steps
    - `stats[3]` - number of matrix exponentials
    - `stats[4]` - Error estimate
    - `stats[5]` - the Krylov size of the last substep

    `n` is the size of the original problem
    `p` is the highest index of the ``φ`` functions

    References:
    * Gaudreault, S., Rainwater, G. and Tokman, M., 2018. KIOPS: A fast adaptive Krylov subspace solver for exponential integrators. Journal of Computational Physics. Based on the PHIPM and EXPMVP codes (http://www1.maths.leeds.ac.uk/~jitse/software.html). https://gitlab.com/stephane.gaudreault/kiops.
    * Niesen, J. and Wright, W.M., 2011. A Krylov subspace method for option pricing. SSRN 1799124
    * Niesen, J. and Wright, W.M., 2012. Algorithm 919: A Krylov subspace algorithm for evaluating the ``φ``-functions appearing in exponential integrators. ACM Transactions on Mathematical Software (TOMS), 38(3), p.22
    """
    
    tau_out = cp.asarray(tau_out, dtype=cp.float64)
    u = cp.asarray(u)
    tol = cp.asarray(tol)

    ppo, n = u.shape
    p = ppo - 1

    if p == 0:
        p = 1
        # Add extra column of zeros
        u = cp.row_stack((u, cp.zeros(len(u))))
    
    # We only allow m to vary between mmin and mmax
    m = max(mmin, min(m_init, mmax))

    # Preallocate matrix
    V = np.zeros((mmax + 1, n + p))
    # Vg: gpu copy of relevant rows of V
    Vg_slots = iop + 2
    Vg = cp.empty((Vg_slots, n + p))
    # V_buf is a pagelocked receiver for Vg
    V_buf = cx.empty_pinned((3, max(n + p, m + 1)))
    H = cp.zeros((mmax + 1, mmax + 1))

    step     = 0
    krystep  = 0
    ireject  = 0
    reject   = 0
    exps     = 0
    sgn      = cp.sign(tau_out[-1])
    tau_now  = 0.0
    tau_end  = cp.abs(tau_out[-1])
    happy    = False
    j        = 0

    conv     = 0.0

    numSteps = len(tau_out)

    # Initial condition
    w = cp.zeros((numSteps, n))
    w[0, :] = u[0, :]

    # compute 1-norm of u
    local_normU = cp.sum(cp.abs(u[1:, :]), axis=1)
    global_normU = cp.empty_like(local_normU)
    cp.cuda.get_current_stream().synchronize()
    MPI.COMM_WORLD.Allreduce([local_normU, MPI.DOUBLE], [global_normU, MPI.DOUBLE])
    normU = cp.amax(global_normU)

    # Normalization factors
    if ppo > 1 and normU > 0:
        ex = cp.ceil(cp.log2(normU))
        nu = 2 ** -ex
        mu = 2 **  ex
    else:
        nu = 1.0
        mu = 1.0

    # Flip the rest of the u matrix
    u_flip = nu * cp.flipud(u[1:, :])

    # Compute an initial starting approximation for the step size
    tau = tau_end

    # Setting the safety factors are tolerance requirements
    if tau_end > 1:
        gamma = 0.2
        gamma_mmax = 0.1
    else:
        gamma = 0.9
        gamma_mmax = 0.6

    delta = 1.4

    # Used in the adaptive selection
    oldm = -1; oldtau = math.nan; omega = math.nan
    orderold = True; kestold = True

    l = 0

    send_stream = cp.cuda.Stream()

    while tau_now < tau_end:

        # Compute necessary starting information
        if j == 0:

            Vg[0, :n] = w[l, :]

            # Update the last part of w
            for k in range(p - 1):
                i = p - k + 1
                Vg[0, n + k] = (tau_now ** i) / math.factorial(i) * mu
            Vg[0, -1] = mu

            # Normalize initial vector (this norm is nonzero)
            local_sum = Vg[0, :n] @ Vg[0, :n]
            global_sum = cp.empty_like(local_sum)
            cp.cuda.get_current_stream().synchronize()
            MPI.COMM_WORLD.Allreduce([local_sum, MPI.DOUBLE], [global_sum, MPI.DOUBLE])
            beta = cp.sqrt(global_sum + Vg[0, n:n + p] @ Vg[0, n:n + p])

            # The first Krylov basis vector
            Vg[0, :] /= beta

            # async transfer Vg[0] to host
            with send_stream:
                Vg[0, :].get(out=V_buf[0, :n + p])

        # Incomplete orthogonalization process
        while j < m:

            j = j + 1
            # this loop's vector will occupy Vg[slot]
            slot = j % Vg_slots
            # the vector will be moved to cpu in V_buf[parity]
            parity = j % 2

            # Augmented matrix - vector product
            Vg[slot, :n] = A(Vg[slot - 1, :n]) + Vg[slot - 1, n:n + p] @ u_flip

            Vg[slot, n:n + p - 1] = Vg[slot - 1, n + 1:n + p]
            Vg[slot, n + p - 1] = 0.0

            # Classical Gram-Schmidt
            ilow = max(0, j - iop)
            low_slots = [x % Vg_slots for x in range(ilow, j)]
            local_sum = Vg[low_slots, :n] @ Vg[slot, :n]
            global_sum = cp.empty_like(local_sum)
            cp.cuda.get_current_stream().synchronize()
            MPI.COMM_WORLD.Allreduce([local_sum, MPI.DOUBLE], [global_sum, MPI.DOUBLE])
            H[ilow:j, j - 1] = global_sum + Vg[low_slots, n:n + p] @ Vg[slot, n:n + p]

            Vg[slot, :n + p] = Vg[slot, :n + p] - Vg[low_slots, :n + p].T @ H[ilow:j, j - 1]

            local_sum = Vg[slot, :n] @ Vg[slot, :n]
            global_sum = cp.empty_like(local_sum)
            cp.cuda.get_current_stream().synchronize()
            MPI.COMM_WORLD.Allreduce([local_sum, MPI.DOUBLE], [global_sum, MPI.DOUBLE])
            nrm = cp.sqrt(global_sum + Vg[slot, n:n + p] @ Vg[slot, n:n + p])

            # host-to-host copy of V_buf (last iteration's vector)
            # profiling indicates that this is a good place to do this
            send_stream.synchronize()
            V[j - 1, :n + p] = V_buf[1 - parity, :n + p]

            # Happy breakdown
            if nrm < tol:
                with send_stream:
                    Vg[slot, :n + p].get(out=V_buf[parity, :n + p])
                happy = True
                break

            H[j, j - 1] = nrm
            Vg[slot, :n + p] = Vg[slot, :n + p] / nrm

            # dev-to-host copy of most recent V row
            with send_stream:
                Vg[slot, :n + p].get(out=V_buf[parity, :n + p])

            krystep += 1

        # To obtain the phi_1 function which is needed for error estimate
        H[0, j] = 1.0

        # Save h_j+1,j and remove it temporarily to compute the exponential of H
        nrm = H[j, j - 1].copy()
        H[j, j - 1] = 0.0

        # Compute the exponential of the augmented matrix
        F = cu_utils.linalg.expm(sgn * tau * H[:j + 1, :j + 1])
        exps += 1

        # while we're waiting we copy out the last V row
        send_stream.synchronize()
        V[j, :n + p] = V_buf[parity, :n + p]

        # Restore the value of H_{m+1,m}
        H[j, j - 1] = nrm

        if happy:
            # Happy breakdown wrap up
            omega = 0.
            err = 0.
            tau_new = min(tau_end - (tau_now + tau), tau)
            m_new = m
            happy = False

        else:
            # Local truncation error estimation
            err = cp.abs(beta * nrm * F[j - 1, j])

            # Error for this step
            oldomega = omega
            omega = tau_end * err / (tau * tol)
            omega = omega.get()

            # Estimate order
            if m == oldm and tau != oldtau and ireject >= 1:
                order = max(1., math.log(omega / oldomega) / cp.log(tau / oldtau).get())
                orderold = False
            elif orderold or ireject == 0:
                orderold = True
                order = j/4
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

            m_opt = cp.ceil(j + cp.log(omega / gamma) / cp.log(kest))
            m_opt = max(mmin, min(mmax, max(cp.floor(3/4 * m), min(m_opt, cp.ceil(4/3 * m)))))

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

            # We're going to need F on the cpu
            with send_stream:
                (beta * F[:j, 0]).get(out=V_buf[0, :j])

            # Yep, got the required tolerance; update
            reject += ireject
            step += 1

            # Update for tau_out in the interval (tau_now, tau_now + 1)
            blownTs = 0
            nextT = tau_now + tau
            for k in range(l, numSteps):
                if cp.abs(tau_out[k]) < cp.abs(nextT):
                    blownTs += 1

            if blownTs != 0:
                # Copy current w to w we continue with
                w[l + blownTs, :] = w[l, :]

                for k in range(blownTs):
                    tauPhantom = tau_out[l + k] - tau_now
                    F2 = cu_utils.linalg.expm(sgn * tauPhantom * H[:j, :j])
                    # Too large for gpu, and can't copy out beforehand :(
                    send_stream.synchronize()
                    np.matmul((beta * F2[:j, 0]).get(out=V_buf[1, :j]), V[:j, :n], out=V_buf[2, :n])
                    with send_stream:
                        w[l + k, :] = cp.asarray(V_buf[2, :n])

                # Advance l.
                l += blownTs

            # Using the standard scheme
            send_stream.synchronize()
            np.matmul(V_buf[0, :j], V[:j, :n], out=V_buf[1, :n])
            with send_stream:
                w[l, :] = cp.asarray(V_buf[1, :n])

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

    send_stream.synchronize()
    if task1 is True:
        for k in range(numSteps):
            w[k, :] = w[k, :] / tau_out[k]

    m_ret = m
    stats = (step, reject, krystep, exps, conv, m_ret)

    return w, stats
