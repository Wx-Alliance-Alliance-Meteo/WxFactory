import math
import numpy
import mpi4py.MPI
import scipy.linalg

from time import time

# post modern arnoldi with
# norm estimate + true 1-sync (i.e. lagged normalization)
# def pmex_ne1s(τ_out, A, u, tol = 1e-7, delta = 1.2, m_init = 1, mmax = 128, reuse_info = True, task1 = False):


def pmex_ne1s_timeinit(τ_out, A, u, tol=1e-7, delta=1.2, m_init=10, mmin=10, mmax=128, reuse_info=False, task1=False):

    rank = mpi4py.MPI.COMM_WORLD.Get_rank()
    size = mpi4py.MPI.COMM_WORLD.Get_size()

    start_init = time()

    ppo, n = u.shape
    p = ppo - 1

    if p == 0:
        p = 1
        # Add extra column of zeros
        u = numpy.row_stack((u, numpy.zeros(len(u))))

    step = 0
    krystep = 0
    ireject = 0
    reject = 0
    exps = 0
    sgn = numpy.sign(τ_out[-1])
    τ_now = 0.0
    τ_end = abs(τ_out[-1])
    happy = False
    j = 0
    conv = 0.0

    numSteps = len(τ_out)

    first_accepted = True

    """
    if not hasattr(pmex_ne1s, "static_mem") or reuse_info is False:
      pmex_ne1s.static_mem = True
      pmex_ne1s.suggested_step = τ_end 
      pmex_ne1s.suggested_m = mmax
      m_init = 12
      m_opt = 1

    else:
      m_init = pmex_ne1s.suggested_m
    """

    # We only allow m to vary between mmin and mmax
    # mmin = 1
    m = max(mmin, min(m_init, mmax))

    # Preallocate matrix
    V = numpy.zeros((mmax + 1, n + p))
    H = numpy.zeros((mmax + 1, mmax + 1))
    Minv = numpy.eye(mmax)
    M = numpy.eye(mmax)
    N = numpy.zeros([mmax, mmax])

    # Initial condition
    w = numpy.zeros((numSteps, n))
    w[0, :] = u[0, :].copy()

    # compute the 1-norm of u
    local_nrmU = numpy.sum(abs(u[1:, :]), axis=1)
    global_normU = numpy.empty_like(local_nrmU)
    mpi4py.MPI.COMM_WORLD.Allreduce([local_nrmU, mpi4py.MPI.DOUBLE], [global_normU, mpi4py.MPI.DOUBLE])
    normU = numpy.amax(global_normU)

    # Normalization factors
    if ppo > 1 and normU > 0:
        ex = math.ceil(math.log2(normU))
        nu = 2 ** (-ex)
        mu = 2 ** (ex)
    else:
        nu = 1.0
        mu = 1.0

    # Flip the rest of the u matrix
    u_flip = nu * numpy.flipud(u[1:, :])

    # Compute and initial starting approximation for the step size
    # τ = min(pmex_ne1s.suggested_step, τ_end)

    # follow same as kiops
    τ = τ_end

    # Setting the safety factors and tolerance requirements
    if τ_end > 1:
        γ = 0.2
        γ_mmax = 0.1
    else:
        γ = 0.9
        γ_mmax = 0.6

    # Used in the adaptive selection
    oldm = -1
    oldτ = math.nan
    ω = math.nan
    kestold = True
    same_τ = None
    reached_mmax = False  # if reached max krylov size, do not normalize final vec
    prev_normalized = False  # if previous vector is normalized, skip certain parts

    l = 0

    end_init = time() - start_init

    #---print cost for initialization---
    if rank == 0:
      file_name = "results_tanya/pmex_timings_init_n" + str(size) + ".txt"
      with open(file_name, 'a') as gg:
        gg.write('{} \n'.format(end_init))
   #--------------------------------------

    while τ_now < τ_end:

        # Compute necessary starting information
        if j == 0:

            H[:, :] = 0.0

            V[0, 0:n] = w[l, :]

            # Update the last part of w
            for k in range(p - 1):
                i = p - k + 1
                V[j, n + k] = (τ_now**i) / math.factorial(i) * mu
            V[j, n + p - 1] = mu

        # keep track of previous norm estimate to scale at next iteration
        prev_nrm_est = 0.0

        # full orthogonalization process
        while j < m:

            j = j + 1

            # 1. Augmented matrix - vector product
            V[j, 0:n] = A(V[j - 1, 0:n]) + V[j - 1, n : n + p] @ u_flip
            V[j, n : n + p - 1] = V[j - 1, n + 1 : n + p]
            V[j, -1] = 0.0

            # 2. if j>1, then re-scale back up by the norm estimate
            # in order to compute true norm
            # do not rescale if we are at the first iteration when we increase
            # the size. The previous vector is normalized with the true norm
            if j > 1 and (not prev_normalized):
                V[j - 1 : j + 1, :] *= prev_nrm_est

            # 3. compute terms needed T matrix
            local_vec = V[0 : j + 1, :] @ V[j - 1 : j + 1, :].T
            global_vec = numpy.empty_like(local_vec)
            mpi4py.MPI.COMM_WORLD.Allreduce([local_vec, mpi4py.MPI.DOUBLE], [global_vec, mpi4py.MPI.DOUBLE])

            # 4. compute norm of previous vector
            nrm = numpy.sqrt(global_vec[j - 1, 0])

            if j == 1:
                β = nrm

            # 5. if the previous vector is NOT normalized, then scale for arnoldi
            if not prev_normalized:

                # 4a. scale for arnoldi
                V[j - 1 : j + 1, :] /= nrm
                global_vec[:, 1] /= nrm
                global_vec[j - 1 : j + 1, 1] /= nrm  # last two elements are scaled twice
                global_vec[0 : j - 1, 0] /= nrm

                # only if j>1 and if the previous vector in not normalized
                # whichis why it's not included in part 6a with T
                if j > 1:
                    H[j - 1, j - 2] = nrm

            # 6. Projection with 2-step Gauss-Sidel to the orthogonal complement
            # Note: this is done in two steps. (1) matvec and (2) a lower
            # triangular solve
            # 6a. here we set the values for matrix M, Minv, N
            if j > 1:
                M[j - 1, 0 : j - 1] = global_vec[0 : j - 1, 0]
                N[0 : j - 1, j - 1] = -global_vec[0 : j - 1, 0]
                Minv[j - 1, 0 : j - 1] = -global_vec[0 : j - 1, 0].T @ Minv[0 : j - 1, 0 : j - 1]

            # 6b. part 1: the mat-vec
            rhs = (numpy.eye(j) + numpy.matmul(N[0:j, 0:j], Minv[0:j, 0:j])) @ global_vec[0:j, 1]

            # 6c. part 2: the lower triangular solve
            sol = scipy.linalg.solve_triangular(
                M[0:j, 0:j], rhs, unit_diagonal=True, check_finite=False, overwrite_b=True
            )

            # 7. Orthogonalize
            V[j, :] -= sol @ V[0:j, :]

            # 8. set values for Hessenberg matrix
            H[0:j, j - 1] = sol

            # 9. compute norm estimate of current vector
            # do not compute it for last element because the norm will
            # be computed using communication
            if j < m:

                # 10. compute norm estimate with quad precision
                sum_vec = numpy.array(global_vec[0:j, 1], numpy.float128) ** 2
                sum_sqrd = numpy.sum(sum_vec)
                # sum_sqrd = sum(global_vec[0:j,1]**2)

                if global_vec[-1, 1] < sum_sqrd:
                    # compute true norm
                    local_sum = V[j, :] @ V[j, :]
                    global_sum = numpy.empty_like(local_sum)
                    mpi4py.MPI.COMM_WORLD.Allreduce([local_sum, mpi4py.MPI.DOUBLE], [global_sum, mpi4py.MPI.DOUBLE])
                    curr_nrm = math.sqrt(global_sum)

                else:
                    curr_nrm = numpy.array(numpy.sqrt(global_vec[-1, 1] - sum_sqrd), numpy.float64)

                # 11. Happy breakdown
                if curr_nrm < tol:
                    happy = True
                    break

                # 12. scale by norm estimate
                V[j, :] /= curr_nrm
                prev_nrm_est = curr_nrm  # save nrm estimate to re-scale up at next step

            # this flag keeps track if the previous vector is
            # normalized with the true norm. important when we
            # increase m, as we can skip "scale for Arnoldi" sections
            prev_normalized = False
            krystep += 1

        # ---end of while loop---

        # normalize final vector
        if not reached_mmax:
            local_sum = V[m, :] @ V[m, :]
            global_sum = numpy.empty_like(local_sum)
            mpi4py.MPI.COMM_WORLD.Allreduce([local_sum, mpi4py.MPI.DOUBLE], [global_sum, mpi4py.MPI.DOUBLE])
            finalNrm = numpy.sqrt(global_sum)

            V[m, :] /= finalNrm
            H[m, m - 1] = finalNrm

            prev_normalized = True  # the previous vector is normalized; skip scale for Arnoldi parts

        # To obtain the phi_1 function which is needed for error estimate
        H[0, j] = 1.0

        # Save h_j+1,j and remove it temporarily to compute the exponential of H
        nrm = H[j, j - 1]
        H[j, j - 1] = 0.0

        # Compute the exponential of the augmented matrix
        F_half = scipy.linalg.expm(sgn * 0.5 * τ * H[0 : j + 1, 0 : j + 1])
        F = F_half @ F_half
        exps += 1

        # Restore the value of H_{m+1,m}
        H[j, j - 1] = nrm

        if happy is True:
            # Happy breakdown wrap up
            ω = 0.0
            err = 0.0
            τ_new = min(τ_end - (τ_now + τ), τ)
            m_new = m
            happy = False

        else:

            # Local truncation error estimation
            err_half = abs(β * nrm * F_half[j - 1, j])
            err = abs(β * nrm * F[j - 1, j])

            # Error for this step
            oldω = ω
            ω = τ_end * err / (τ * tol)

            # Estimate order
            order = math.log(err / err_half) / math.log(2)

            # Estimate k
            if m != oldm and τ == oldτ and ireject >= 1:
                kest = max(1.1, (ω / oldω) ** (1 / (oldm - m)))
                kestold = False
            elif kestold is True or ireject == 0:
                kest = 2
                kestold = True
            else:
                kestold = True

            if ω > delta:
                remaining_time = τ_end - τ_now
            else:
                remaining_time = τ_end - (τ_now + τ)

            # Krylov adaptivity
            same_τ = min(remaining_time, τ)

            τ_opt = τ * (γ / ω) ** (1 / order)
            τ_opt = min(remaining_time, max(τ / 5, min(5 * τ, τ_opt)))

            m_opt = math.ceil(j + math.log(ω / γ) / math.log(kest))
            m_opt = max(mmin, min(mmax, max(math.floor(3 / 4 * m), min(m_opt, math.ceil(4 / 3 * m)))))

            if j == mmax:
                reached_mmax = True
                if ω > delta:
                    m_new = j
                    τ_new = τ * (γ_mmax / ω) ** (1 / order)
                    τ_new = min(τ_end - τ_now, max(τ / 5, τ_new))
                else:
                    τ_new = τ_opt
                    m_new = m
            else:
                if same_τ < τ:
                    m_new = m  # We reduced tau to avoid small step size. Then keep m constant.
                else:
                    m_new = m_opt
                τ_new = same_τ

        # Check error against target
        if ω <= delta:
            # Yep, got the required tolerance; update
            reject += ireject
            step += 1

            #if first_accepted:
            #    pmex_ne1s.suggested_step = min(pmex_ne1s.suggested_step, τ) 
            #    pmex_ne1s.suggested_m    = min(pmex_ne1s.suggested_m, m_opt)
            #    first_accepted = False

            # Udate for τ_out in the interval (τ_now, τ_now + τ)
            blownTs = 0
            nextT = τ_now + τ
            for k in range(l, numSteps):
                if abs(τ_out[k]) < abs(nextT):
                    blownTs += 1

            if blownTs != 0:
                # Copy current w to w we continue with.
                w[l + blownTs, :] = w[l, :].copy()

                for k in range(blownTs):
                    τPhantom = τ_out[l + k] - τ_now
                    F2 = scipy.linalg.expm(sgn * τPhantom * H[0:j, :j])
                    w[l + k, :] = β * F2[:j, 0] @ V[:j, :n]

                # Advance l.
                l += blownTs

            # Using the standard scheme
            w[l, :] = β * F[:j, 0] @ V[:j, :n]

            # Update τ_out
            τ_now += τ

            j = 0
            ireject = 0
            reached_mmax = False
            prev_normalized = False

            conv += err

        else:
            # Nope, try again
            ireject += 1

            # Restore the original matrix
            H[0, j] = 0.0

        oldτ = τ
        τ = τ_new

        oldm = m
        m = m_new

    if task1 is True:
        for k in range(numSteps):
            w[k, :] = w[k, :] / τ_out[k]

    m_ret = m

    stats = (step, reject, krystep, exps, conv, m_ret)

    return w, stats
