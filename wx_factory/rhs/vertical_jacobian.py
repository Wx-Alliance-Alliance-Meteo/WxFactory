"""Analytic Jacobian of the vertically-stiff partition f1 (PartRosExp2).

This assembles / applies the exact derivative of the discrete f1 built in
``RHSDirecFluxReconstruction_mpi_v2.implicit`` -- the operator ``J1`` the design note derives
(eq:JacAction / eq:discJacAction). The heart of it is the pointwise vertical-flux Jacobian A3
(eq:A3G): F3 and its Jacobian are pointwise functions of the conserved state, so the action of J1 is
that pointwise 5x5 map composed with the same fixed DFR / Rusanov linear operators that f1 uses.

Component/conserved-variable order is (rho, rho*u1, rho*u2, rho*w, rho*theta).
"""

import numpy
from numpy.typing import NDArray

from ..common.definitions import (
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_w,
    idx_rho_theta,
    heat_capacity_ratio,
    p0,
    cpd,
    cvd,
    Rd,
    gravity,
)
from ..common.matmul import apply_op

_mid_k = numpy.s_[..., 1:-1, :, :, :]
_bot_k = numpy.s_[..., 0, :, :, :]
_top_k = numpy.s_[..., -1, :, :, :]


_MOM = (idx_rho_u1, idx_rho_u2, idx_rho_w)


def ad_matvec(dq, q, pressure, direction, h1d, h2d, h3d):
    """Apply the flux Jacobian A^d(q) of the direction-``direction`` flux to dq, pointwise (no sqrtG).

    ``direction`` is 0, 1, 2 for x1, x2, x3; ``h1d, h2d, h3d`` are the contravariant metric components
    h^{i,direction+1} (i=1,2,3) at the same points as ``q``. With md = the direction's momentum, u^d its
    velocity and cs2/theta = gamma p / rho_theta, the rows of A^d are

        (A^d dq)_rho = dq_md
        (A^d dq)_rui = u^d dq_rui + u^i dq_md - u^i u^d dq_rho + h^{i,d} (cs2/theta) dq_rt   (i = 1,2,3)
        (A^d dq)_rt  = u^d dq_rt + theta dq_md - u^d theta dq_rho

    (the ρu^d diagonal automatically gives 2u^d dq_md - (u^d)^2 dq_rho). For direction 2 this is A3.
    """
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_w] / rho)
    theta = q[idx_rho_theta] / rho
    md = _MOM[direction]
    ud = u[direction]
    cs2_over_theta = heat_capacity_ratio * pressure / q[idx_rho_theta]
    hd = (h1d, h2d, h3d)

    d_rho = dq[idx_rho]
    d_md = dq[md]
    d_rt = dq[idx_rho_theta]

    out = dq * 0.0  # zeros like dq (same dtype/device)
    out[idx_rho] = d_md
    for i in range(3):
        out[_MOM[i]] = ud * dq[_MOM[i]] + u[i] * d_md - u[i] * ud * d_rho + hd[i] * cs2_over_theta * d_rt
    out[idx_rho_theta] = ud * d_rt + theta * d_md - ud * theta * d_rho
    return out


def ad_matrix(q, pressure, direction, h1d, h2d, h3d, xp):
    """Build the explicit 5x5 flux Jacobian A^d(q) at every point, indexed [..., row, col] in the
    natural component order (rho, rho_u1, rho_u2, rho_w, rho_theta). Same map as :func:`ad_matvec`."""
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_w] / rho)
    theta = q[idx_rho_theta] / rho
    md = _MOM[direction]
    ud = u[direction]
    cs2_over_theta = heat_capacity_ratio * pressure / q[idx_rho_theta]
    hd = (h1d, h2d, h3d)

    A = xp.zeros(rho.shape + (5, 5), dtype=q.dtype)
    A[..., idx_rho, md] = 1.0
    for i in range(3):
        mi = _MOM[i]
        A[..., mi, mi] += ud
        A[..., mi, md] += u[i]
        A[..., mi, idx_rho] += -u[i] * ud
        A[..., mi, idx_rho_theta] += hd[i] * cs2_over_theta
    A[..., idx_rho_theta, idx_rho_theta] += ud
    A[..., idx_rho_theta, md] += theta
    A[..., idx_rho_theta, idx_rho] += -ud * theta
    return A


def a3_matvec(dq: NDArray, q: NDArray, pressure: NDArray, h13: NDArray, h23: NDArray, h33: NDArray) -> NDArray:
    """Vertical-flux Jacobian A3(q) dq (eq:A3G) -- the direction-2 case of :func:`ad_matvec`."""
    return ad_matvec(dq, q, pressure, 2, h13, h23, h33)


def _trace_metric(m, comp):
    """h^{comp,3} at the vertical interface points."""
    return m.h_contra_itf_k_new[comp, 2]


def j1_prepare(rhsobj, q):
    """Compute the frozen base of ``implicit_jvp`` once (identical for every probe of the assembly and
    for every matvec of a step). Returns COPIES of what ``implicit_jvp`` reads, so later evaluations on
    the same RHS object cannot clobber them."""
    rhsobj.implicit(q)
    return (
        rhsobj.ops,
        rhsobj.pressure.copy(),
        rhsobj.q_itf_x3.copy(),
        rhsobj.q_itf_full_x3.copy(),
        rhsobj.wflux_pres_x3.copy(),
        rhsobj.wflux_pres_itf_x3.copy(),
        rhsobj.log_p.copy(),
        rhsobj.pressure_itf_x3.copy(),
    )


def implicit_jvp(rhsobj, q: NDArray, dq: NDArray, base=None) -> NDArray:
    """Analytic action of the frozen-lambda Jacobian J1 of ``rhsobj.implicit`` on ``dq``.

    Mirrors ``implicit(q)`` with every flux replaced by its linearization: the interior vertical flux
    by sqrtG*A3(q)*dq, and the Rusanov interface flux by its frozen-lambda differential
    delta f* = sqrtG_itf * [ 1/2 (A3(q_d) dq_d + A3(q_u) dq_u) - 1/2 eig (dq_u - dq_d) ]. The trace
    extrapolation of rho and rho_theta uses the same log-exp map as ``implicit`` (linearized here); the
    momenta extrapolate linearly. eig is frozen. The wall uses an M-reflected ghost for the plain rows.
    ``base`` is the cached tuple from :func:`j1_prepare`; if omitted it is recomputed (slow).

    ``dq`` may carry an extra *batch* axis inserted right after the component axis, i.e. shape
    (nv, nb, nz, ny, nx, ns**3) instead of (nv, nz, ny, nx, ns**3). All the frozen base quantities are
    independent of the perturbation, so they simply broadcast over that axis; the result then has the
    same batched shape. This lets the block assembly probe many basis vectors per kernel launch, which
    is what actually costs time here (the assembly is launch-bound, not FLOP-bound)."""
    xp = rhsobj.device.xp
    m = rhsobj.metric

    if base is None:
        base = j1_prepare(rhsobj, q)
    ops, pressure, q_itf_x3, q_itf_full_x3, wflux_pres_x3, wflux_pres_itf_x3, log_p, pressure_itf_x3 = base
    op_extrap = ops.extrap_z
    op_dz = ops.derivative_z
    op_corr = ops.correction_DU
    n = rhsobj.geom.num_solpts**2

    # --- interior: delta f_x3 = sqrtG * A3(q) dq ---
    h13, h23, h33 = m.h_contra_new[0, 2], m.h_contra_new[1, 2], m.h_contra_new[2, 2]
    dfx3 = m.sqrtG_new * a3_matvec(dq, q, pressure, h13, h23, h33)

    # --- traces of dq (linearize the log-exp extrapolation of rho, rho_theta; momenta are linear) ---
    dq_itf = apply_op(dq, op_extrap)
    q_itf = q_itf_x3  # log-exp traces of q, as used by f1
    dq_itf[idx_rho] = q_itf[idx_rho] * apply_op(dq[idx_rho] / q[idx_rho], op_extrap)
    dq_itf[idx_rho_theta] = q_itf[idx_rho_theta] * apply_op(dq[idx_rho_theta] / q[idx_rho_theta], op_extrap)

    # --- ghost-padded full trace arrays (mid = real elements; zero-gradient ghosts as in prepare) ---
    def to_full(itf, base_full):
        # shape = itf's leading (component [+ batch]) axes + the ghost-padded trailing axes
        full = xp.zeros(itf.shape[:-4] + base_full.shape[-4:], dtype=itf.dtype)
        full[_mid_k] = itf
        full[..., 0, :, :, n:] = full[..., 1, :, :, :n]
        full[..., -1, :, :, :n] = full[..., -2, :, :, n:]
        return full

    qf = q_itf_full_x3
    dqf = to_full(dq_itf, qf)

    # Wall (u3 = 0 at the top/bottom): for the plain-row interface flux the ghost vertical momentum is
    # reflected (M = diag(1,1,1,-1,1), i.e. u3_ghost = -u3_neighbor). This makes the advective wall
    # flux cancel and leaves only the pressure term, matching f1 for rho, rho_theta, rho_u1, rho_u2.
    # The rho_w row uses the well-balanced path below with the un-reflected traces, so it is kept
    # separate. Only the ghost elements (0 and -1) are touched.
    qf_r = qf.copy()
    dqf_r = dqf.copy()
    # (indexed on the element axis -4, so this is correct with or without a batch axis)
    qf_r[idx_rho_w][_bot_k] *= -1.0
    qf_r[idx_rho_w][_top_k] *= -1.0
    dqf_r[idx_rho_w][_bot_k] *= -1.0
    dqf_r[idx_rho_w][_top_k] *= -1.0

    # pressure at q traces (for eig) and its metric
    p_itf = p0 * xp.exp((cpd / cvd) * xp.log(qf[idx_rho_theta] * (Rd / p0)))

    south = xp.s_[..., 1:, :, :, :n]
    north = xp.s_[..., :-1, :, :, n:]
    h33_itf = m.h_contra_itf_k_new[2, 2]
    sg_itf = m.sqrtG_itf_k_new

    w_d = qf[idx_rho_w][north] / qf[idx_rho][north]
    w_u = qf[idx_rho_w][south] / qf[idx_rho][south]
    eig_d = xp.abs(w_d) + xp.sqrt(h33_itf[north] * heat_capacity_ratio * p_itf[north] / qf[idx_rho][north])
    eig_u = xp.abs(w_u) + xp.sqrt(h33_itf[south] * heat_capacity_ratio * p_itf[south] / qf[idx_rho][south])
    eig = xp.maximum(eig_d, eig_u)

    # linearized advective+pressure flux via A3 at the two traces (M-reflected ghost for the wall)
    dflux_d = sg_itf[north] * a3_matvec(
        dqf_r[north], qf_r[north], p_itf[north], _trace_metric(m, 0)[north], _trace_metric(m, 1)[north], h33_itf[north]
    )
    dflux_u = sg_itf[south] * a3_matvec(
        dqf_r[south], qf_r[south], p_itf[south], _trace_metric(m, 0)[south], _trace_metric(m, 1)[south], h33_itf[south]
    )

    dfitf_full = xp.zeros_like(dqf)
    dfitf_full[north] = 0.5 * (dflux_d + dflux_u - eig * sg_itf[north] * (dqf_r[south] - dqf_r[north]))
    dfitf_full[south] = dfitf_full[north]
    dfitf = dfitf_full[_mid_k]

    # --- well-balanced rho_w Jacobian (eq:wbjac): overwrites the A3 rho_w row ---
    # base (frozen) pressure operators of the residual
    wfp = wflux_pres_x3  # sqrtG h33 at solution points (constant in q)
    w_presa_base = apply_op(wfp, op_dz)
    apply_op(wflux_pres_itf_x3, op_corr, out=w_presa_base, beta=1.0)
    logp_bdy = xp.log(pressure_itf_x3)
    w_presb_base = apply_op(log_p, op_dz)
    apply_op(logp_bdy, op_corr, out=w_presb_base, beta=1.0)
    w_presb_base = w_presb_base * wfp

    # interior deltas
    w = q[idx_rho_w] / q[idx_rho]
    dp = heat_capacity_ratio * pressure / q[idx_rho_theta] * dq[idx_rho_theta]  # delta p
    dlogp = heat_capacity_ratio / q[idx_rho_theta] * dq[idx_rho_theta]  # delta log p
    dwadv_x3 = m.sqrtG_new * (2.0 * w * dq[idx_rho_w] - w**2 * dq[idx_rho])  # delta(sqrtG w rho_w)

    # interface deltas (frozen eig for the advective part; pressure weighting for the metric part)
    p_d, p_u = p_itf[north], p_itf[south]
    rt_d, rt_u = qf[idx_rho_theta][north], qf[idx_rho_theta][south]
    dp_d = heat_capacity_ratio * p_d / rt_d * dqf[idx_rho_theta][north]
    dp_u = heat_capacity_ratio * p_u / rt_u * dqf[idx_rho_theta][south]
    drw_d, drw_u = dqf[idx_rho_w][north], dqf[idx_rho_w][south]

    dwadv_full = xp.zeros_like(dqf[idx_rho])
    dwadv_d = sg_itf[north] * (2.0 * w_d * drw_d - w_d**2 * dqf[idx_rho][north])
    dwadv_u = sg_itf[south] * (2.0 * w_u * drw_u - w_u**2 * dqf[idx_rho][south])
    dwadv_full[north] = 0.5 * (dwadv_d + dwadv_u - eig * sg_itf[north] * (drw_u - drw_d))
    dwadv_full[south] = dwadv_full[north]

    Gd = sg_itf[north] * h33_itf[north]
    Gu = sg_itf[south] * h33_itf[south]
    dwpres_full = xp.zeros_like(dqf[idx_rho])
    dwpres_full[north] = 0.5 * Gu / p_d * (dp_u - (p_u / p_d) * dp_d)
    dwpres_full[south] = 0.5 * Gd / p_u * (dp_d - (p_d / p_u) * dp_u)

    dlogp_itf = heat_capacity_ratio * dq_itf[idx_rho_theta] / q_itf[idx_rho_theta]

    # assemble delta of the well-balanced rho_w residual
    dw_df3 = apply_op(dwadv_x3, op_dz)
    apply_op(dwadv_full[_mid_k], op_corr, out=dw_df3, beta=1.0)
    dw_presa = apply_op(dwpres_full[_mid_k], op_corr)  # interior delta(sqrtG h33) = 0
    dw_presb = apply_op(dlogp, op_dz)
    apply_op(dlogp_itf, op_corr, out=dw_presb, beta=1.0)
    dw_presb = dw_presb * wfp
    drhs_w = dw_df3 + dp * (w_presa_base + w_presb_base) + pressure * (dw_presa + dw_presb)

    # --- combine: 4 plain rows via A3/B+-, then the well-balanced rho_w row; scale; gravity Jacobian ---
    out = apply_op(dfx3, op_dz)
    apply_op(dfitf, op_corr, out=out, beta=1.0)
    out[idx_rho_w] = drhs_w
    out *= -m.inv_sqrtG_new
    out[idx_rho_w] -= (
        m.inv_dzdeta_new * gravity * m.inv_sqrtG_new * ((m.sqrtG_new * dq[idx_rho]) @ ops.highfilter_k)
    )
    return out


# ---------------------------------------------------------------------------------------------------
# Block-tridiagonal assembly of J1 per vertical column (pencil).
#
# A column is a fixed horizontal position (jy, jx, h) with h the horizontal solution point inside an
# element; within the last axis of the state the vertical index is the outer factor,
# solpt = v * num_solpts**2 + h  (derivative_z = kron(diff_solpt, I_{ns^2})). The column matrix is
# block-tridiagonal over the nz vertical elements with dense blocks of size m = 5 * num_solpts, the
# block DOF ordered as (variable, vertical solution point) -> var * ns + s.
#
# The blocks are obtained by probing the (validated) action with a 3-colouring of the vertical
# elements: consecutive elements have distinct colours mod 3, so one action per (variable, vsolpt,
# colour) fills one column of the lower/diagonal/upper blocks. This is the direct-but-costly route;
# the analytic assembly of eq:lower--eq:upper / eq:wbjac is the intended optimisation.
# ---------------------------------------------------------------------------------------------------


def _col_dims(rhsobj, q):
    ns = rhsobj.geom.num_solpts
    nv, nz, ny, nx, _ = q.shape
    return ns, ns * ns, nv, nz, ny, nx, nv * ns, ny * nx * ns * ns  # ns, nh, nv, nz, ny, nx, m, ncol


def state_to_col(rhsobj, x):
    """(nv, nz, ny, nx, ns**3) -> (ncol, nz, m) with block DOF = var*ns + vsolpt."""
    xp = rhsobj.device.xp
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, x)
    x6 = x.reshape(nv, nz, ny, nx, ns, nh)
    p = xp.transpose(x6, (2, 3, 5, 1, 0, 4))  # (ny, nx, nh, nz, nv, ns)
    return p.reshape(ncol, nz, m)


def col_to_state(rhsobj, xc, ref):
    """Inverse of :func:`state_to_col` (``ref`` supplies the target shape)."""
    xp = rhsobj.device.xp
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, ref)
    p = xc.reshape(ny, nx, nh, nz, nv, ns)
    x6 = xp.transpose(p, (4, 3, 0, 1, 5, 2))  # (nv, nz, ny, nx, ns, nh)
    return x6.reshape(ref.shape)


def batched_state_to_col(rhsobj, x, ref):
    """Batched :func:`state_to_col`: (nv, nb, nz, ny, nx, ns**3) -> (nb, ncol, nz, m). ``ref`` is an
    unbatched state supplying the grid dimensions."""
    xp = rhsobj.device.xp
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, ref)
    nb = x.shape[1]
    x7 = x.reshape(nv, nb, nz, ny, nx, ns, nh)
    p = xp.transpose(x7, (1, 3, 4, 6, 2, 0, 5))  # (nb, ny, nx, nh, nz, nv, ns)
    return p.reshape(nb, ncol, nz, m)


def assemble_j1_blocks(rhsobj, q, batch=3):
    """Assemble the block-tridiagonal Jacobian J1 per column. Returns (L, A, U), each of shape
    (ncol, nz, m, m): the sub-, main- and super-diagonal blocks, so that
    (J1 dq)_e = L_e dq_{e-1} + A_e dq_e + U_e dq_{e+1} per column.

    The 3-colouring needs nv*ns = 15 probes per colour, 45 in all. Issuing those one at a time is
    launch-bound rather than FLOP-bound, so they are instead run ``batch`` at a time through the
    batch-aware :func:`implicit_jvp`, and the results are scattered into the blocks with one fancy
    -indexed assignment per (colour, chunk, diagonal) instead of one per (probe, element). ``batch``
    trades kernel launches against peak memory -- the working set of ``implicit_jvp`` scales with it,
    and 6 ranks share one GPU here. Measured on the DCMIP 2-1 grid (20^3 elements, ns = 3, 6 ranks on
    one 8 GB GPU): batch = 1 / 3 / 5 / 9 gives 0.45 / 0.31 / 0.31 / 0.34 s, and batch = 15 runs out of
    memory. The gain saturates at 3 because the per-probe kernels are already bandwidth-bound at full
    grid size, so batching only recovers the fixed launch overhead. Results are bit-identical to the
    unbatched probes for every batch size."""
    xp = rhsobj.device.xp
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, q)
    L = xp.zeros((ncol, nz, m, m), dtype=q.dtype)
    A = xp.zeros((ncol, nz, m, m), dtype=q.dtype)
    U = xp.zeros((ncol, nz, m, m), dtype=q.dtype)

    base = j1_prepare(rhsobj, q)  # frozen once; every probe reuses it
    probes = [(v, s) for v in range(nv) for s in range(ns)]  # block-column j = v*ns + s, in order
    nj = len(probes)
    batch = max(1, min(batch, nj))

    for c in range(3):
        # for this colour each element falls in exactly one of the three diagonals (e%3, (e-1)%3 and
        # (e+1)%3 are distinct mod 3), so the scatters below never overlap
        eA = xp.asarray([e for e in range(nz) if e % 3 == c])
        eL = xp.asarray([e for e in range(1, nz) if (e - 1) % 3 == c])
        eU = xp.asarray([e for e in range(nz - 1) if (e + 1) % 3 == c])
        for start in range(0, nj, batch):
            chunk = probes[start : start + batch]
            nb = len(chunk)
            dq = xp.zeros((nv, nb, nz, ny, nx, ns * nh), dtype=q.dtype)
            for p, (v, s) in enumerate(chunk):
                dq[v, p, c::3, :, :, s * nh : (s + 1) * nh] = 1.0
            Pc = batched_state_to_col(rhsobj, implicit_jvp(rhsobj, q, dq, base), q)  # (nb, ncol, nz, m)
            js = slice(start, start + nb)
            for blk, es in ((A, eA), (L, eL), (U, eU)):
                if es.size:  # (ncol, |es|, m, nb) <- (nb, ncol, |es|, m)
                    blk[:, es, :, js] = xp.transpose(Pc[:, :, es, :], (1, 2, 3, 0))
    return L, A, U


# ---------------------------------------------------------------------------------------------------
# Direct (analytic) assembly of the same blocks -- no probing.
#
# Every operator in ``implicit_jvp`` acts on the vertical solution-point index alone (the horizontal
# index rides along untouched): with solpt = s*ns**2 + h, ``derivative_z``, ``extrap_z``,
# ``correction_DU`` and ``highfilter_k`` all reduce to out[s] = sum_s' M[s,s'] in[s'] for
# M = diff_solpt (Dz), extrap_down/up (eL/eR), correction[:,0]/[:,1] (dL/dR) and highfilter (HF)
# respectively. So the whole column Jacobian is a sum of Kronecker products of those 1D reference
# matrices with pointwise 5x5 blocks -- eq:lower / eq:diagblk / eq:upper of the design note -- and can
# be written down directly instead of being probed 45 times.
#
# Interface bookkeeping. The padded trace arrays index elements as p = 0 (bottom ghost), p = 1..nz
# (real element p-1), p = nz+1 (top ghost); the last axis is (down face | up face). ``implicit_jvp``
# slices north = [..., :-1, :, :, n:] and south = [..., 1:, :, :, :n], so interface *slot* t (t = 0..nz)
# pairs the down side p = t up-face with the up side p = t+1 down-face:
#     t = 0                : bottom wall  (both sides are element 0's down trace, ghost side reflected)
#     t = 1..nz-1          : interior interface between elements t-1 and t
#     t = nz               : top wall     (both sides are element nz-1's up trace, ghost side reflected)
# Element e then reads slot t = e at its down face and slot t = e+1 at its up face.
# ---------------------------------------------------------------------------------------------------


def _g2c(xp, arr, ns, nh, ny, nx):
    """Solution-point grid array (nz, ny, nx, ns*nh) -> column layout (ncol, nz, ns)."""
    nz = arr.shape[0]
    a = arr.reshape(nz, ny, nx, ns, nh)
    return xp.transpose(a, (1, 2, 4, 0, 3)).reshape(ny * nx * nh, nz, ns)


def _i2c(xp, arr, nh, ny, nx):
    """Scalar interface array (np, ny, nx, 2*nh) -> (ncol, np, 2) with face 0 = down, 1 = up."""
    npd = arr.shape[0]
    a = arr.reshape(npd, ny, nx, 2, nh)
    return xp.transpose(a, (1, 2, 4, 0, 3)).reshape(ny * nx * nh, npd, 2)


def _i5c(xp, arr, nh, ny, nx):
    """State interface array (nv, np, ny, nx, 2*nh) -> (ncol, np, 2, nv)."""
    nv, npd = arr.shape[0], arr.shape[1]
    a = arr.reshape(nv, npd, ny, nx, 2, nh)
    return xp.transpose(a, (2, 3, 5, 1, 4, 0)).reshape(ny * nx * nh, npd, 2, nv)


def assemble_j1_blocks_analytic(rhsobj, q):
    """Assemble the block-tridiagonal J1 directly from its closed form -- same (L, A, U) contract as
    :func:`assemble_j1_blocks`, but built from the 1D reference operators and pointwise 5x5 blocks
    instead of 45 probes of :func:`implicit_jvp`.

    Term by term this mirrors ``implicit_jvp`` exactly: the interior vertical flux (Dz x sqrtG A3), the
    Rusanov interface fluxes (dL/dR x B+/B- acting on the linearized log-exp traces), the M-reflected
    walls folded into the first and last diagonal blocks, the well-balanced rho_w row (eq:wbjac /
    eq:wbGjac) replacing the plain rho_w row, the -1/sqrtG row scaling and the filtered gravity term."""
    xp = rhsobj.device.xp
    met = rhsobj.metric
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, q)
    gam = heat_capacity_ratio
    rw, rt, rr = idx_rho_w, idx_rho_theta, idx_rho
    dt_ = q.dtype

    base = j1_prepare(rhsobj, q)
    ops, pressure, q_itf_x3, qf, wflux_pres_x3, wflux_pres_itf_x3, log_p, pressure_itf_x3 = base

    # --- 1D reference operators (see the module note above for the reduction) ---
    Dz = ops.diff_solpt  # D_int
    eLv, eRv = ops.extrap_down, ops.extrap_up  # e_L, e_R
    dLv, dRv = ops.correction[:, 0], ops.correction[:, 1]  # d_L tilde, d_R tilde
    HF = ops.highfilter
    eye_s = xp.eye(ns, dtype=dt_)

    def g2c(a):
        return _g2c(xp, a, ns, nh, ny, nx)

    # --- solution-point quantities in column layout ---
    qc = state_to_col(rhsobj, q).reshape(ncol, nz, nv, ns)
    pc = g2c(pressure)
    sgc, isgc, idzc = g2c(met.sqrtG_new), g2c(met.inv_sqrtG_new), g2c(met.inv_dzdeta_new)
    A3s = ad_matrix(  # (ncol, nz, ns, 5, 5)
        xp.transpose(qc, (2, 0, 1, 3)),
        pc,
        2,
        g2c(met.h_contra_new[0, 2]),
        g2c(met.h_contra_new[1, 2]),
        g2c(met.h_contra_new[2, 2]),
        xp,
    )

    # --- trace quantities in column layout (padded: nz + 2 elements, 2 faces) ---
    qfc = _i5c(xp, qf, nh, ny, nx)  # (ncol, nz+2, 2, nv)
    sgfc = _i2c(xp, met.sqrtG_itf_k_new, nh, ny, nx)
    h33fc = _i2c(xp, _trace_metric(met, 2), nh, ny, nx)
    p_itf_c = p0 * xp.exp((cpd / cvd) * xp.log(qfc[..., rt] * (Rd / p0)))
    qfc_r = qfc.copy()  # M-reflected ghosts, for the plain-row interface flux only
    qfc_r[:, 0, :, rw] *= -1.0
    qfc_r[:, -1, :, rw] *= -1.0
    A3f = ad_matrix(  # (ncol, nz+2, 2, 5, 5)
        xp.transpose(qfc_r, (3, 0, 1, 2)),
        p_itf_c,
        2,
        _i2c(xp, _trace_metric(met, 0), nh, ny, nx),
        _i2c(xp, _trace_metric(met, 1), nh, ny, nx),
        h33fc,
        xp,
    )

    # --- per-slot (t = 0..nz) interface data: down side = (p=t, up face), up side = (p=t+1, down) ---
    dn, up = numpy.s_[:, : nz + 1, 1], numpy.s_[:, 1:, 0]
    sg_n, sg_s = sgfc[dn], sgfc[up]
    rho_d, rho_u = qfc[dn][..., rr], qfc[up][..., rr]
    w_d, w_u = qfc[dn][..., rw] / rho_d, qfc[up][..., rw] / rho_u  # eig/wb use the UNreflected traces
    p_d, p_u = p_itf_c[dn], p_itf_c[up]
    rt_d, rt_u = qfc[dn][..., rt], qfc[up][..., rt]
    eig = xp.maximum(
        xp.abs(w_d) + xp.sqrt(h33fc[dn] * gam * p_d / rho_d),
        xp.abs(w_u) + xp.sqrt(h33fc[up] * gam * p_u / rho_u),
    )
    I5 = xp.eye(nv, dtype=dt_).reshape(1, 1, nv, nv)
    lam = (eig * sg_n)[..., None, None]
    Bp = 0.5 * (sg_n[..., None, None] * A3f[dn] + lam * I5)  # acts on the down-side trace
    Bm = 0.5 * (sg_s[..., None, None] * A3f[up] - lam * I5)  # acts on the up-side trace

    # --- linearized trace extrapolation: dq_itf[e, v, face] = sum_s E[e, v, s] dq[e, v, s].
    #     rho and rho_theta go through the log-exp map, so their row is scaled by q_itf / q. ---
    qitfc = _i5c(xp, q_itf_x3, nh, ny, nx)  # (ncol, nz, 2, nv)
    EL = xp.zeros((ncol, nz, nv, ns), dtype=dt_) + eLv
    ER = xp.zeros((ncol, nz, nv, ns), dtype=dt_) + eRv
    for var in (rr, rt):
        EL[:, :, var, :] = eLv * (qitfc[:, :, 0, var][..., None] / qc[:, :, var, :])
        ER[:, :, var, :] = eRv * (qitfc[:, :, 1, var][..., None] / qc[:, :, var, :])

    # ================= plain rows =================
    shp = (ncol, nz, nv, ns, nv, ns)  # [column, element, row var, row solpt, col var, col solpt]
    L = xp.zeros(shp, dtype=dt_)
    A = xp.zeros(shp, dtype=dt_)
    U = xp.zeros(shp, dtype=dt_)

    # interior volume term: Dz x (sqrtG A3)
    A += xp.einsum("os,ces,cesij->ceiojs", Dz, sgc, A3s)

    # interior interfaces: slot t = 1..nz-1 feeds element t (down face, dL) and element t-1 (up, dR)
    IT, ED, EU = slice(1, nz), slice(1, nz), slice(0, nz - 1)
    L[:, ED] += xp.einsum("o,ceij,cejs->ceiojs", dLv, Bp[:, IT], ER[:, EU])
    A[:, ED] += xp.einsum("o,ceij,cejs->ceiojs", dLv, Bm[:, IT], EL[:, ED])
    A[:, EU] += xp.einsum("o,ceij,cejs->ceiojs", dRv, Bp[:, IT], ER[:, EU])
    U[:, EU] += xp.einsum("o,ceij,cejs->ceiojs", dRv, Bm[:, IT], EL[:, ED])

    # walls: both sides of the slot are the same trace, the ghost side reflected by M -> self-coupling
    Mv = xp.ones(nv, dtype=dt_)
    Mv[rw] = -1.0
    A[:, 0] += xp.einsum("o,cij,cjs->ciojs", dLv, Bp[:, 0] * Mv + Bm[:, 0], EL[:, 0])
    A[:, nz - 1] += xp.einsum("o,cij,cjs->ciojs", dRv, Bp[:, nz] + Bm[:, nz] * Mv, ER[:, nz - 1])

    # ================= well-balanced rho_w row (replaces the plain one) =================
    L[:, :, rw] = 0.0
    A[:, :, rw] = 0.0
    U[:, :, rw] = 0.0

    rhoc, rtc = qc[:, :, rr, :], qc[:, :, rt, :]
    wc = qc[:, :, rw, :] / rhoc

    # advective interior: Dz[sqrtG (2 w drho_w - w^2 drho)]
    A[:, :, rw, :, rw, :] += xp.einsum("os,ces->ceos", Dz, sgc * 2.0 * wc)
    A[:, :, rw, :, rr, :] += xp.einsum("os,ces->ceos", Dz, -sgc * wc**2)

    # advective interfaces: same Rusanov slots, but only the (rho, rho_w) columns and no reflection
    cwp = xp.zeros((ncol, nz + 1, nv), dtype=dt_)
    cwm = xp.zeros((ncol, nz + 1, nv), dtype=dt_)
    cwp[..., rw] = 0.5 * (2.0 * sg_n * w_d + eig * sg_n)
    cwp[..., rr] = -0.5 * sg_n * w_d**2
    cwm[..., rw] = 0.5 * (2.0 * sg_s * w_u - eig * sg_n)
    cwm[..., rr] = -0.5 * sg_s * w_u**2
    L[:, ED, rw] += xp.einsum("o,cej,cejs->ceojs", dLv, cwp[:, IT], ER[:, EU])
    A[:, ED, rw] += xp.einsum("o,cej,cejs->ceojs", dLv, cwm[:, IT], EL[:, ED])
    A[:, EU, rw] += xp.einsum("o,cej,cejs->ceojs", dRv, cwp[:, IT], ER[:, EU])
    U[:, EU, rw] += xp.einsum("o,cej,cejs->ceojs", dRv, cwm[:, IT], EL[:, ED])
    A[:, 0, rw] += xp.einsum("o,cj,cjs->cojs", dLv, cwp[:, 0] + cwm[:, 0], EL[:, 0])
    A[:, nz - 1, rw] += xp.einsum("o,cj,cjs->cojs", dRv, cwp[:, nz] + cwm[:, nz], ER[:, nz - 1])

    # dp * (frozen pressure operators of the residual)
    wfp = wflux_pres_x3
    w_presa_base = apply_op(wfp, ops.derivative_z)
    apply_op(wflux_pres_itf_x3, ops.correction_DU, out=w_presa_base, beta=1.0)
    w_presb_base = apply_op(log_p, ops.derivative_z)
    apply_op(xp.log(pressure_itf_x3), ops.correction_DU, out=w_presb_base, beta=1.0)
    w_presb_base = w_presb_base * wfp
    dpdrt = gam * pc / rtc  # d(pressure)/d(rho_theta) at the solution points
    A[:, :, rw, :, rt, :] += xp.einsum("os,ces->ceos", eye_s, g2c(w_presa_base + w_presb_base) * dpdrt)

    # p * dG^R (eq:wbGjac): pressure-weighted central metric term, rho_theta column only.
    # The north and south faces of a slot use DIFFERENT expressions, so element e's down face (south
    # form, alpha) and up face (north form, beta) are kept separate.
    Gd, Gu = sg_n * h33fc[dn], sg_s * h33fc[up]
    cp_d, cp_u = gam * p_d / rt_d, gam * p_u / rt_u  # dp_d / dq_d and dp_u / dq_u at the traces
    alpha_d = 0.5 * Gd / p_u * cp_d
    alpha_u = -0.5 * Gd / p_u * (p_d / p_u) * cp_u
    beta_u = 0.5 * Gu / p_d * cp_u
    beta_d = -0.5 * Gu / p_d * (p_u / p_d) * cp_d
    pdL, pdR = pc * dLv, pc * dRv
    L[:, ED, rw, :, rt, :] += xp.einsum("ceo,ce,ces->ceos", pdL[:, ED], alpha_d[:, IT], ER[:, EU, rt])
    A[:, ED, rw, :, rt, :] += xp.einsum("ceo,ce,ces->ceos", pdL[:, ED], alpha_u[:, IT], EL[:, ED, rt])
    A[:, EU, rw, :, rt, :] += xp.einsum("ceo,ce,ces->ceos", pdR[:, EU], beta_d[:, IT], ER[:, EU, rt])
    U[:, EU, rw, :, rt, :] += xp.einsum("ceo,ce,ces->ceos", pdR[:, EU], beta_u[:, IT], EL[:, ED, rt])
    A[:, 0, rw, :, rt, :] += xp.einsum("co,c,cs->cos", pdL[:, 0], alpha_d[:, 0] + alpha_u[:, 0], EL[:, 0, rt])
    A[:, nz - 1, rw, :, rt, :] += xp.einsum(
        "co,c,cs->cos", pdR[:, nz - 1], beta_d[:, nz] + beta_u[:, nz], ER[:, nz - 1, rt]
    )

    # p * sqrtG h33 * D^c[d log p] -- entirely local to the element (its own traces, not the padded
    # ones), so it only touches the diagonal block
    pw = pc * g2c(wfp)
    A[:, :, rw, :, rt, :] += xp.einsum("ceo,os,ces->ceos", pw, Dz, gam / rtc)
    A[:, :, rw, :, rt, :] += xp.einsum(
        "ceo,o,ces->ceos", pw, dLv, gam * EL[:, :, rt, :] / qitfc[:, :, 0, rt][..., None]
    )
    A[:, :, rw, :, rt, :] += xp.einsum(
        "ceo,o,ces->ceos", pw, dRv, gam * ER[:, :, rt, :] / qitfc[:, :, 1, rt][..., None]
    )

    # ================= common -1/sqrtG row scaling, then the filtered gravity term =================
    scale = (-isgc)[:, :, None, :, None, None]
    L *= scale
    A *= scale
    U *= scale
    A[:, :, rw, :, rr, :] -= xp.einsum("ceo,os,ces->ceos", idzc * gravity * isgc, HF, sgc)

    return L.reshape(ncol, nz, m, m), A.reshape(ncol, nz, m, m), U.reshape(ncol, nz, m, m)


def blocks_matvec(rhsobj, L, A, U, xc):
    """Block-tridiagonal matvec: out_e = L_e xc_{e-1} + A_e xc_e + U_e xc_{e+1}, per column."""
    xp = rhsobj.device.xp
    out = xp.einsum("ceij,cej->cei", A, xc)
    out[:, 1:] += xp.einsum("ceij,cej->cei", L[:, 1:], xc[:, :-1])
    out[:, :-1] += xp.einsum("ceij,cej->cei", U[:, :-1], xc[:, 1:])
    return out


def block_thomas_solve(rhsobj, L, A, U, b, dt):
    """Solve (I - (dt/2) J1) x = b per column by block-Thomas, given the tridiagonal blocks (L,A,U)
    of J1 and the right-hand side b of shape (ncol, nz, m). Returns x of the same shape.

    M = I - (dt/2) J1, so the sub/main/super blocks of M are Lm=-(dt/2)L, Am=I-(dt/2)A, Um=-(dt/2)U.
    Forward:  C_0 = Am_0;  C_e = Am_e - Lm_e C_{e-1}^{-1} Um_{e-1}
              d_0 = b_0;   d_e = b_e - Lm_e C_{e-1}^{-1} d_{e-1}
    Back:     x_{E-1} = C_{E-1}^{-1} d_{E-1};  x_e = C_e^{-1}(d_e - Um_e x_{e+1}).
    Each C_{e-1}^{-1} is applied to [Um_{e-1} | d_{e-1}] with a single batched solve."""
    xp = rhsobj.device.xp
    a = 0.5 * dt
    ncol, nz, m, _ = A.shape
    # Factor and solve in double even in single-precision runs: the blocks are tiny, so the cost is
    # negligible and the stiff column solve stays accurate. To keep the memory footprint small (6
    # ranks share one GPU), the double-precision blocks are formed one vertical element at a time
    # rather than as full (ncol, nz, m, m) copies. The result is returned in the input dtype.
    acc = xp.float64
    out_dtype = b.dtype
    eye = xp.eye(m, dtype=acc).reshape(1, m, m)

    def Am(e):
        return eye - a * A[:, e].astype(acc)

    def Lm(e):
        return (-a) * L[:, e].astype(acc)

    def Um(e):
        return (-a) * U[:, e].astype(acc)

    # Each diagonal block C[e] is used in two solves -- once in the forward sweep (as C[e-1], applied
    # to [Um | d]) and once in the back substitution -- so its LU is worth keeping instead of
    # refactoring it the second time (measured: the back-sub refactorization is ~0.06 s/step, ~20% of
    # this solve). torch exposes a batched lu_factor / lu_solve; on backends that do not, fall back to
    # solve() (which refactors each time) so the CPU/cupy paths keep working.
    if hasattr(xp.linalg, "lu_factor") and hasattr(xp.linalg, "lu_solve"):
        def factor(mat):
            return xp.linalg.lu_factor(mat)

        def fsolve(fac, rhs):
            return xp.linalg.lu_solve(fac[0], fac[1], rhs)

    else:
        def factor(mat):
            return mat

        def fsolve(fac, rhs):
            return xp.linalg.solve(fac, rhs)

    b = b.astype(acc)
    fac = [None] * nz
    d = [None] * nz
    fac[0] = factor(Am(0))
    d[0] = b[:, 0]
    for e in range(1, nz):
        # one batched solve for both the block update and the RHS update, reusing C[e-1]'s factors
        rhs_join = xp.concatenate([Um(e - 1), d[e - 1][..., None]], axis=-1)  # (ncol, m, m+1)
        sol = fsolve(fac[e - 1], rhs_join)
        T = sol[..., :m]  # C_{e-1}^{-1} Um_{e-1}
        y = sol[..., m]  # C_{e-1}^{-1} d_{e-1}
        fac[e] = factor(Am(e) - Lm(e) @ T)
        d[e] = b[:, e] - (Lm(e) @ y[..., None])[..., 0]

    x = [None] * nz
    x[nz - 1] = fsolve(fac[nz - 1], d[nz - 1][..., None])[..., 0]
    for e in range(nz - 2, -1, -1):
        rhs_e = d[e] - (Um(e) @ x[e + 1][..., None])[..., 0]
        x[e] = fsolve(fac[e], rhs_e[..., None])[..., 0]
    return xp.stack(x, axis=1).astype(out_dtype)


def m_matvec(rhsobj, L, A, U, xc, dt):
    """Apply M = I - (dt/2) J1 (block-tridiagonal) to a column vector xc (ncol, nz, m)."""
    return xc - 0.5 * dt * blocks_matvec(rhsobj, L, A, U, xc)


# ---------------------------------------------------------------------------------------------------
# Analytic horizontal flux-divergence Jacobian J2_flux v (PartRosExp2 exponential part).
#
# Mirrors ``RHSDirecFluxReconstruction_mpi_v2.horizontal_flux_div`` linearized: the interior fluxes
# by sqrtG*A^d*dq (d = 0, 1 for x1, x2) and the horizontal Rusanov interface fluxes by their
# frozen-lambda differential, with the perturbation traces halo-exchanged (the cube-sphere panels
# couple horizontally across ranks; the exchange also rotates the momentum vectors, which is linear
# and so applies to dq). No wall -- horizontal boundaries are panel-coupled through the exchange.
# ---------------------------------------------------------------------------------------------------

_mid_i = numpy.s_[..., 1:-1, :]
_mid_j = numpy.s_[..., 1:-1, :, :]


def _p_itf(xp, rt):
    return p0 * xp.exp((cpd / cvd) * xp.log(rt * (Rd / p0)))


def _hori_interface(xp, direction, qf, dqf, sg, hci, left, right):
    """Frozen-lambda differential of the horizontal Rusanov flux for one direction.

    ``left``/``right`` are the (east, west) [x1] or (north, south) [x2] element slices; ``sg`` and
    ``hci`` are the interface sqrtG and h_contra for the direction. Returns the full interface-flux
    perturbation array (same shape as qf)."""
    p_itf = _p_itf(xp, qf[idx_rho_theta])
    mom = _MOM[direction]
    u_r = qf[mom][right] / qf[idx_rho][right]
    u_l = qf[mom][left] / qf[idx_rho][left]
    h00 = hci[direction, direction]
    eig_l = xp.abs(u_l) + xp.sqrt(h00[left] * heat_capacity_ratio * p_itf[left] / qf[idx_rho][left])
    eig_r = xp.abs(u_r) + xp.sqrt(h00[right] * heat_capacity_ratio * p_itf[right] / qf[idx_rho][right])
    eig = xp.maximum(eig_l, eig_r)

    dflux_l = sg[left] * ad_matvec(
        dqf[left], qf[left], p_itf[left], direction, hci[direction, 0][left], hci[direction, 1][left], hci[direction, 2][left]
    )
    dflux_r = sg[right] * ad_matvec(
        dqf[right], qf[right], p_itf[right], direction, hci[direction, 0][right], hci[direction, 1][right], hci[direction, 2][right]
    )
    out = xp.zeros_like(qf)
    out[left] = 0.5 * (dflux_l + dflux_r - eig * sg[left] * (dqf[right] - dqf[left]))
    out[right] = out[left]
    return out


def j2_prepare(rhsobj, q):
    """Compute the frozen base of ``j2_flux_matvec`` once. Its ``horizontal_flux_div`` base (traces,
    exchanged neighbours, pressure) is identical for every PMEX matvec of a step, so it is wasteful to
    recompute it per matvec. Returns COPIES, so the FD forcing evaluations cannot clobber them."""
    rhsobj.horizontal_flux_div(q)
    return (
        rhsobj.ops,
        rhsobj.pressure.copy(),
        rhsobj.q_itf_x1.copy(),
        rhsobj.q_itf_x2.copy(),
        rhsobj.q_itf_full_x1.copy(),
        rhsobj.q_itf_full_x2.copy(),
    )


def j2_flux_matvec(rhsobj, q, dq, base=None):
    """Analytic action of the Jacobian of ``horizontal_flux_div`` on ``dq`` (FD-free). ``base`` is the
    cached tuple from :func:`j2_prepare`; if omitted it is recomputed (convenient but slow)."""
    xp = rhsobj.device.xp
    m = rhsobj.metric
    if base is None:
        base = j2_prepare(rhsobj, q)
    ops, pressure, qitf1, qitf2, qf1, qf2 = base
    n = rhsobj.geom.num_solpts**2
    isz = rhsobj.geom.itf_size
    hc = m.h_contra_new

    # --- interior: sqrtG A^d dq, d = 0 (x1), 1 (x2) ---
    df_x1 = m.sqrtG_new * ad_matvec(dq, q, pressure, 0, hc[0, 0], hc[0, 1], hc[0, 2])
    df_x2 = m.sqrtG_new * ad_matvec(dq, q, pressure, 1, hc[1, 0], hc[1, 1], hc[1, 2])
    out = apply_op(df_x1, ops.derivative_x)
    apply_op(df_x2, ops.derivative_y, out=out, beta=1.0)

    # --- perturbation traces (linearize the log-exp extrapolation of rho, rho_theta) ---
    dqi1 = apply_op(dq, ops.extrap_x)
    dqi2 = apply_op(dq, ops.extrap_y)
    for dqi, op, qitf in ((dqi1, ops.extrap_x, qitf1), (dqi2, ops.extrap_y, qitf2)):
        dqi[idx_rho] = qitf[idx_rho] * apply_op(dq[idx_rho] / q[idx_rho], op)
        dqi[idx_rho_theta] = qitf[idx_rho_theta] * apply_op(dq[idx_rho_theta] / q[idx_rho_theta], op)

    # --- halo-exchange the perturbation traces ---
    req = rhsobj.ptopo.start_exchange_euler_3d(
        dqi2[..., 0, :, :isz],
        dqi2[..., -1, :, isz:],
        dqi1[..., 0, :isz],
        dqi1[..., -1, isz:],
        rhsobj.geom.boundary_sn_new,
        rhsobj.geom.boundary_we_new,
        flip_dim=(-3, -1),
    )
    dq_s, dq_n, dq_w, dq_e = req.wait()

    # --- ghost-padded perturbation trace arrays (mid = local, faces = exchanged neighbours) ---
    dqf1 = xp.zeros_like(qf1)
    dqf2 = xp.zeros_like(qf2)
    dqf1[_mid_i] = dqi1
    dqf2[_mid_j] = dqi2
    dqf1[..., 0, isz:] = dq_w
    dqf1[..., -1, :isz] = dq_e
    dqf2[..., 0, :, isz:] = dq_s
    dqf2[..., -1, :, :isz] = dq_n

    # --- interface Rusanov differential, x1 (west/east) and x2 (south/north) ---
    west = numpy.s_[..., 1:, :n]
    east = numpy.s_[..., :-1, n:]
    south = numpy.s_[..., 1:, :, :n]
    north = numpy.s_[..., :-1, :, n:]
    dfitf1 = _hori_interface(xp, 0, qf1, dqf1, m.sqrtG_itf_i_new, m.h_contra_itf_i_new, east, west)
    dfitf2 = _hori_interface(xp, 1, qf2, dqf2, m.sqrtG_itf_j_new, m.h_contra_itf_j_new, north, south)

    apply_op(dfitf1[_mid_i], ops.correction_WE, out=out, beta=1.0)
    apply_op(dfitf2[_mid_j], ops.correction_SN, out=out, beta=1.0)
    out *= -m.inv_sqrtG_new
    return out
