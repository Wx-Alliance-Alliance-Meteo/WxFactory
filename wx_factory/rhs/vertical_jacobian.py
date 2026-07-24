"""Analytic Jacobian of the vertically-stiff partition f1 (PartRosExp2).

Computes the derivative of f1 from RHSDirecFluxReconstruction_mpi_v2.implicit.
Variable order: (rho, rho*u1, rho*u2, rho*w, rho*theta).
"""

import numpy
from numpy.typing import NDArray

from ..common.definitions import (
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
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


_MOM = (idx_rho_u1, idx_rho_u2, idx_rho_u3)


def ad_matvec(dq, q, pressure, direction, h1d, h2d, h3d):
    """Apply direction-specific pointwise flux Jacobian A^d(q) to dq."""
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_u3] / rho)
    theta = q[idx_rho_theta] / rho
    md = _MOM[direction]
    ud = u[direction]
    cs2_over_theta = heat_capacity_ratio * pressure / q[idx_rho_theta]
    hd = (h1d, h2d, h3d)

    d_rho = dq[idx_rho]
    d_md = dq[md]
    d_rt = dq[idx_rho_theta]

    out = dq * 0.0
    out[idx_rho] = d_md
    for i in range(3):
        out[_MOM[i]] = ud * dq[_MOM[i]] + u[i] * d_md - u[i] * ud * d_rho + hd[i] * cs2_over_theta * d_rt
    out[idx_rho_theta] = ud * d_rt + theta * d_md - ud * theta * d_rho
    return out


def ad_matrix(q, pressure, direction, h1d, h2d, h3d, xp):
    """Build explicit 5x5 flux Jacobian A^d(q) at each point."""
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_u3] / rho)
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
    """Vertical-flux Jacobian A3(q) dq (direction-2 case)."""
    return ad_matvec(dq, q, pressure, 2, h13, h23, h33)


def _trace_metric(m, comp):
    """Metric component h^{comp,3} at vertical interfaces."""
    return m.h_contra_itf_k_new[comp, 2]


def j1_prepare(rhsobj, q):
    """Precompute frozen base variables for implicit_jvp."""
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
    """Compute analytic Jacobian-vector product J1 * dq.

    Supports batched dq on axis 1.
    """
    xp = rhsobj.device.xp
    m = rhsobj.metric

    if base is None:
        base = j1_prepare(rhsobj, q)
    ops, pressure, q_itf_x3, q_itf_full_x3, wflux_pres_x3, wflux_pres_itf_x3, log_p, pressure_itf_x3 = base
    op_extrap = ops.extrap_z
    op_dz = ops.derivative_z
    op_corr = ops.correction_DU
    n = rhsobj.geom.num_solpts**2

    # Interior flux perturbation
    h13, h23, h33 = m.h_contra_new[0, 2], m.h_contra_new[1, 2], m.h_contra_new[2, 2]
    dfx3 = m.sqrtG_new * a3_matvec(dq, q, pressure, h13, h23, h33)

    # Trace perturbations (linearized log-exp for rho, rho_theta)
    dq_itf = apply_op(dq, op_extrap)
    q_itf = q_itf_x3
    dq_itf[idx_rho] = q_itf[idx_rho] * apply_op(dq[idx_rho] / q[idx_rho], op_extrap)
    dq_itf[idx_rho_theta] = q_itf[idx_rho_theta] * apply_op(dq[idx_rho_theta] / q[idx_rho_theta], op_extrap)

    # Pad traces with ghosts
    def to_full(itf, base_full):
        full = xp.zeros(itf.shape[:-4] + base_full.shape[-4:], dtype=itf.dtype)
        full[_mid_k] = itf
        full[..., 0, :, :, n:] = full[..., 1, :, :, :n]
        full[..., -1, :, :, :n] = full[..., -2, :, :, n:]
        return full

    qf = q_itf_full_x3
    dqf = to_full(dq_itf, qf)

    # Wall reflection of vertical momentum for plain rows
    qf_r = qf.copy()
    dqf_r = dqf.copy()
    qf_r[idx_rho_u3][_bot_k] *= -1.0
    qf_r[idx_rho_u3][_top_k] *= -1.0
    dqf_r[idx_rho_u3][_bot_k] *= -1.0
    dqf_r[idx_rho_u3][_top_k] *= -1.0

    # Interface pressure
    p_itf = p0 * xp.exp((cpd / cvd) * xp.log(qf[idx_rho_theta] * (Rd / p0)))

    south = xp.s_[..., 1:, :, :, :n]
    north = xp.s_[..., :-1, :, :, n:]
    h33_itf = m.h_contra_itf_k_new[2, 2]
    sg_itf = m.sqrtG_itf_k_new

    w_d = qf[idx_rho_u3][north] / qf[idx_rho][north]
    w_u = qf[idx_rho_u3][south] / qf[idx_rho][south]
    eig_d = xp.abs(w_d) + xp.sqrt(h33_itf[north] * heat_capacity_ratio * p_itf[north] / qf[idx_rho][north])
    eig_u = xp.abs(w_u) + xp.sqrt(h33_itf[south] * heat_capacity_ratio * p_itf[south] / qf[idx_rho][south])
    eig = xp.maximum(eig_d, eig_u)

    # Linearized advective and pressure fluxes at traces
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

    # Well-balanced rho_w residual terms
    wfp = wflux_pres_x3
    w_presa_base = apply_op(wfp, op_dz)
    apply_op(wflux_pres_itf_x3, op_corr, out=w_presa_base, beta=1.0)
    logp_bdy = xp.log(pressure_itf_x3)
    w_presb_base = apply_op(log_p, op_dz)
    apply_op(logp_bdy, op_corr, out=w_presb_base, beta=1.0)
    w_presb_base = w_presb_base * wfp

    # Interior perturbations
    w = q[idx_rho_u3] / q[idx_rho]
    dp = heat_capacity_ratio * pressure / q[idx_rho_theta] * dq[idx_rho_theta]
    dlogp = heat_capacity_ratio / q[idx_rho_theta] * dq[idx_rho_theta]
    dwadv_x3 = m.sqrtG_new * (2.0 * w * dq[idx_rho_u3] - w**2 * dq[idx_rho])

    # Interface perturbations
    p_d, p_u = p_itf[north], p_itf[south]
    rt_d, rt_u = qf[idx_rho_theta][north], qf[idx_rho_theta][south]
    dp_d = heat_capacity_ratio * p_d / rt_d * dqf[idx_rho_theta][north]
    dp_u = heat_capacity_ratio * p_u / rt_u * dqf[idx_rho_theta][south]
    drw_d, drw_u = dqf[idx_rho_u3][north], dqf[idx_rho_u3][south]

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

    # Assemble well-balanced rho_w residual perturbation
    dw_df3 = apply_op(dwadv_x3, op_dz)
    apply_op(dwadv_full[_mid_k], op_corr, out=dw_df3, beta=1.0)
    dw_presa = apply_op(dwpres_full[_mid_k], op_corr)
    dw_presb = apply_op(dlogp, op_dz)
    apply_op(dlogp_itf, op_corr, out=dw_presb, beta=1.0)
    dw_presb = dw_presb * wfp
    drhs_w = dw_df3 + dp * (w_presa_base + w_presb_base) + pressure * (dw_presa + dw_presb)

    # Combine rows, apply scaling and gravity Jacobian
    out = apply_op(dfx3, op_dz)
    apply_op(dfitf, op_corr, out=out, beta=1.0)
    out[idx_rho_u3] = drhs_w
    out *= -m.inv_sqrtG_new
    out[idx_rho_u3] -= (
        m.inv_dzdeta_new * gravity * m.inv_sqrtG_new * ((m.sqrtG_new * dq[idx_rho]) @ ops.highfilter_k)
    )
    return out


# -----------------------------------------------------------------------------
# Column-wise block-tridiagonal assembly of J1.
# Vertical blocks are dense (5*ns x 5*ns), ordered as (variable, solpt).
# -----------------------------------------------------------------------------


def _col_dims(rhsobj, q):
    ns = rhsobj.geom.num_solpts
    nv, nz, ny, nx, _ = q.shape
    return ns, ns * ns, nv, nz, ny, nx, nv * ns, ny * nx * ns * ns  # ns, nh, nv, nz, ny, nx, m, ncol


def state_to_col(rhsobj, x):
    """Convert grid layout to column layout (nv, nz, ny, nx, ns**3) -> (ncol, nz, m)."""
    xp = rhsobj.device.xp
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, x)
    x6 = x.reshape(nv, nz, ny, nx, ns, nh)
    p = xp.transpose(x6, (2, 3, 5, 1, 0, 4))
    return p.reshape(ncol, nz, m)


def col_to_state(rhsobj, xc, ref):
    """Inverse of state_to_col."""
    xp = rhsobj.device.xp
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, ref)
    p = xc.reshape(ny, nx, nh, nz, nv, ns)
    x6 = xp.transpose(p, (4, 3, 0, 1, 5, 2))
    return x6.reshape(ref.shape)


def batched_state_to_col(rhsobj, x, ref):
    """Convert batched state to column layout (nv, nb, nz, ny, nx, ns**3) -> (nb, ncol, nz, m)."""
    xp = rhsobj.device.xp
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, ref)
    nb = x.shape[1]
    x7 = x.reshape(nv, nb, nz, ny, nx, ns, nh)
    p = xp.transpose(x7, (1, 3, 4, 6, 2, 0, 5))  # (nb, ny, nx, nh, nz, nv, ns)
    return p.reshape(nb, ncol, nz, m)


def assemble_j1_blocks(rhsobj, q, batch=3):
    """Assemble block-tridiagonal J1 blocks (L, A, U) per column via 3-color probing."""
    xp = rhsobj.device.xp
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, q)
    L = xp.zeros((ncol, nz, m, m), dtype=q.dtype)
    A = xp.zeros((ncol, nz, m, m), dtype=q.dtype)
    U = xp.zeros((ncol, nz, m, m), dtype=q.dtype)

    base = j1_prepare(rhsobj, q)
    probes = [(v, s) for v in range(nv) for s in range(ns)]
    nj = len(probes)
    batch = max(1, min(batch, nj))

    for c in range(3):
        # Elements partition by color to avoid overlapping diagonal writes
        eA = xp.asarray([e for e in range(nz) if e % 3 == c])
        eL = xp.asarray([e for e in range(1, nz) if (e - 1) % 3 == c])
        eU = xp.asarray([e for e in range(nz - 1) if (e + 1) % 3 == c])
        for start in range(0, nj, batch):
            chunk = probes[start : start + batch]
            nb = len(chunk)
            dq = xp.zeros((nv, nb, nz, ny, nx, ns * nh), dtype=q.dtype)
            for p, (v, s) in enumerate(chunk):
                dq[v, p, c::3, :, :, s * nh : (s + 1) * nh] = 1.0
            Pc = batched_state_to_col(rhsobj, implicit_jvp(rhsobj, q, dq, base), q)
            js = slice(start, start + nb)
            for blk, es in ((A, eA), (L, eL), (U, eU)):
                if es.size:
                    blk[:, es, :, js] = xp.transpose(Pc[:, :, es, :], (1, 2, 3, 0))
    return L, A, U


# -----------------------------------------------------------------------------
# Direct (analytic) assembly of blocks without probing.
# Uses 1D reference operators and pointwise 5x5 blocks.
# Interface slot t (0..nz) pairs element t-1 up face with element t down face.
# -----------------------------------------------------------------------------


def _g2c(xp, arr, ns, nh, ny, nx):
    """Convert grid array to column layout (ncol, nz, ns)."""
    nz = arr.shape[0]
    a = arr.reshape(nz, ny, nx, ns, nh)
    return xp.transpose(a, (1, 2, 4, 0, 3)).reshape(ny * nx * nh, nz, ns)


def _i2c(xp, arr, nh, ny, nx):
    """Convert scalar interface array to column layout (ncol, np, 2)."""
    npd = arr.shape[0]
    a = arr.reshape(npd, ny, nx, 2, nh)
    return xp.transpose(a, (1, 2, 4, 0, 3)).reshape(ny * nx * nh, npd, 2)


def _i5c(xp, arr, nh, ny, nx):
    """Convert state interface array to column layout (ncol, np, 2, nv)."""
    nv, npd = arr.shape[0], arr.shape[1]
    a = arr.reshape(nv, npd, ny, nx, 2, nh)
    return xp.transpose(a, (2, 3, 5, 1, 4, 0)).reshape(ny * nx * nh, npd, 2, nv)


def assemble_j1_blocks_analytic(rhsobj, q):
    """Assemble block-tridiagonal J1 directly from closed form (L, A, U)."""
    xp = rhsobj.device.xp
    met = rhsobj.metric
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, q)
    gam = heat_capacity_ratio
    rw, rt, rr = idx_rho_u3, idx_rho_theta, idx_rho
    dt_ = q.dtype

    base = j1_prepare(rhsobj, q)
    ops, pressure, q_itf_x3, qf, wflux_pres_x3, wflux_pres_itf_x3, log_p, pressure_itf_x3 = base

    # 1D reference operators
    Dz = ops.diff_solpt  # D_int
    eLv, eRv = ops.extrap_down, ops.extrap_up  # e_L, e_R
    dLv, dRv = ops.correction[:, 0], ops.correction[:, 1]  # d_L tilde, d_R tilde
    HF = ops.highfilter
    eye_s = xp.eye(ns, dtype=dt_)

    def g2c(a):
        return _g2c(xp, a, ns, nh, ny, nx)

    # Grid variables in column layout
    qc = state_to_col(rhsobj, q).reshape(ncol, nz, nv, ns)
    pc = g2c(pressure)
    sgc, isgc, idzc = g2c(met.sqrtG_new), g2c(met.inv_sqrtG_new), g2c(met.inv_dzdeta_new)
    A3s = ad_matrix(
        xp.transpose(qc, (2, 0, 1, 3)),
        pc,
        2,
        g2c(met.h_contra_new[0, 2]),
        g2c(met.h_contra_new[1, 2]),
        g2c(met.h_contra_new[2, 2]),
        xp,
    )

    # Trace variables in column layout (padded: nz + 2 elements, 2 faces)
    qfc = _i5c(xp, qf, nh, ny, nx)
    sgfc = _i2c(xp, met.sqrtG_itf_k_new, nh, ny, nx)
    h33fc = _i2c(xp, _trace_metric(met, 2), nh, ny, nx)
    p_itf_c = p0 * xp.exp((cpd / cvd) * xp.log(qfc[..., rt] * (Rd / p0)))
    qfc_r = qfc.copy()  # M-reflected ghosts for plain rows
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

    # Interface slot data: down side (p=t, up face), up side (p=t+1, down face)
    dn, up = numpy.s_[:, : nz + 1, 1], numpy.s_[:, 1:, 0]
    sg_n, sg_s = sgfc[dn], sgfc[up]
    rho_d, rho_u = qfc[dn][..., rr], qfc[up][..., rr]
    w_d, w_u = qfc[dn][..., rw] / rho_d, qfc[up][..., rw] / rho_u  # eig/wb use unreflected traces
    p_d, p_u = p_itf_c[dn], p_itf_c[up]
    rt_d, rt_u = qfc[dn][..., rt], qfc[up][..., rt]
    eig = xp.maximum(
        xp.abs(w_d) + xp.sqrt(h33fc[dn] * gam * p_d / rho_d),
        xp.abs(w_u) + xp.sqrt(h33fc[up] * gam * p_u / rho_u),
    )
    I5 = xp.eye(nv, dtype=dt_).reshape(1, 1, nv, nv)
    lam = (eig * sg_n)[..., None, None]
    Bp = 0.5 * (sg_n[..., None, None] * A3f[dn] + lam * I5)
    Bm = 0.5 * (sg_s[..., None, None] * A3f[up] - lam * I5)

    # Linearized trace extrapolation (rho and rho_theta use log-exp map scaling)
    qitfc = _i5c(xp, q_itf_x3, nh, ny, nx)
    EL = xp.zeros((ncol, nz, nv, ns), dtype=dt_) + eLv
    ER = xp.zeros((ncol, nz, nv, ns), dtype=dt_) + eRv
    for var in (rr, rt):
        EL[:, :, var, :] = eLv * (qitfc[:, :, 0, var][..., None] / qc[:, :, var, :])
        ER[:, :, var, :] = eRv * (qitfc[:, :, 1, var][..., None] / qc[:, :, var, :])

    # ================= plain rows =================
    shp = (ncol, nz, nv, ns, nv, ns)
    L = xp.zeros(shp, dtype=dt_)
    A = xp.zeros(shp, dtype=dt_)
    U = xp.zeros(shp, dtype=dt_)

    # Interior volume: Dz x (sqrtG A3)
    A += xp.einsum("os,ces,cesij->ceiojs", Dz, sgc, A3s)

    # Interior interfaces: slot t = 1..nz-1
    IT, ED, EU = slice(1, nz), slice(1, nz), slice(0, nz - 1)
    L[:, ED] += xp.einsum("o,ceij,cejs->ceiojs", dLv, Bp[:, IT], ER[:, EU])
    A[:, ED] += xp.einsum("o,ceij,cejs->ceiojs", dLv, Bm[:, IT], EL[:, ED])
    A[:, EU] += xp.einsum("o,ceij,cejs->ceiojs", dRv, Bp[:, IT], ER[:, EU])
    U[:, EU] += xp.einsum("o,ceij,cejs->ceiojs", dRv, Bm[:, IT], EL[:, ED])

    # Walls: ghost side reflected by M
    Mv = xp.ones(nv, dtype=dt_)
    Mv[rw] = -1.0
    A[:, 0] += xp.einsum("o,cij,cjs->ciojs", dLv, Bp[:, 0] * Mv + Bm[:, 0], EL[:, 0])
    A[:, nz - 1] += xp.einsum("o,cij,cjs->ciojs", dRv, Bp[:, nz] + Bm[:, nz] * Mv, ER[:, nz - 1])

    # ================= well-balanced rho_w row =================
    L[:, :, rw] = 0.0
    A[:, :, rw] = 0.0
    U[:, :, rw] = 0.0

    rhoc, rtc = qc[:, :, rr, :], qc[:, :, rt, :]
    wc = qc[:, :, rw, :] / rhoc

    # Advective interior
    A[:, :, rw, :, rw, :] += xp.einsum("os,ces->ceos", Dz, sgc * 2.0 * wc)
    A[:, :, rw, :, rr, :] += xp.einsum("os,ces->ceos", Dz, -sgc * wc**2)

    # Advective interfaces
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

    # Pressure operator perturbation
    wfp = wflux_pres_x3
    w_presa_base = apply_op(wfp, ops.derivative_z)
    apply_op(wflux_pres_itf_x3, ops.correction_DU, out=w_presa_base, beta=1.0)
    w_presb_base = apply_op(log_p, ops.derivative_z)
    apply_op(xp.log(pressure_itf_x3), ops.correction_DU, out=w_presb_base, beta=1.0)
    w_presb_base = w_presb_base * wfp
    dpdrt = gam * pc / rtc
    A[:, :, rw, :, rt, :] += xp.einsum("os,ces->ceos", eye_s, g2c(w_presa_base + w_presb_base) * dpdrt)

    # Metric perturbation
    Gd, Gu = sg_n * h33fc[dn], sg_s * h33fc[up]
    cp_d, cp_u = gam * p_d / rt_d, gam * p_u / rt_u
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

    # Local element term
    pw = pc * g2c(wfp)
    A[:, :, rw, :, rt, :] += xp.einsum("ceo,os,ces->ceos", pw, Dz, gam / rtc)
    A[:, :, rw, :, rt, :] += xp.einsum(
        "ceo,o,ces->ceos", pw, dLv, gam * EL[:, :, rt, :] / qitfc[:, :, 0, rt][..., None]
    )
    A[:, :, rw, :, rt, :] += xp.einsum(
        "ceo,o,ces->ceos", pw, dRv, gam * ER[:, :, rt, :] / qitfc[:, :, 1, rt][..., None]
    )

    # Scale by -1/sqrtG and add filtered gravity
    scale = (-isgc)[:, :, None, :, None, None]
    L *= scale
    A *= scale
    U *= scale
    A[:, :, rw, :, rr, :] -= xp.einsum("ceo,os,ces->ceos", idzc * gravity * isgc, HF, sgc)

    return L.reshape(ncol, nz, m, m), A.reshape(ncol, nz, m, m), U.reshape(ncol, nz, m, m)


def blocks_matvec(rhsobj, L, A, U, xc):
    """Apply block-tridiagonal operator per column."""
    xp = rhsobj.device.xp
    out = xp.einsum("ceij,cej->cei", A, xc)
    out[:, 1:] += xp.einsum("ceij,cej->cei", L[:, 1:], xc[:, :-1])
    out[:, :-1] += xp.einsum("ceij,cej->cei", U[:, :-1], xc[:, 1:])
    return out


def block_thomas_solve(rhsobj, L, A, U, b, dt):
    """Solve (I - (dt/2) J1) x = b per column via block-Thomas algorithm."""
    xp = rhsobj.device.xp
    a = 0.5 * dt
    ncol, nz, m, _ = A.shape
    out_dtype = b.dtype
    # Form blocks one element at a time to save memory. The elimination needs more precision than
    # the float32 state carries -- without it the density row loses ~400 ulps to cancellation -- but
    # factoring in float64 is expensive on a consumer GPU (1/64 the float32 rate) and this sweep is
    # entirely FLOP-bound. So for a single-precision state, factor in float32 and buy the accuracy
    # back with one step of iterative refinement whose residual is formed in float64.
    refine = out_dtype == xp.float32
    acc = xp.float32 if refine else xp.float64
    eye = xp.eye(m, dtype=acc).reshape(1, m, m)

    def Am(e):
        return eye - a * A[:, e].astype(acc)

    def Lm(e):
        return (-a) * L[:, e].astype(acc)

    def Um(e):
        return (-a) * U[:, e].astype(acc)

    # Cache LU factorization of diagonal blocks to avoid refactoring during back-substitution
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
        # Batched solve for block update and RHS update
        rhs_join = xp.concatenate([Um(e - 1), d[e - 1][..., None]], axis=-1)  # (ncol, m, m+1)
        sol = fsolve(fac[e - 1], rhs_join)
        T = sol[..., :m]  # C_{e-1}^{-1} Um_{e-1}
        y = sol[..., m]  # C_{e-1}^{-1} d_{e-1}
        fac[e] = factor(Am(e) - Lm(e) @ T)
        d[e] = b[:, e] - (Lm(e) @ y[..., None])[..., 0]

    def substitute(rhs):
        """Solve M z = rhs with the factors already computed, rhs given per vertical element."""
        dd = [None] * nz
        dd[0] = rhs[0]
        for e in range(1, nz):
            y = fsolve(fac[e - 1], dd[e - 1][..., None])[..., 0]
            dd[e] = rhs[e] - (Lm(e) @ y[..., None])[..., 0]
        z = [None] * nz
        z[nz - 1] = fsolve(fac[nz - 1], dd[nz - 1][..., None])[..., 0]
        for e in range(nz - 2, -1, -1):
            rhs_e = dd[e] - (Um(e) @ z[e + 1][..., None])[..., 0]
            z[e] = fsolve(fac[e], rhs_e[..., None])[..., 0]
        return z

    x = [None] * nz
    x[nz - 1] = fsolve(fac[nz - 1], d[nz - 1][..., None])[..., 0]
    for e in range(nz - 2, -1, -1):
        rhs_e = d[e] - (Um(e) @ x[e + 1][..., None])[..., 0]
        x[e] = fsolve(fac[e], rhs_e[..., None])[..., 0]

    if refine:
        # Residual r = b - M x in float64, one vertical element at a time: a float64 copy of all of
        # L, A and U at once would be ~400 MB per rank, which does not fit alongside the other five.
        f64 = xp.float64
        eye64 = xp.eye(m, dtype=f64).reshape(1, m, m)
        r = [None] * nz
        for e in range(nz):
            r[e] = b[:, e].astype(f64) - ((eye64 - a * A[:, e].astype(f64)) @ x[e].astype(f64)[..., None])[..., 0]
            if e > 0:
                r[e] += a * (L[:, e].astype(f64) @ x[e - 1].astype(f64)[..., None])[..., 0]
            if e < nz - 1:
                r[e] += a * (U[:, e].astype(f64) @ x[e + 1].astype(f64)[..., None])[..., 0]
        dx = substitute([r[e].astype(acc) for e in range(nz)])
        x = [x[e] + dx[e] for e in range(nz)]

    return xp.stack(x, axis=1).astype(out_dtype)


def m_matvec(rhsobj, L, A, U, xc, dt):
    """Apply M = I - (dt/2) J1 to column vector xc."""
    return xc - 0.5 * dt * blocks_matvec(rhsobj, L, A, U, xc)


# -----------------------------------------------------------------------------
# Analytic horizontal flux-divergence Jacobian J2_flux * v.
# Linearizes horizontal_flux_div with halo-exchanged perturbation traces.
# -----------------------------------------------------------------------------

_mid_i = numpy.s_[..., 1:-1, :]
_mid_j = numpy.s_[..., 1:-1, :, :]


def _p_itf(xp, rt):
    return p0 * xp.exp((cpd / cvd) * xp.log(rt * (Rd / p0)))


def _hori_interface(xp, direction, qf, dqf, sg, hci, left, right):
    """Compute frozen-lambda differential of horizontal Rusanov flux."""
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
    """Precompute frozen base variables for j2_flux_matvec."""
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
    """Analytic Jacobian-vector product of horizontal_flux_div on dq."""
    xp = rhsobj.device.xp
    m = rhsobj.metric
    if base is None:
        base = j2_prepare(rhsobj, q)
    ops, pressure, qitf1, qitf2, qf1, qf2 = base
    n = rhsobj.geom.num_solpts**2
    isz = rhsobj.geom.itf_size
    hc = m.h_contra_new

    # Interior flux perturbations
    df_x1 = m.sqrtG_new * ad_matvec(dq, q, pressure, 0, hc[0, 0], hc[0, 1], hc[0, 2])
    df_x2 = m.sqrtG_new * ad_matvec(dq, q, pressure, 1, hc[1, 0], hc[1, 1], hc[1, 2])
    out = apply_op(df_x1, ops.derivative_x)
    apply_op(df_x2, ops.derivative_y, out=out, beta=1.0)

    # Trace perturbations
    dqi1 = apply_op(dq, ops.extrap_x)
    dqi2 = apply_op(dq, ops.extrap_y)
    for dqi, op, qitf in ((dqi1, ops.extrap_x, qitf1), (dqi2, ops.extrap_y, qitf2)):
        dqi[idx_rho] = qitf[idx_rho] * apply_op(dq[idx_rho] / q[idx_rho], op)
        dqi[idx_rho_theta] = qitf[idx_rho_theta] * apply_op(dq[idx_rho_theta] / q[idx_rho_theta], op)

    # Exchange perturbation traces
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

    # Pad traces with exchanged ghost data
    dqf1 = xp.zeros_like(qf1)
    dqf2 = xp.zeros_like(qf2)
    dqf1[_mid_i] = dqi1
    dqf2[_mid_j] = dqi2
    dqf1[..., 0, isz:] = dq_w
    dqf1[..., -1, :isz] = dq_e
    dqf2[..., 0, :, isz:] = dq_s
    dqf2[..., -1, :, :isz] = dq_n

    # Horizontal interface Rusanov flux differentials
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


# -----------------------------------------------------------------------------
# Analytic Jacobian of the non-stiff forcing (forcing_only) -- Christoffel/Coriolis/pressure
# source plus the DCMIP Rayleigh sponge. This term is pointwise: gravity is the only non-local
# contribution to forcing_only and it cancels exactly there (added and subtracted with the same
# formula), so d(forcing_only)/dq is a per-point 5x5 map with no grid coupling.
# -----------------------------------------------------------------------------


def forcing_jac_prepare(rhsobj, q):
    """Precompute the frozen state-dependent coefficients for :func:`forcing_jvp`.

    All of these are functions of the fixed state q, which PartRosExp2 holds constant across every
    matvec of one PMEX solve, so they are computed once per step. Returns the velocities, the
    analytic pressure derivative dp/d(rho_theta) = gamma p / rho_theta, and the frozen Rayleigh
    sponge coefficients (or None when the case has no sponge)."""
    xp = rhsobj.device.xp
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_u3] / rho)
    rt = q[idx_rho_theta]
    # Same equation of state as the pointwise kernel: p = p0 (Rd/p0 rho_theta)^gamma.
    p = p0 * xp.exp(heat_capacity_ratio * xp.log((Rd / p0) * rt))
    dpdrt = heat_capacity_ratio * p / rt

    ray = None
    case = getattr(rhsobj.pde, "case_number", None)
    if case in (21, 22):
        from ..init.dcmip import dcmip_schar_damping_coeffs

        rate, u1ref, u2ref, u3ref = dcmip_schar_damping_coeffs(rhsobj.metric, rhsobj.pde.geometry, shear=(case == 22))
        ray = (rate, (u1ref, u2ref, u3ref))

    return u, dpdrt, ray


def forcing_jvp(rhsobj, q, v, base=None):
    """Analytic Jacobian-vector product of ``forcing_only`` on v (pointwise, no grid coupling).

    Mirrors ``compute_forcing_1`` (the momentum source F^d = 2 rho (c0k u^k) + c_ij (rho u^i u^j +
    h^ij p), summed over i<=j) direction by direction, plus the linear Rayleigh sponge. The rho and
    rho_theta rows of forcing_only are identically zero, so only the momentum rows are filled. The
    overall minus sign is ``rhs -= forcing``."""
    xp = rhsobj.device.xp
    m = rhsobj.metric
    if base is None:
        base = forcing_jac_prepare(rhsobj, q)
    u, dpdrt, ray = base
    u1, u2, u3 = u
    ch = m.christoffel  # (3 directions, 9 components, ...spatial)
    hc = m.h_contra_new
    h11, h12, h13 = hc[0, 0], hc[0, 1], hc[0, 2]
    h22, h23, h33 = hc[1, 1], hc[1, 2], hc[2, 2]

    dr = v[idx_rho]
    dm = (v[idx_rho_u1], v[idx_rho_u2], v[idx_rho_u3])
    drt = v[idx_rho_theta]

    out = xp.zeros_like(v)
    for d, row in enumerate(_MOM):
        c01, c02, c03 = ch[d, 0], ch[d, 1], ch[d, 2]
        c11, c12, c13 = ch[d, 3], ch[d, 4], ch[d, 5]
        c22, c23, c33 = ch[d, 6], ch[d, 7], ch[d, 8]

        # Rows of the pointwise forcing Jacobian dF^d/dq, in conserved variables.
        a_r = -(c11 * u1 * u1 + 2.0 * c12 * u1 * u2 + 2.0 * c13 * u1 * u3
                + c22 * u2 * u2 + 2.0 * c23 * u2 * u3 + c33 * u3 * u3)
        a_m1 = 2.0 * c01 + 2.0 * (c11 * u1 + c12 * u2 + c13 * u3)
        a_m2 = 2.0 * c02 + 2.0 * (c12 * u1 + c22 * u2 + c23 * u3)
        a_m3 = 2.0 * c03 + 2.0 * (c13 * u1 + c23 * u2 + c33 * u3)
        a_rt = (c11 * h11 + 2.0 * c12 * h12 + 2.0 * c13 * h13
                + c22 * h22 + 2.0 * c23 * h23 + c33 * h33) * dpdrt

        dF = a_r * dr + a_m1 * dm[0] + a_m2 * dm[1] + a_m3 * dm[2] + a_rt * drt

        if ray is not None:
            rate, uref = ray
            # Rayleigh sponge R^d = rate (rho_u^d - rho uref^d), linear in q: dR^d = rate (dm_d - uref^d dr).
            dF = dF + rate * (dm[d] - uref[d] * dr)

        out[row] = -dF

    return out
