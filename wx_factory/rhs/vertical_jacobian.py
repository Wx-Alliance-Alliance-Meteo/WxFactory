"""Analytic Jacobians for the PartRosExp2 partitions."""

import numpy
import torch
from numpy.typing import NDArray

from ..common.definitions import (
    Rd,
    cpd,
    cvd,
    gravity,
    heat_capacity_ratio,
    idx_rho,
    idx_rho_theta,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
    p0,
)
from ..common.matmul import apply_op, maximum

_mid_k = numpy.s_[..., 1:-1, :, :, :]
_bot_k = numpy.s_[..., 0, :, :, :]
_top_k = numpy.s_[..., -1, :, :, :]


_MOM = (idx_rho_u1, idx_rho_u2, idx_rho_u3)


def ad_matvec(dq, q, pressure, direction, h1d, h2d, h3d):
    """Apply a directional pointwise flux Jacobian."""
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


def ad_matrix(q, pressure, direction, h1d, h2d, h3d):
    """Build a directional pointwise flux Jacobian."""
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_u3] / rho)
    theta = q[idx_rho_theta] / rho
    md = _MOM[direction]
    ud = u[direction]
    cs2_over_theta = heat_capacity_ratio * pressure / q[idx_rho_theta]
    hd = (h1d, h2d, h3d)

    A = torch.zeros(rho.shape + (5, 5), dtype=q.dtype)
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
    """Apply the vertical pointwise flux Jacobian."""
    return ad_matvec(dq, q, pressure, 2, h13, h23, h33)


def _trace_metric(m, comp):
    """Return a vertical-interface metric component."""
    return m.h_contra_itf_k_new[comp, 2]


def _vertical_flux_jvp(rhsobj, q, dq, ops, pressure, q_itf_x3, q_itf_full_x3):
    """Linearize the vertical volume and Rusanov interface fluxes."""
    m = rhsobj.metric
    op_extrap = ops.extrap_z
    n = rhsobj.geom.num_solpts**2

    # Interior flux perturbation
    h13, h23, h33 = m.h_contra_new[0, 2], m.h_contra_new[1, 2], m.h_contra_new[2, 2]
    dfx3 = m.sqrtG_new * a3_matvec(dq, q, pressure, h13, h23, h33)

    # Density traces follow the logarithmic RHS extrapolation.
    dq_itf = apply_op(dq, op_extrap)
    q_itf = q_itf_x3
    dq_itf[idx_rho] = q_itf[idx_rho] * apply_op(dq[idx_rho] / q[idx_rho], op_extrap)
    dq_itf[idx_rho_theta] = q_itf[idx_rho_theta] * apply_op(dq[idx_rho_theta] / q[idx_rho_theta], op_extrap)

    # Pad traces with ghosts
    def to_full(itf, base_full):
        full = torch.zeros(itf.shape[:-4] + base_full.shape[-4:], dtype=itf.dtype)
        full[_mid_k] = itf
        full[..., 0, :, :, n:] = full[..., 1, :, :, :n]
        full[..., -1, :, :, :n] = full[..., -2, :, :, n:]
        return full

    qf = q_itf_full_x3
    dqf = to_full(dq_itf, qf)

    # Reflect vertical momentum at rigid walls.
    qf_r = qf.copy()
    dqf_r = dqf.copy()
    qf_r[idx_rho_u3][_bot_k] *= -1.0
    qf_r[idx_rho_u3][_top_k] *= -1.0
    dqf_r[idx_rho_u3][_bot_k] *= -1.0
    dqf_r[idx_rho_u3][_top_k] *= -1.0

    # Interface pressure
    p_itf = p0 * torch.exp((cpd / cvd) * torch.log(qf[idx_rho_theta] * (Rd / p0)))

    south = numpy.s_[..., 1:, :, :, :n]
    north = numpy.s_[..., :-1, :, :, n:]
    h33_itf = m.h_contra_itf_k_new[2, 2]
    sg_itf = m.sqrtG_itf_k_new

    w_d = qf[idx_rho_u3][north] / qf[idx_rho][north]
    w_u = qf[idx_rho_u3][south] / qf[idx_rho][south]
    eig_d = torch.abs(w_d) + torch.sqrt(h33_itf[north] * heat_capacity_ratio * p_itf[north] / qf[idx_rho][north])
    eig_u = torch.abs(w_u) + torch.sqrt(h33_itf[south] * heat_capacity_ratio * p_itf[south] / qf[idx_rho][south])
    eig = maximum(eig_d, eig_u)

    # Linearize the interface fluxes.
    dflux_d = sg_itf[north] * a3_matvec(
        dqf_r[north], qf_r[north], p_itf[north], _trace_metric(m, 0)[north], _trace_metric(m, 1)[north], h33_itf[north]
    )
    dflux_u = sg_itf[south] * a3_matvec(
        dqf_r[south], qf_r[south], p_itf[south], _trace_metric(m, 0)[south], _trace_metric(m, 1)[south], h33_itf[south]
    )

    dfitf_full = torch.zeros_like(dqf)
    dfitf_full[north] = 0.5 * (dflux_d + dflux_u - eig * sg_itf[north] * (dqf_r[south] - dqf_r[north]))
    dfitf_full[south] = dfitf_full[north]

    # Return shared interface data used by the well-balanced vertical-momentum row.
    return dfx3, dfitf_full[_mid_k], (dqf, dq_itf, p_itf, eig, w_d, w_u)


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
    """Apply J1 to one or more directions on axis 1."""
    m = rhsobj.metric

    if base is None:
        base = j1_prepare(rhsobj, q)
    ops, pressure, q_itf_x3, q_itf_full_x3, wflux_pres_x3, wflux_pres_itf_x3, log_p, pressure_itf_x3 = base
    op_dz = ops.derivative_z
    op_corr = ops.correction_DU
    n = rhsobj.geom.num_solpts**2

    dfx3, dfitf, shared = _vertical_flux_jvp(rhsobj, q, dq, ops, pressure, q_itf_x3, q_itf_full_x3)
    dqf, dq_itf, p_itf, eig, w_d, w_u = shared
    q_itf = q_itf_x3

    south = numpy.s_[..., 1:, :, :, :n]
    north = numpy.s_[..., :-1, :, :, n:]
    h33_itf = m.h_contra_itf_k_new[2, 2]
    sg_itf = m.sqrtG_itf_k_new
    qf = q_itf_full_x3

    # Well-balanced vertical-momentum terms.
    wfp = wflux_pres_x3
    w_presa_base = apply_op(wfp, op_dz)
    apply_op(wflux_pres_itf_x3, op_corr, out=w_presa_base, beta=1.0)
    logp_bdy = torch.log(pressure_itf_x3)
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

    dwadv_full = torch.zeros_like(dqf[idx_rho])
    dwadv_d = sg_itf[north] * (2.0 * w_d * drw_d - w_d**2 * dqf[idx_rho][north])
    dwadv_u = sg_itf[south] * (2.0 * w_u * drw_u - w_u**2 * dqf[idx_rho][south])
    dwadv_full[north] = 0.5 * (dwadv_d + dwadv_u - eig * sg_itf[north] * (drw_u - drw_d))
    dwadv_full[south] = dwadv_full[north]

    Gd = sg_itf[north] * h33_itf[north]
    Gu = sg_itf[south] * h33_itf[south]
    dwpres_full = torch.zeros_like(dqf[idx_rho])
    dwpres_full[north] = 0.5 * Gu / p_d * (dp_u - (p_u / p_d) * dp_d)
    dwpres_full[south] = 0.5 * Gd / p_u * (dp_d - (p_d / p_u) * dp_u)

    dlogp_itf = heat_capacity_ratio * dq_itf[idx_rho_theta] / q_itf[idx_rho_theta]

    # Assemble the vertical-momentum perturbation.
    dw_df3 = apply_op(dwadv_x3, op_dz)
    apply_op(dwadv_full[_mid_k], op_corr, out=dw_df3, beta=1.0)
    dw_presa = apply_op(dwpres_full[_mid_k], op_corr)
    dw_presb = apply_op(dlogp, op_dz)
    apply_op(dlogp_itf, op_corr, out=dw_presb, beta=1.0)
    dw_presb = dw_presb * wfp
    drhs_w = dw_df3 + dp * (w_presa_base + w_presb_base) + pressure * (dw_presa + dw_presb)

    # Apply the metric factor and gravity Jacobian.
    out = apply_op(dfx3, op_dz)
    apply_op(dfitf, op_corr, out=out, beta=1.0)
    out[idx_rho_u3] = drhs_w
    out *= -m.inv_sqrtG_new
    out[idx_rho_u3] -= m.inv_dzdeta_new * gravity * m.inv_sqrtG_new * ((m.sqrtG_new * dq[idx_rho]) @ ops.highfilter_k)
    # Assign horizontal-momentum derivatives to J2.
    out[idx_rho_u1] = 0.0
    out[idx_rho_u2] = 0.0
    return out


# Column-wise block-tridiagonal assembly of J1.


def _col_dims(rhsobj, q):
    ns = rhsobj.geom.num_solpts
    nv, nz, ny, nx, _ = q.shape
    return ns, ns * ns, nv, nz, ny, nx, nv * ns, ny * nx * ns * ns  # ns, nh, nv, nz, ny, nx, m, ncol


def state_to_col(rhsobj, x):
    """Convert grid layout to column layout."""
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, x)
    x6 = x.reshape(nv, nz, ny, nx, ns, nh)
    p = torch.permute(x6, (2, 3, 5, 1, 0, 4))
    return p.reshape(ncol, nz, m)


def col_to_state(rhsobj, xc, ref):
    """Inverse of state_to_col."""
    ns, nh, nv, nz, ny, nx, _, _ = _col_dims(rhsobj, ref)
    p = xc.reshape(ny, nx, nh, nz, nv, ns)
    x6 = torch.permute(p, (4, 3, 0, 1, 5, 2))
    return x6.reshape(ref.shape)


def batched_state_to_col(rhsobj, x, ref):
    """Convert batched states to column layout."""
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, ref)
    nb = x.shape[1]
    x7 = x.reshape(nv, nb, nz, ny, nx, ns, nh)
    p = torch.permute(x7, (1, 3, 4, 6, 2, 0, 5))  # (nb, ny, nx, nh, nz, nv, ns)
    return p.reshape(nb, ncol, nz, m)


def assemble_j1_blocks(rhsobj, q, batch=3):
    """Assemble J1 blocks by three-color probing."""
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, q)
    L = torch.zeros((ncol, nz, m, m), dtype=q.dtype)
    A = torch.zeros((ncol, nz, m, m), dtype=q.dtype)
    U = torch.zeros((ncol, nz, m, m), dtype=q.dtype)

    base = j1_prepare(rhsobj, q)
    probes = [(v, s) for v in range(nv) for s in range(ns)]
    nj = len(probes)
    batch = max(1, min(batch, nj))

    for c in range(3):
        # A color has disjoint block rows.
        eA = torch.asarray([e for e in range(nz) if e % 3 == c])
        eL = torch.asarray([e for e in range(1, nz) if (e - 1) % 3 == c])
        eU = torch.asarray([e for e in range(nz - 1) if (e + 1) % 3 == c])
        for start in range(0, nj, batch):
            chunk = probes[start : start + batch]
            nb = len(chunk)
            dq = torch.zeros((nv, nb, nz, ny, nx, ns * nh), dtype=q.dtype)
            for p, (v, s) in enumerate(chunk):
                dq[v, p, c::3, :, :, s * nh : (s + 1) * nh] = 1.0
            Pc = batched_state_to_col(rhsobj, implicit_jvp(rhsobj, q, dq, base), q)
            js = slice(start, start + nb)
            for blk, es in ((A, eA), (L, eL), (U, eU)):
                if es.size:
                    blk[:, es, :, js] = torch.permute(Pc[:, :, es, :], (1, 2, 3, 0))
    return L, A, U


# Direct block assembly from one-dimensional operators.


def _g2c(arr, ns, nh, ny, nx):
    """Convert a scalar grid array to column layout."""
    nz = arr.shape[0]
    a = arr.reshape(nz, ny, nx, ns, nh)
    return torch.permute(a, (1, 2, 4, 0, 3)).reshape(ny * nx * nh, nz, ns)


def _i2c(arr, nh, ny, nx):
    """Convert scalar interfaces to column layout."""
    npd = arr.shape[0]
    a = arr.reshape(npd, ny, nx, 2, nh)
    return torch.permute(a, (1, 2, 4, 0, 3)).reshape(ny * nx * nh, npd, 2)


def _i5c(arr, nh, ny, nx):
    """Convert state interfaces to column layout."""
    nv, npd = arr.shape[0], arr.shape[1]
    a = arr.reshape(nv, npd, ny, nx, 2, nh)
    return torch.permute(a, (2, 3, 5, 1, 4, 0)).reshape(ny * nx * nh, npd, 2, nv)


def assemble_vertical_blocks(rhsobj, q):
    """Assemble the exact block-tridiagonal vertical-flux Jacobian."""
    met = rhsobj.metric
    ns, nh, nv, nz, ny, nx, m, ncol = _col_dims(rhsobj, q)
    gam = heat_capacity_ratio
    rw, rt, rr = idx_rho_u3, idx_rho_theta, idx_rho
    dt_ = q.dtype

    base = j1_prepare(rhsobj, q)
    ops, pressure, q_itf_x3, qf, wflux_pres_x3, wflux_pres_itf_x3, log_p, pressure_itf_x3 = base

    # One-dimensional reference operators.
    Dz = ops.diff_solpt  # D_int
    eLv, eRv = ops.extrap_down, ops.extrap_up  # e_L, e_R
    dLv, dRv = ops.correction[:, 0], ops.correction[:, 1]  # d_L tilde, d_R tilde
    HF = ops.highfilter
    eye_s = torch.eye(ns, dtype=dt_)

    def g2c(a):
        return _g2c(a, ns, nh, ny, nx)

    # Volume data in column layout.
    qc = state_to_col(rhsobj, q).reshape(ncol, nz, nv, ns)
    pc = g2c(pressure)
    sgc, isgc, idzc = g2c(met.sqrtG_new), g2c(met.inv_sqrtG_new), g2c(met.inv_dzdeta_new)
    A3s = ad_matrix(
        torch.permute(qc, (2, 0, 1, 3)),
        pc,
        2,
        g2c(met.h_contra_new[0, 2]),
        g2c(met.h_contra_new[1, 2]),
        g2c(met.h_contra_new[2, 2]),
    )

    # Padded traces in column layout.
    qfc = _i5c(qf, nh, ny, nx)
    sgfc = _i2c(met.sqrtG_itf_k_new, nh, ny, nx)
    h33fc = _i2c(_trace_metric(met, 2), nh, ny, nx)
    p_itf_c = p0 * torch.exp((cpd / cvd) * torch.log(qfc[..., rt] * (Rd / p0)))
    qfc_r = qfc.copy()  # M-reflected ghosts for plain rows
    qfc_r[:, 0, :, rw] *= -1.0
    qfc_r[:, -1, :, rw] *= -1.0
    A3f = ad_matrix(  # (ncol, nz+2, 2, 5, 5)
        torch.permute(qfc_r, (3, 0, 1, 2)),
        p_itf_c,
        2,
        _i2c(_trace_metric(met, 0), nh, ny, nx),
        _i2c(_trace_metric(met, 1), nh, ny, nx),
        h33fc,
    )

    # Pair upper and lower traces at each interface.
    dn, up = numpy.s_[:, : nz + 1, 1], numpy.s_[:, 1:, 0]
    sg_n, sg_s = sgfc[dn], sgfc[up]
    rho_d, rho_u = qfc[dn][..., rr], qfc[up][..., rr]
    w_d, w_u = qfc[dn][..., rw] / rho_d, qfc[up][..., rw] / rho_u  # eig/wb use unreflected traces
    p_d, p_u = p_itf_c[dn], p_itf_c[up]
    rt_d, rt_u = qfc[dn][..., rt], qfc[up][..., rt]
    s_d = torch.sqrt(h33fc[dn] * gam * p_d / rho_d)  # acoustic term c_s sqrt(h33), down trace
    s_u = torch.sqrt(h33fc[up] * gam * p_u / rho_u)  # up trace
    a_d = torch.abs(w_d) + s_d
    a_u = torch.abs(w_u) + s_u
    eig = maximum(a_d, a_u)
    I5 = torch.eye(nv, dtype=dt_).reshape(1, 1, nv, nv)
    lam = (eig * sg_n)[..., None, None]
    Bp = 0.5 * (sg_n[..., None, None] * A3f[dn] + lam * I5)
    Bm = 0.5 * (sg_s[..., None, None] * A3f[up] - lam * I5)

    # Include the derivative of the maximum Rusanov speed; the lower trace wins ties.
    down_wins = a_d >= a_u
    w_w = torch.where(down_wins, w_d, w_u)
    rho_w = torch.where(down_wins, rho_d, rho_u)
    rt_w = torch.where(down_wins, rt_d, rt_u)
    s_w = torch.where(down_wins, s_d, s_u)
    sgn_w = torch.sgn(w_w)
    glam = torch.zeros_like(qfc[dn])  # grad_q lambda at the winning trace, (ncol, nz+1, nv)
    glam[..., rr] = -sgn_w * w_w / rho_w - s_w / (2.0 * rho_w)
    glam[..., rw] = sgn_w / rho_w
    glam[..., rt] = gam * s_w / (2.0 * rt_w)
    jump = qfc[up] - qfc[dn]  # q_L^{e+1} - q_R^e (unreflected traces), (ncol, nz+1, nv)
    Lam = (-0.5 * sg_n)[..., None, None] * jump[..., :, None] * glam[..., None, :]
    zero_lam = torch.zeros_like(Lam)
    Bp = Bp + torch.where(down_wins[..., None, None], Lam, zero_lam)  # winning trace on the down side
    Bm = Bm + torch.where(down_wins[..., None, None], zero_lam, Lam)  # winning trace on the up side

    # Linearize logarithmic density extrapolation.
    qitfc = _i5c(q_itf_x3, nh, ny, nx)
    EL = torch.zeros((ncol, nz, nv, ns), dtype=dt_) + eLv
    ER = torch.zeros((ncol, nz, nv, ns), dtype=dt_) + eRv
    for var in (rr, rt):
        EL[:, :, var, :] = eLv * (qitfc[:, :, 0, var][..., None] / qc[:, :, var, :])
        ER[:, :, var, :] = eRv * (qitfc[:, :, 1, var][..., None] / qc[:, :, var, :])

    # Conservative rows.
    shp = (ncol, nz, nv, ns, nv, ns)
    L = torch.zeros(shp, dtype=dt_)
    A = torch.zeros(shp, dtype=dt_)
    U = torch.zeros(shp, dtype=dt_)

    # Volume contribution.
    A += torch.einsum("os,ces,cesij->ceiojs", Dz, sgc, A3s)

    # Interior interfaces.
    IT, ED, EU = slice(1, nz), slice(1, nz), slice(0, nz - 1)
    L[:, ED] += torch.einsum("o,ceij,cejs->ceiojs", dLv, Bp[:, IT], ER[:, EU])
    A[:, ED] += torch.einsum("o,ceij,cejs->ceiojs", dLv, Bm[:, IT], EL[:, ED])
    A[:, EU] += torch.einsum("o,ceij,cejs->ceiojs", dRv, Bp[:, IT], ER[:, EU])
    U[:, EU] += torch.einsum("o,ceij,cejs->ceiojs", dRv, Bm[:, IT], EL[:, ED])

    # Reflected wall states.
    Mv = torch.ones(nv, dtype=dt_)
    Mv[rw] = -1.0
    A[:, 0] += torch.einsum("o,cij,cjs->ciojs", dLv, Bp[:, 0] * Mv + Bm[:, 0], EL[:, 0])
    A[:, nz - 1] += torch.einsum("o,cij,cjs->ciojs", dRv, Bp[:, nz] + Bm[:, nz] * Mv, ER[:, nz - 1])

    # Well-balanced vertical-momentum row.
    L[:, :, rw] = 0.0
    A[:, :, rw] = 0.0
    U[:, :, rw] = 0.0

    rhoc, rtc = qc[:, :, rr, :], qc[:, :, rt, :]
    wc = qc[:, :, rw, :] / rhoc

    # Advective interior
    A[:, :, rw, :, rw, :] += torch.einsum("os,ces->ceos", Dz, sgc * 2.0 * wc)
    A[:, :, rw, :, rr, :] += torch.einsum("os,ces->ceos", Dz, -sgc * wc**2)

    # Advective interfaces
    cwp = torch.zeros((ncol, nz + 1, nv), dtype=dt_)
    cwm = torch.zeros((ncol, nz + 1, nv), dtype=dt_)
    cwp[..., rw] = 0.5 * (2.0 * sg_n * w_d + eig * sg_n)
    cwp[..., rr] = -0.5 * sg_n * w_d**2
    cwm[..., rw] = 0.5 * (2.0 * sg_s * w_u - eig * sg_n)
    cwm[..., rr] = -0.5 * sg_s * w_u**2
    # Restore the wave-speed derivative in the well-balanced row.
    lam_rw = (-0.5 * sg_n * jump[..., rw])[..., None] * glam
    zero_rw = torch.zeros_like(lam_rw)
    cwp = cwp + torch.where(down_wins[..., None], lam_rw, zero_rw)
    cwm = cwm + torch.where(down_wins[..., None], zero_rw, lam_rw)
    L[:, ED, rw] += torch.einsum("o,cej,cejs->ceojs", dLv, cwp[:, IT], ER[:, EU])
    A[:, ED, rw] += torch.einsum("o,cej,cejs->ceojs", dLv, cwm[:, IT], EL[:, ED])
    A[:, EU, rw] += torch.einsum("o,cej,cejs->ceojs", dRv, cwp[:, IT], ER[:, EU])
    U[:, EU, rw] += torch.einsum("o,cej,cejs->ceojs", dRv, cwm[:, IT], EL[:, ED])
    # Reflected wall states give a zero advective derivative.

    # Pressure perturbation.
    wfp = wflux_pres_x3
    w_presa_base = apply_op(wfp, ops.derivative_z)
    apply_op(wflux_pres_itf_x3, ops.correction_DU, out=w_presa_base, beta=1.0)
    w_presb_base = apply_op(log_p, ops.derivative_z)
    apply_op(torch.log(pressure_itf_x3), ops.correction_DU, out=w_presb_base, beta=1.0)
    w_presb_base = w_presb_base * wfp
    dpdrt = gam * pc / rtc
    A[:, :, rw, :, rt, :] += torch.einsum("os,ces->ceos", eye_s, g2c(w_presa_base + w_presb_base) * dpdrt)

    # Metric-pressure contribution.
    Gd, Gu = sg_n * h33fc[dn], sg_s * h33fc[up]
    cp_d, cp_u = gam * p_d / rt_d, gam * p_u / rt_u
    alpha_d = 0.5 * Gd / p_u * cp_d
    alpha_u = -0.5 * Gd / p_u * (p_d / p_u) * cp_u
    beta_u = 0.5 * Gu / p_d * cp_u
    beta_d = -0.5 * Gu / p_d * (p_u / p_d) * cp_d
    pdL, pdR = pc * dLv, pc * dRv
    L[:, ED, rw, :, rt, :] += torch.einsum("ceo,ce,ces->ceos", pdL[:, ED], alpha_d[:, IT], ER[:, EU, rt])
    A[:, ED, rw, :, rt, :] += torch.einsum("ceo,ce,ces->ceos", pdL[:, ED], alpha_u[:, IT], EL[:, ED, rt])
    A[:, EU, rw, :, rt, :] += torch.einsum("ceo,ce,ces->ceos", pdR[:, EU], beta_d[:, IT], ER[:, EU, rt])
    U[:, EU, rw, :, rt, :] += torch.einsum("ceo,ce,ces->ceos", pdR[:, EU], beta_u[:, IT], EL[:, ED, rt])
    A[:, 0, rw, :, rt, :] += torch.einsum("co,c,cs->cos", pdL[:, 0], alpha_d[:, 0] + alpha_u[:, 0], EL[:, 0, rt])
    A[:, nz - 1, rw, :, rt, :] += torch.einsum(
        "co,c,cs->cos", pdR[:, nz - 1], beta_d[:, nz] + beta_u[:, nz], ER[:, nz - 1, rt]
    )

    # Local contribution.
    pw = pc * g2c(wfp)
    A[:, :, rw, :, rt, :] += torch.einsum("ceo,os,ces->ceos", pw, Dz, gam / rtc)
    A[:, :, rw, :, rt, :] += torch.einsum(
        "ceo,o,ces->ceos", pw, dLv, gam * EL[:, :, rt, :] / qitfc[:, :, 0, rt][..., None]
    )
    A[:, :, rw, :, rt, :] += torch.einsum(
        "ceo,o,ces->ceos", pw, dRv, gam * ER[:, :, rt, :] / qitfc[:, :, 1, rt][..., None]
    )

    # Apply the metric factor and filtered gravity.
    scale = (-isgc)[:, :, None, :, None, None]
    L *= scale
    A *= scale
    U *= scale
    A[:, :, rw, :, rr, :] -= torch.einsum("ceo,os,ces->ceos", idzc * gravity * isgc, HF, sgc)

    L, A, U = L.reshape(ncol, nz, m, m), A.reshape(ncol, nz, m, m), U.reshape(ncol, nz, m, m)

    if rhsobj.y_invariant_slab:
        # Each variable occupies ns consecutive rows.
        yrow = numpy.s_[..., idx_rho_u2 * ns : (idx_rho_u2 + 1) * ns, :]
        L[yrow] = 0.0
        A[yrow] = 0.0
        U[yrow] = 0.0

    return L, A, U


def _momentum_rows(ns: int) -> slice:
    """Return the horizontal-momentum rows in column layout."""
    return slice(idx_rho_u1 * ns, (idx_rho_u2 + 1) * ns)


def _retained_rows(ns: int, device) -> NDArray:
    """Return the column-layout rows of the variables f1 keeps: rho, rho_w and rho_theta."""
    return torch.cat(
        (
            torch.arange(idx_rho * ns, (idx_rho + 1) * ns, device=device),
            torch.arange(idx_rho_u3 * ns, (idx_rho_theta + 1) * ns, device=device),
        )
    )


def split_vertical_blocks(rhsobj, L, A, U):
    """Split the vertical blocks into the part each partition needs.

    Returns ``(momentum, retained)``. ``momentum`` holds the horizontal-momentum rows, which f2
    carries and whose columns span every variable. ``retained`` holds the square sub-blocks of the
    three variables f1 keeps, which is the system the column solve actually inverts: the momentum
    rows of J1 are empty and the retained rows do not depend on those variables, so the two sets of
    unknowns decouple exactly.
    """
    ns = rhsobj.geom.num_solpts
    rows = _momentum_rows(ns)
    keep = _retained_rows(ns, L.device)
    momentum = tuple(b[..., rows, :].clone() for b in (L, A, U))
    retained = tuple(b[..., keep, :][..., :, keep].clone() for b in (L, A, U))
    return momentum, retained


def solve_retained_columns(rhsobj, retained, bc, dt):
    """Solve (I - dt/2 J1) x = b in column layout using only the variables f1 retains.

    The horizontal-momentum rows of J1 are empty, so those rows of the system are the identity and
    their solution is the right-hand side unchanged; only the retained sub-system is inverted, which
    is (3/5)^3 of the block-Thomas work of the full five-variable form.
    """
    keep = _retained_rows(rhsobj.geom.num_solpts, bc.device)
    x = bc.clone()
    x[..., keep] = block_thomas_solve(rhsobj, *retained, bc[..., keep], dt)
    return x


def momentum_blocks_matvec(rhsobj, momentum, x):
    """Apply the horizontal-momentum rows of the vertical block operator to a state-layout vector."""
    Lm, Am, Um = momentum
    rows = _momentum_rows(rhsobj.geom.num_solpts)
    xc = state_to_col(rhsobj, x)

    out = torch.einsum("ceij,cej->cei", Am, xc)
    out[:, 1:] += torch.einsum("ceij,cej->cei", Lm[:, 1:], xc[:, :-1])
    out[:, :-1] += torch.einsum("ceij,cej->cei", Um[:, :-1], xc[:, 1:])

    full = torch.zeros_like(xc)
    full[..., rows] = out
    return col_to_state(rhsobj, full, x)


def assemble_j1_blocks_analytic(rhsobj, q):
    """Block-tridiagonal J1 in the full five-variable layout, with the rows f1 omits set to zero."""
    L, A, U = assemble_vertical_blocks(rhsobj, q)
    rows = _momentum_rows(rhsobj.geom.num_solpts)
    for blocks in (L, A, U):
        blocks[..., rows, :] = 0.0
    return L, A, U


def blocks_matvec(rhsobj, L, A, U, xc):
    """Apply block-tridiagonal operator per column."""
    out = torch.einsum("ceij,cej->cei", A, xc)
    out[:, 1:] += torch.einsum("ceij,cej->cei", L[:, 1:], xc[:, :-1])
    out[:, :-1] += torch.einsum("ceij,cej->cei", U[:, :-1], xc[:, 1:])
    return out


def block_thomas_solve(rhsobj, L, A, U, b, dt):
    """Solve (I - (dt/2) J1) x = b per column via block-Thomas algorithm."""
    a = 0.5 * dt
    _, nz, m, _ = A.shape
    out_dtype = b.dtype
    # Mixed mode uses float32 factors and one float64-residual refinement.
    refine = out_dtype == torch.float32
    acc = torch.float32 if refine else torch.float64
    eye = torch.eye(m, dtype=acc).reshape(1, m, m)

    def Am(e):
        return eye - a * A[:, e].astype(acc)

    def Lm(e):
        return (-a) * L[:, e].astype(acc)

    def Um(e):
        return (-a) * U[:, e].astype(acc)

    # Reuse diagonal-block factors during substitution.
    if hasattr(torch.linalg, "lu_factor") and hasattr(torch.linalg, "lu_solve"):

        def factor(mat):
            return torch.linalg.lu_factor(mat)

        def fsolve(fac, rhs):
            return torch.linalg.lu_solve(fac[0], fac[1], rhs)

    else:

        def factor(mat):
            return mat

        def fsolve(fac, rhs):
            return torch.linalg.solve(fac, rhs)

    b = b.astype(acc)
    fac = [None] * nz
    d = [None] * nz
    fac[0] = factor(Am(0))
    d[0] = b[:, 0]
    for e in range(1, nz):
        # Update the diagonal block and right-hand side together.
        rhs_join = torch.concatenate([Um(e - 1), d[e - 1][..., None]], dim=-1)  # (ncol, m, m+1)
        sol = fsolve(fac[e - 1], rhs_join)
        T = sol[..., :m]  # C_{e-1}^{-1} Um_{e-1}
        y = sol[..., m]  # C_{e-1}^{-1} d_{e-1}
        fac[e] = factor(Am(e) - Lm(e) @ T)
        d[e] = b[:, e] - (Lm(e) @ y[..., None])[..., 0]

    def substitute(rhs):
        """Solve with the existing block factors."""
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
        # Form the float64 residual one element at a time to limit memory use.
        f64 = torch.float64
        eye64 = torch.eye(m, dtype=f64).reshape(1, m, m)
        r = [None] * nz
        for e in range(nz):
            r[e] = b[:, e].astype(f64) - ((eye64 - a * A[:, e].astype(f64)) @ x[e].astype(f64)[..., None])[..., 0]
            if e > 0:
                r[e] += a * (L[:, e].astype(f64) @ x[e - 1].astype(f64)[..., None])[..., 0]
            if e < nz - 1:
                r[e] += a * (U[:, e].astype(f64) @ x[e + 1].astype(f64)[..., None])[..., 0]
        dx = substitute([r[e].astype(acc) for e in range(nz)])
        x = [x[e] + dx[e] for e in range(nz)]

    return torch.stack(x, dim=1).astype(out_dtype)


def m_matvec(rhsobj, L, A, U, xc, dt):
    """Apply M = I - (dt/2) J1 to column vector xc."""
    return xc - 0.5 * dt * blocks_matvec(rhsobj, L, A, U, xc)


# Analytic flux-divergence contribution to J2.

_mid_i = numpy.s_[..., 1:-1, :]
_mid_j = numpy.s_[..., 1:-1, :, :]


def _p_itf(rt):
    return p0 * torch.exp((cpd / cvd) * torch.log(rt * (Rd / p0)))


def _hori_interface(direction, qf, dqf, sg, hci, left, right):
    """Exact differential of a horizontal Rusanov interface flux."""
    p_itf = _p_itf(qf[idx_rho_theta])
    mom = _MOM[direction]
    u_r = qf[mom][right] / qf[idx_rho][right]
    u_l = qf[mom][left] / qf[idx_rho][left]
    h00 = hci[direction, direction]
    sound_l = torch.sqrt(h00[left] * heat_capacity_ratio * p_itf[left] / qf[idx_rho][left])
    sound_r = torch.sqrt(h00[right] * heat_capacity_ratio * p_itf[right] / qf[idx_rho][right])
    eig_l = torch.abs(u_l) + sound_l
    eig_r = torch.abs(u_r) + sound_r
    eig = maximum(eig_l, eig_r)

    dflux_l = sg[left] * ad_matvec(
        dqf[left],
        qf[left],
        p_itf[left],
        direction,
        hci[direction, 0][left],
        hci[direction, 1][left],
        hci[direction, 2][left],
    )
    dflux_r = sg[right] * ad_matvec(
        dqf[right],
        qf[right],
        p_itf[right],
        direction,
        hci[direction, 0][right],
        hci[direction, 1][right],
        hci[direction, 2][right],
    )

    left_wins = eig_l >= eig_r
    rho_w = torch.where(left_wins, qf[idx_rho][left], qf[idx_rho][right])
    rt_w = torch.where(left_wins, qf[idx_rho_theta][left], qf[idx_rho_theta][right])
    u_w = torch.where(left_wins, u_l, u_r)
    sound_w = torch.where(left_wins, sound_l, sound_r)
    drho_w = torch.where(left_wins, dqf[idx_rho][left], dqf[idx_rho][right])
    dmom_w = torch.where(left_wins, dqf[mom][left], dqf[mom][right])
    drt_w = torch.where(left_wins, dqf[idx_rho_theta][left], dqf[idx_rho_theta][right])
    du_w = (dmom_w - u_w * drho_w) / rho_w
    dsound_w = 0.5 * sound_w * (heat_capacity_ratio * drt_w / rt_w - drho_w / rho_w)
    deig = torch.sgn(u_w) * du_w + dsound_w

    out = torch.zeros_like(qf)
    out[left] = 0.5 * (dflux_l + dflux_r - sg[left] * (eig * (dqf[right] - dqf[left]) + deig * (qf[right] - qf[left])))
    out[right] = out[left]
    return out


def _hori_wb_interface(direction, qf, dqf, sg, hci, left, right):
    """Differentiate the two interface fluxes used by the well-balanced rho-u3 row."""
    pressure = _p_itf(qf[idx_rho_theta])
    dpressure = heat_capacity_ratio * pressure * dqf[idx_rho_theta] / qf[idx_rho_theta]
    mom = _MOM[direction]
    u_l = qf[mom][left] / qf[idx_rho][left]
    u_r = qf[mom][right] / qf[idx_rho][right]
    du_l = (dqf[mom][left] - u_l * dqf[idx_rho][left]) / qf[idx_rho][left]
    du_r = (dqf[mom][right] - u_r * dqf[idx_rho][right]) / qf[idx_rho][right]

    sound_l = torch.sqrt(hci[direction, direction][left] * heat_capacity_ratio * pressure[left] / qf[idx_rho][left])
    sound_r = torch.sqrt(hci[direction, direction][right] * heat_capacity_ratio * pressure[right] / qf[idx_rho][right])
    eig_l = torch.abs(u_l) + sound_l
    eig_r = torch.abs(u_r) + sound_r
    left_wins = eig_l >= eig_r
    eig = maximum(eig_l, eig_r)
    deig_l = torch.sgn(u_l) * du_l + 0.5 * sound_l * (
        heat_capacity_ratio * dqf[idx_rho_theta][left] / qf[idx_rho_theta][left]
        - dqf[idx_rho][left] / qf[idx_rho][left]
    )
    deig_r = torch.sgn(u_r) * du_r + 0.5 * sound_r * (
        heat_capacity_ratio * dqf[idx_rho_theta][right] / qf[idx_rho_theta][right]
        - dqf[idx_rho][right] / qf[idx_rho][right]
    )
    deig = torch.where(left_wins, deig_l, deig_r)

    rw_l, rw_r = qf[idx_rho_u3][left], qf[idx_rho_u3][right]
    drw_l, drw_r = dqf[idx_rho_u3][left], dqf[idx_rho_u3][right]
    dadv = 0.5 * (
        sg[left] * (du_l * rw_l + u_l * drw_l)
        + sg[right] * (du_r * rw_r + u_r * drw_r)
        - sg[left] * (eig * (drw_r - drw_l) + deig * (rw_r - rw_l))
    )

    gp_l = sg[left] * hci[direction, 2][left]
    gp_r = sg[right] * hci[direction, 2][right]
    numerator = 0.5 * (gp_l * pressure[left] + gp_r * pressure[right])
    dnumerator = 0.5 * (gp_l * dpressure[left] + gp_r * dpressure[right])
    dpres_l = (dnumerator * pressure[left] - numerator * dpressure[left]) / pressure[left] ** 2
    dpres_r = (dnumerator * pressure[right] - numerator * dpressure[right]) / pressure[right] ** 2

    adv = torch.zeros_like(qf[idx_rho])
    pres = torch.zeros_like(adv)
    adv[left], adv[right] = dadv, dadv
    pres[left], pres[right] = dpres_l, dpres_r
    return adv, pres


def j2_prepare(rhsobj, q, momentum_blocks=None):
    """Precompute the state and vertical blocks used by ``j2_flux_matvec``."""
    if momentum_blocks is None:
        L, A, U = assemble_vertical_blocks(rhsobj, q)
        # split_vertical_blocks returns (momentum, retained); only the momentum rows belong to J2.
        momentum_blocks, _ = split_vertical_blocks(rhsobj, L, A, U)
    rhsobj.horizontal_flux_div(q)
    return (
        rhsobj.ops,
        rhsobj.pressure.copy(),
        rhsobj.q_itf_x1.copy(),
        rhsobj.q_itf_x2.copy(),
        rhsobj.q_itf_full_x1.copy(),
        rhsobj.q_itf_full_x2.copy(),
        momentum_blocks,
    )


def j2_flux_matvec(rhsobj, q, dq, base=None):
    """Apply the flux contribution to J2."""
    m = rhsobj.metric
    if base is None:
        base = j2_prepare(rhsobj, q)
    ops, pressure, qitf1, qitf2, qf1, qf2, momentum_blocks = base
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
    dqf1 = torch.zeros_like(qf1)
    dqf2 = torch.zeros_like(qf2)
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
    dfitf1 = _hori_interface(0, qf1, dqf1, m.sqrtG_itf_i_new, m.h_contra_itf_i_new, east, west)
    dfitf2 = _hori_interface(1, qf2, dqf2, m.sqrtG_itf_j_new, m.h_contra_itf_j_new, north, south)

    apply_op(dfitf1[_mid_i], ops.correction_WE, out=out, beta=1.0)
    apply_op(dfitf2[_mid_j], ops.correction_SN, out=out, beta=1.0)

    out *= -m.inv_sqrtG_new

    # The vertical blocks already include the outer metric factor.
    out += momentum_blocks_matvec(rhsobj, momentum_blocks, dq)

    # Replace the plain vertical-momentum row with its well-balanced differential.
    rw = idx_rho_u3
    rt = idx_rho_theta
    gp1 = m.sqrtG_new * hc[0, 2]
    gp2 = m.sqrtG_new * hc[1, 2]
    u1 = q[idx_rho_u1] / q[idx_rho]
    u2 = q[idx_rho_u2] / q[idx_rho]
    du1 = (dq[idx_rho_u1] - u1 * dq[idx_rho]) / q[idx_rho]
    du2 = (dq[idx_rho_u2] - u2 * dq[idx_rho]) / q[idx_rho]
    dadv1 = m.sqrtG_new * (du1 * q[rw] + u1 * dq[rw])
    dadv2 = m.sqrtG_new * (du2 * q[rw] + u2 * dq[rw])
    dadv = apply_op(dadv1, ops.derivative_x)
    apply_op(dadv2, ops.derivative_y, out=dadv, beta=1.0)

    dwadv1, dwpres1 = _hori_wb_interface(0, qf1, dqf1, m.sqrtG_itf_i_new, m.h_contra_itf_i_new, east, west)
    dwadv2, dwpres2 = _hori_wb_interface(1, qf2, dqf2, m.sqrtG_itf_j_new, m.h_contra_itf_j_new, north, south)
    apply_op(dwadv1[_mid_i], ops.correction_WE, out=dadv, beta=1.0)
    apply_op(dwadv2[_mid_j], ops.correction_SN, out=dadv, beta=1.0)

    p1 = _p_itf(qf1[rt])
    p2 = _p_itf(qf2[rt])
    wp1_l = 0.5 * (
        m.sqrtG_itf_i_new[east] * m.h_contra_itf_i_new[0, 2][east] * p1[east]
        + m.sqrtG_itf_i_new[west] * m.h_contra_itf_i_new[0, 2][west] * p1[west]
    )
    wp2_l = 0.5 * (
        m.sqrtG_itf_j_new[north] * m.h_contra_itf_j_new[1, 2][north] * p2[north]
        + m.sqrtG_itf_j_new[south] * m.h_contra_itf_j_new[1, 2][south] * p2[south]
    )
    wp1 = torch.zeros_like(qf1[idx_rho])
    wp2 = torch.zeros_like(qf2[idx_rho])
    wp1[east], wp1[west] = wp1_l / p1[east], wp1_l / p1[west]
    wp2[north], wp2[south] = wp2_l / p2[north], wp2_l / p2[south]

    logp = torch.log(pressure)
    cbase = apply_op(gp1, ops.derivative_x)
    apply_op(gp2, ops.derivative_y, out=cbase, beta=1.0)
    apply_op(wp1[_mid_i], ops.correction_WE, out=cbase, beta=1.0)
    apply_op(wp2[_mid_j], ops.correction_SN, out=cbase, beta=1.0)
    logp1 = torch.log(p1)
    logp2 = torch.log(p2)
    dlogp1 = heat_capacity_ratio * dqf1[rt] / qf1[rt]
    dlogp2 = heat_capacity_ratio * dqf2[rt] / qf2[rt]
    presb1 = apply_op(logp, ops.derivative_x)
    presb2 = apply_op(logp, ops.derivative_y)
    apply_op(logp1[_mid_i], ops.correction_WE, out=presb1, beta=1.0)
    apply_op(logp2[_mid_j], ops.correction_SN, out=presb2, beta=1.0)
    cbase += gp1 * presb1 + gp2 * presb2

    dlogp = heat_capacity_ratio * dq[rt] / q[rt]
    dc = apply_op(dwpres1[_mid_i], ops.correction_WE)
    apply_op(dwpres2[_mid_j], ops.correction_SN, out=dc, beta=1.0)
    dlog1 = apply_op(dlogp, ops.derivative_x)
    dlog2 = apply_op(dlogp, ops.derivative_y)
    apply_op(dlogp1[_mid_i], ops.correction_WE, out=dlog1, beta=1.0)
    apply_op(dlogp2[_mid_j], ops.correction_SN, out=dlog2, beta=1.0)
    dc += gp1 * dlog1 + gp2 * dlog2
    dp = heat_capacity_ratio * pressure * dq[rt] / q[rt]
    out[rw] = -m.inv_sqrtG_new * (dadv + dp * cbase + pressure * dc)

    if rhsobj.y_invariant_slab:
        out[idx_rho_u2] = 0.0  # The pinned tendency has zero derivative.

    return out


# Pointwise Jacobian of the non-stiff forcing.


def forcing_jac_prepare(rhsobj, q):
    """Precompute state-dependent coefficients for ``forcing_jvp``."""
    rho = q[idx_rho]
    u = (q[idx_rho_u1] / rho, q[idx_rho_u2] / rho, q[idx_rho_u3] / rho)
    rt = q[idx_rho_theta]
    # Match the pointwise equation of state.
    p = p0 * torch.exp(heat_capacity_ratio * torch.log((Rd / p0) * rt))
    dpdrt = heat_capacity_ratio * p / rt

    ray = None
    case = getattr(rhsobj.pde, "case_number", None)
    if case in (21, 22):
        from ..init.dcmip import dcmip_schar_damping_coeffs

        rate, u1ref, u2ref, u3ref = dcmip_schar_damping_coeffs(rhsobj.metric, rhsobj.pde.geometry, shear=(case == 22))
        ray = (rate, (u1ref, u2ref, u3ref))

    return u, dpdrt, ray


def forcing_jvp(rhsobj, q, v, base=None):
    """Apply the pointwise Jacobian of ``forcing_only``."""
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

    out = torch.zeros_like(v)
    for d, row in enumerate(_MOM):
        c01, c02, c03 = ch[d, 0], ch[d, 1], ch[d, 2]
        c11, c12, c13 = ch[d, 3], ch[d, 4], ch[d, 5]
        c22, c23, c33 = ch[d, 6], ch[d, 7], ch[d, 8]

        # Momentum row in conserved variables.
        a_r = -(
            c11 * u1 * u1
            + 2.0 * c12 * u1 * u2
            + 2.0 * c13 * u1 * u3
            + c22 * u2 * u2
            + 2.0 * c23 * u2 * u3
            + c33 * u3 * u3
        )
        a_m1 = 2.0 * c01 + 2.0 * (c11 * u1 + c12 * u2 + c13 * u3)
        a_m2 = 2.0 * c02 + 2.0 * (c12 * u1 + c22 * u2 + c23 * u3)
        a_m3 = 2.0 * c03 + 2.0 * (c13 * u1 + c23 * u2 + c33 * u3)
        a_rt = (c11 * h11 + 2.0 * c12 * h12 + 2.0 * c13 * h13 + c22 * h22 + 2.0 * c23 * h23 + c33 * h33) * dpdrt

        dF = a_r * dr + a_m1 * dm[0] + a_m2 * dm[1] + a_m3 * dm[2] + a_rt * drt

        if ray is not None:
            rate, uref = ray
            # Linear Rayleigh-sponge contribution.
            dF = dF + rate * (dm[d] - uref[d] * dr)

        out[row] = -dF

    if rhsobj.y_invariant_slab:
        out[idx_rho_u2] = 0.0  # The pinned tendency has zero derivative.

    return out
