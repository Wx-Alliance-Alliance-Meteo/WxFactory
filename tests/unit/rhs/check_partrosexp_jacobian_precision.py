"""Accuracy of the PartRosExp2 partitions, Jacobians and column solve on DCMIP 2-1.

    ./scripts/run.sh -n 6 python tests/unit/rhs/check_partrosexp_jacobian_precision.py double
    ./scripts/run.sh -n 6 python tests/unit/rhs/check_partrosexp_jacobian_precision.py mixed

The double pass checks the invariants that must hold at any precision and writes a reference to the
scratch directory. The mixed pass reloads it, repeats the work in single precision and reports how
far each quantity has moved. Run the double pass first; the reference has to be regenerated whenever
the partition changes, since the states it stores lie on the model trajectory.

Two passes rather than one process because ``precision`` is a property of the device and the RHS
holds its metric terms at that precision, so a single Simulation cannot evaluate the same discrete
operator both ways. Reloading the double-precision state also keeps the base point bit-identical
between the passes, leaving arithmetic precision as the only difference.

Set WX_JACOBIAN_SCRATCH to choose where the reference is written, WX_JACOBIAN_DEVICE (cpu/cuda) and
WX_JACOBIAN_NEH to override the horizontal resolution. A full-resolution double-precision run does
not fit on an 8 GB GPU, so the double pass usually wants WX_JACOBIAN_DEVICE=cpu.

This is a standalone script rather than a discovered test because differentiable mode is
process-wide and cannot be switched off once enabled (see check_ad_jacobian.py), and because it needs
6 ranks and a full DCMIP 2-1 grid.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))

import torch
import torch.autograd.forward_ad as fwad
from mpi4py import MPI

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.definitions import (
    idx_rho,
    idx_rho_u1,
    idx_rho_u2,
    idx_rho_u3,
)
from wx_factory.rhs.vertical_jacobian import (
    assemble_j1_blocks_analytic,
    assemble_vertical_blocks,
    blocks_matvec,
    col_to_state,
    forcing_jac_prepare,
    forcing_jvp,
    j2_flux_matvec,
    j2_prepare,
    m_matvec,
    solve_retained_columns,
    split_vertical_blocks,
    state_to_col,
)
from wx_factory.simulation import Simulation

VARIABLES = ["rho", "rho_u1", "rho_u2", "rho_w", "rho_theta"]

# The step size the reported quantities are formed at. dt reaches the Jacobians only as a scale
# factor, but it does set the state the trajectory reaches.
DT = 5.0
NUM_STEPS = 15

comm = MPI.COMM_WORLD
scratch = os.environ.get("WX_JACOBIAN_SCRATCH", "/tmp/jacobian_precision")
os.makedirs(scratch, exist_ok=True)


# ---------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------


def build_simulation(precision: str) -> Simulation:
    """A DCMIP 2-1 Simulation at the requested precision, with output disabled."""
    here = os.path.dirname(os.path.realpath(__file__))
    text = readfile(os.path.join(here, "..", "..", "..", "config", "dcmip21_ROSEXP.ini"))
    text = text.replace("precision = mixed", f"precision = {precision}")
    # Forward AD supplies the reference the analytic Jacobians are checked against.
    text = text.replace("jacobian_method = fd", "jacobian_method = ad")
    # Pin dt so the measurements stay comparable when the base config changes.
    text = re.sub(r"(?m)^dt\s*=.*$", f"dt = {DT}", text)
    text = text.replace(
        "num_elements_horizontal = 20", f"num_elements_horizontal = {os.environ.get('WX_JACOBIAN_NEH', '20')}"
    )
    text = text.replace("pytorch_device = cuda", f"pytorch_device = {os.environ.get('WX_JACOBIAN_DEVICE', 'cuda')}")
    text = text.replace("output_freq = 100", "output_freq = 0")
    text = text.replace("base_output_file = dcmip21_rosexp", f"base_output_file = diag_{os.getpid()}")
    return Simulation(Configuration(text, load_default_schema()), quiet=True)


def path(name: str) -> str:
    return os.path.join(scratch, f"{name}_rank{comm.rank}.pt")


def progress(message: str) -> None:
    if comm.rank == 0:
        print(f"... {message}", flush=True)


def report(lines) -> None:
    """Print on rank 0 only. Every value must already be computed on all ranks: the norms below are
    collective, and calling one inside a rank-0 block deadlocks."""
    if comm.rank == 0:
        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------------------------
# States and norms
# ---------------------------------------------------------------------------------------------


def perturbed_state(Q: torch.Tensor, seed: int = 1234, w_speed: float = 2.0) -> torch.Tensor:
    """A state with single-signed velocities, away from the u = 0 kink of the Rusanov flux.

    The DCMIP 2-1 initial state has w = 0 everywhere, which is exactly where |u3| is not
    differentiable, so comparisons there measure the kink rather than the quantity of interest. This
    state is synthetic and far more energetic than anything physical; treat it as a smoothness
    control, not as an operating point.
    """
    Q = Q.clone()
    gen = torch.Generator().manual_seed(seed)
    noise = torch.rand(Q.shape, generator=gen, dtype=Q.dtype, device="cpu").to(Q.device) - 0.5
    Q = Q * (1.0 + 0.02 * noise)  # a few percent keeps rho and rho_theta positive
    for idx in (idx_rho_u1, idx_rho_u2, idx_rho_u3):
        vfac = 0.5 + torch.rand(Q[idx_rho].shape, generator=gen, dtype=Q.dtype, device="cpu").to(Q.device)
        Q[idx] = w_speed * Q[idx_rho] * vfac
    return Q


def evolved_state(sim) -> torch.Tensor:
    """The state PartRosExp2 reaches after NUM_STEPS steps, i.e. a realistic operating point."""
    Q = sim.initial_state.Q.clone()
    for step in range(NUM_STEPS):
        Q = sim.integrator.step(Q, DT)
        w = comm.allreduce((Q[idx_rho_u3] / Q[idx_rho]).abs().max().item(), op=MPI.MAX)
        progress(f"step {step + 1}/{NUM_STEPS}  max|w| = {w:.4e} m/s")
    return Q


CASES = {
    "initial state (w = 0, on the |u3| kink)": lambda sim: sim.initial_state.Q.clone(),
    "perturbed state (synthetic, large w)": lambda sim: perturbed_state(sim.initial_state.Q),
    f"step {NUM_STEPS} (realistic operating point)": evolved_state,
}


def global_norm(x: torch.Tensor) -> float:
    """Norm over the whole distributed state, so the reported errors are the true global ones."""
    local = torch.linalg.norm(x.to(torch.float64)).item() ** 2
    return comm.allreduce(local, op=MPI.SUM) ** 0.5


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    den = global_norm(b)
    return global_norm(a - b) / den if den > 0.0 else global_norm(a - b)


def per_variable(a: torch.Tensor, b: torch.Tensor):
    """Row errors relative to the whole field.

    Not relative to each row's own norm: a row that is near zero -- meridional momentum in a
    non-sheared zonal flow, say -- then reports a large error for an absolutely tiny discrepancy.
    """
    whole = global_norm(b)
    return [global_norm(a[row] - b[row]) / whole for row in range(a.shape[0])]


def row_line(label: str, values) -> str:
    return f"      {label} " + "  ".join(f"{n}={e:.2e}" for n, e in zip(VARIABLES, values))


# ---------------------------------------------------------------------------------------------
# Quantities under test
# ---------------------------------------------------------------------------------------------


def analytic_actions(rhs, Q: torch.Tensor, v: torch.Tensor):
    """J1 . v and J2 . v exactly as PartRosExp2 forms them."""
    L, A, U = assemble_vertical_blocks(rhs, Q)
    momentum, _ = split_vertical_blocks(rhs, L, A, U)

    j1_blocks = assemble_j1_blocks_analytic(rhs, Q)
    j1v = col_to_state(rhs, blocks_matvec(rhs, *j1_blocks, state_to_col(rhs, v)), v)

    j2v = j2_flux_matvec(rhs, Q, v, j2_prepare(rhs, Q, momentum))
    j2v = j2v + forcing_jvp(rhs, Q, v, forcing_jac_prepare(rhs, Q))
    return j1v, j2v


def ad_actions(rhs, Q: torch.Tensor, v: torch.Tensor):
    """The same actions by forward-mode AD, which is the reference."""

    def tangent_of(func):
        with fwad.dual_level():
            return fwad.unpack_dual(func(fwad.make_dual(Q, v))).tangent.clone()

    return tangent_of(rhs.implicit), tangent_of(rhs.explicit), tangent_of(rhs)


def column_solve(rhs, Q: torch.Tensor):
    """Solve (I - dt/2 J1) x = dt f1, the system PartRosExp2 inverts, and its backward error.

    The load is the one the integrator feeds the solve up to the phi term, so the conditioning seen
    here is that of a real step.
    """
    L, A, U = assemble_vertical_blocks(rhs, Q)
    _, retained = split_vertical_blocks(rhs, L, A, U)

    bc = state_to_col(rhs, DT * rhs.implicit(Q))
    xc = solve_retained_columns(rhs, retained, bc, DT)

    j1_blocks = tuple(b.to(torch.float64) for b in assemble_j1_blocks_analytic(rhs, Q))
    residual = m_matvec(rhs, *j1_blocks, xc.to(torch.float64), DT) - bc.to(torch.float64)
    return bc, xc, global_norm(residual) / global_norm(bc)


# ---------------------------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------------------------


def run_double() -> None:
    """Check the precision-independent invariants and write the reference."""
    progress("building the double-precision simulation")
    sim = build_simulation("double")
    rhs = sim.rhs.full
    saved = {}

    for tag, make in CASES.items():
        progress(f"double pass: {tag}")
        Q = make(sim).to(torch.float64)
        gen = torch.Generator().manual_seed(9)
        v = (torch.rand(Q.shape, generator=gen, dtype=Q.dtype, device="cpu").to(Q.device) - 0.5) * Q.abs()

        full = rhs(Q).clone()
        f1 = rhs.implicit(Q).clone()
        f2 = rhs.explicit(Q).clone()
        j1_an, j2_an = analytic_actions(rhs, Q, v)
        j1_ad, j2_ad, full_ad = ad_actions(rhs, Q, v)
        bc, xc, backward = column_solve(rhs, Q)

        # f1 + f2 must reproduce the unsplit operator, and each Jacobian must differentiate the
        # partition it belongs to. These hold at any precision; a failure is a structural bug.
        identity = relative(f1 + f2, full)
        j1_err, j2_err = relative(j1_an, j1_ad), relative(j2_an, j2_ad)
        sum_err = relative(j1_an + j2_an, full_ad)

        # f1 keeps no horizontal-momentum rows, and the rows it keeps do not depend on those
        # variables, which is what lets the column solve drop to three variables.
        ns = rhs.geom.num_solpts
        mom = slice(idx_rho_u1 * ns, (idx_rho_u2 + 1) * ns)
        j1_blocks = assemble_j1_blocks_analytic(rhs, Q)
        whole = max(global_norm(b) for b in j1_blocks)
        rows = max(global_norm(b[..., mom, :]) for b in j1_blocks) / whole
        cols = max(global_norm(b[..., :, mom]) for b in j1_blocks) / whole

        # Each partition should be individually balanced on a state in equilibrium: a large factor
        # means f1 and f2 cancel, and a partitioned integrator destroys that cancellation.
        split = []
        for row in range(full.shape[0]):
            n_full = global_norm(full[row])
            worst = max(global_norm(f1[row]), global_norm(f2[row]))
            split.append(worst / n_full if n_full > 0 else float("inf"))

        saved[tag] = {
            "Q": Q.cpu(),
            "v": v.cpu(),
            "f1": f1.cpu(),
            "f2": f2.cpu(),
            "j1_ad": j1_ad.cpu(),
            "j2_ad": j2_ad.cpu(),
            "thomas_b": bc.cpu(),
            "thomas_x": xc.cpu(),
        }

        report(
            [
                f"\n[double] {tag}",
                f"    ||f1 + f2 - full|| / ||full||   : {identity:.3e}",
                f"    J1 vs AD of implicit            : {j1_err:.3e}",
                f"    J2 vs AD of explicit            : {j2_err:.3e}",
                f"    J1 + J2 vs AD of full           : {sum_err:.3e}",
                f"    J1 blocks, momentum rows / cols : {rows:.3e} / {cols:.3e}",
                f"    column solve backward error     : {backward:.3e}",
                row_line("balance split factor", split),
            ]
        )

    torch.save(saved, path("reference"))
    comm.Barrier()
    progress(f"reference written to {scratch}")


def run_mixed() -> None:
    """Repeat the same quantities in single precision, against the reference."""
    progress("building the single-precision simulation")
    sim = build_simulation("mixed")
    rhs = sim.rhs.full
    reference = torch.load(path("reference"), weights_only=True)

    for tag in CASES:
        progress(f"float32 pass: {tag}")
        ref = reference[tag]
        Q = ref["Q"].to(torch.float32).to(sim.device.torch_device)
        v = ref["v"].to(torch.float32).to(sim.device.torch_device)

        j1_an, j2_an = analytic_actions(rhs, Q, v)
        j1_ad, j2_ad, _ = ad_actions(rhs, Q, v)
        _, xc, backward = column_solve(rhs, Q)

        def against(x, key, ref=ref):
            return relative(x.to(torch.float64), ref[key].to(sim.device.torch_device))

        # Splitting the Jacobian error against the float32 AD as well as the float64 reference
        # separates what the closed form costs from what any float32 evaluation costs.
        lines = [
            f"\n[float32] {tag}",
            f"    J1 analytic vs AD f64           : {against(j1_an, 'j1_ad'):.3e}",
            f"    J1 AD f32   vs AD f64           : {against(j1_ad, 'j1_ad'):.3e}",
            f"    J2 analytic vs AD f64           : {against(j2_an, 'j2_ad'):.3e}",
            f"    J2 AD f32   vs AD f64           : {against(j2_ad, 'j2_ad'):.3e}",
            f"    column solve backward error     : {backward:.3e}",
            f"    column solve x vs x_f64         : {against(xc, 'thomas_x'):.3e}",
        ]

        # The partitions are evaluated in double and cast back, so `implicit`/`explicit` are called
        # directly here to measure what single precision alone would cost.
        for name, single, key in (("f1", rhs.implicit, "f1"), ("f2", rhs.explicit, "f2")):
            value = single(Q).clone().to(torch.float64)
            target = ref[key].to(sim.device.torch_device)
            lines.append(f"    {name} in float32 vs f64          : {relative(value, target):.3e}")
            lines.append(row_line(f"{name} by row", per_variable(value, target)))

        report(lines)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "double"
    if mode == "double":
        run_double()
    elif mode == "mixed":
        run_mixed()
    else:
        raise SystemExit(f"unknown mode {mode!r}; expected 'double' or 'mixed'")
