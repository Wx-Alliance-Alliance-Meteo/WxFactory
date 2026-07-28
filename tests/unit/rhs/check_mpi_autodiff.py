"""Check the RHS derivative, forward and reverse, across cubed-sphere halo exchanges.

    WX_FACTORY_DIFFERENTIABLE=1 mpirun -n 6 python tests/unit/rhs/check_mpi_autodiff.py

This standalone test requires six ranks and differentiable mode at process startup.

Forward mode is checked against finite differences; reverse mode uses
``<J v, w> = <v, J^T w>``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))

import torch
import torch.autograd.forward_ad as fwad
from mpi4py import MPI

from wx_factory import process_topology
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_u3
from wx_factory.device import differentiable_mode
from wx_factory.simulation import Simulation

comm = MPI.COMM_WORLD

if not differentiable_mode():
    if comm.rank == 0:
        print("set WX_FACTORY_DIFFERENTIABLE=1 (before any tensor is created) and rerun")
    sys.exit(1)

here = os.path.dirname(os.path.realpath(__file__))
config = Configuration(readfile(os.path.join(here, "cubed_sphere_ad_config.ini")), load_default_schema())
sim = Simulation(config)
rhs = sim.rhs.full

# Use a smooth, rank-dependent state so the halo derivative is observable.
Q = sim.initial_state.Q.clone().to(torch.float64)
gen = torch.Generator().manual_seed(1234 + comm.rank)
Q = Q * (1.0 + 0.02 * (torch.rand(Q.shape, generator=gen, dtype=torch.float64) - 0.5))
for idx in (idx_rho_u1, idx_rho_u2, idx_rho_u3):
    Q[idx] = 2.0 * Q[idx_rho] * (0.5 + torch.rand(Q[idx_rho].shape, generator=gen, dtype=torch.float64))

v = (torch.rand(Q.shape, generator=gen, dtype=torch.float64) - 0.5) * Q.abs()
w = (torch.rand(Q.shape, generator=gen, dtype=torch.float64) - 0.5) * Q.abs()


def global_dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return comm.allreduce(float(torch.sum(a * b, dtype=torch.float64)))


def global_norm(x: torch.Tensor) -> float:
    return global_dot(x, x) ** 0.5


with fwad.dual_level():
    ad = fwad.unpack_dual(rhs(fwad.make_dual(Q, v))).tangent.clone()

scale = global_norm(ad)
report = [f"||AD tangent|| = {scale:.6e}"]

# Compare against central differences over a range of step sizes.
best = None
for eps in (1e-4, 1e-5, 1e-6):
    fd = (rhs(Q + eps * v) - rhs(Q - eps * v)) / (2.0 * eps)
    rel = global_norm(fd - ad) / scale
    best = rel if best is None else min(best, rel)
    report.append(f"  FD(eps={eps:.0e}) vs AD                : {rel:.3e}")

# Confirm that dropping halo tangents produces a detectable error.
process_topology._DIFFERENTIABLE_EXCHANGE = False
try:
    with fwad.dual_level():
        without = fwad.unpack_dual(rhs(fwad.make_dual(Q, v))).tangent.clone()
finally:
    process_topology._DIFFERENTIABLE_EXCHANGE = True

dropped = global_norm(without - ad) / scale
report.append(f"  tangent WITHOUT halo exchange vs AD : {dropped:.3e}  (control: must be large)")

# Check the VJP over the complete distributed state.
Qg = Q.clone().requires_grad_(True)
(vjp,) = torch.autograd.grad(rhs(Qg), Qg, grad_outputs=w)

forward_side = global_dot(ad, w)
adjoint_side = global_dot(v, vjp)
adjoint_error = abs(forward_side - adjoint_side) / max(abs(forward_side), abs(adjoint_side))
report.append(f"  <J v, w>                            : {forward_side!r}")
report.append(f"  <v, J^T w>                          : {adjoint_side!r}")
report.append(f"  relative difference                 : {adjoint_error:.3e}")

failures = []
if best > 1.0e-8:
    failures.append(f"FD never approaches the AD tangent (best {best:.3e}); the halo may be dropping it")
if dropped < 1.0e-3:
    failures.append(f"disabling the halo tangent changed nothing ({dropped:.3e}); the control is not exercised")
if adjoint_error > 1.0e-12:
    failures.append(f"the adjoint does not match the tangent ({adjoint_error:.3e})")

if comm.rank == 0:
    print("\n".join(report), flush=True)
    print("FAIL: " + "; ".join(failures) if failures else "OK", flush=True)

sys.exit(1 if failures else 0)
