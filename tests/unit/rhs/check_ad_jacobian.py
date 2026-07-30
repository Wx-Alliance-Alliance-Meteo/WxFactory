"""Check the ``jacobian_method = ad`` Jacobian action against the other two methods.

    python tests/unit/rhs/check_ad_jacobian.py

Single rank, no environment variable: the point of the check is that the configuration alone turns
differentiable mode on, before the device installs its inference-mode guard.

This is a standalone script rather than a discovered test because differentiable mode is
process-wide and cannot be switched off once enabled -- it would put every later test in the same
process onto the AD code paths (fresh RHS scratch arrays, the matmul ``out=`` fallback).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", ".."))

import torch

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.device import differentiable_mode
from wx_factory.simulation import Simulation
from wx_factory.solvers.matvec import matvec_fun

here = os.path.dirname(os.path.realpath(__file__))
config_text = readfile(os.path.join(here, "vertical_jacobian_config.ini"))
config_text = config_text.replace("jacobian_method = fd", "jacobian_method = ad")
config = Configuration(config_text, load_default_schema())

if differentiable_mode():
    print("differentiable mode was already on; this check cannot prove the config enabled it")
    sys.exit(1)

sim = Simulation(config, quiet=True)

rhs = sim.rhs.full
Q = sim.initial_state.Q.clone()
gen = torch.Generator().manual_seed(4321)
v = (torch.rand(Q.shape, generator=gen, dtype=Q.dtype) - 0.5) * Q.abs()

dt = 1.0
base = rhs(Q)
ad = matvec_fun(v, dt, Q, base, rhs, "ad")
complex_step = matvec_fun(v, dt, Q, base, rhs, "complex")
fd = matvec_fun(v, dt, Q, base, rhs, "fd")

# Central differences are second-order accurate, so they bound how closely any exact method can be
# confirmed here; the complex step sits the same distance away.
eps = 1.0e-6
central = (dt * (rhs(Q + eps * v) - rhs(Q - eps * v)) / (2.0 * eps)).flatten()


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


report = [
    f"differentiable mode enabled from the config : {differentiable_mode()}",
    f"device allows autograd                      : {sim.device.allows_autograd}",
    f"||J.v||                                     : {torch.linalg.norm(ad).item():.6e}",
    f"  ad      vs complex step : {relative(ad, complex_step):.3e}",
    f"  ad      vs central FD   : {relative(ad, central):.3e}",
    f"  complex vs central FD   : {relative(complex_step, central):.3e}",
    f"  fd      vs central FD   : {relative(fd, central):.3e}",
]

failures = []
if not differentiable_mode():
    failures.append("the configuration did not enable differentiable mode")
if relative(ad, complex_step) > 1.0e-12:
    failures.append(f"AD disagrees with the complex step ({relative(ad, complex_step):.3e})")
if relative(ad, central) > 1.0e-5:
    failures.append(f"AD disagrees with central differences ({relative(ad, central):.3e})")

print("\n".join(report), flush=True)
print("FAIL: " + "; ".join(failures) if failures else "OK", flush=True)

sys.exit(1 if failures else 0)
