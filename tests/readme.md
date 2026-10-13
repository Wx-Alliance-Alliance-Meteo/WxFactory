# Testing

WxFactory uses `unittest` with a small MPI-aware compatibility layer. Run commands from the
repository root.

## Test data

The single-process numerical tests use local configurations and generated fields. Historical
integration reference states live in the `tests/data` submodule. Fetch it before running those
comparisons:

```bash
git submodule update --init tests/data
```

Note that `tests/data/temp`, which some tests create for their scratch output, will block the
initial checkout if it is present before the submodule is cloned.

## Complete test suite

```bash
./tests/run_all_tests.sh
```

This runs the single-process unit tests, the six-rank MPI unit tests, and the configured integration
tests. The MPI tests require an MPI installation and enough ranks for the selected cases.

## Individual suites

Single-process unit tests:

```bash
./tests/unit/run_tests.py
```

MPI unit tests (six ranks are sufficient for the required cases; additional 24- and 54-rank cases
are enabled when enough ranks are supplied):

```bash
mpirun -n 6 ./tests/unit/run_mpi_tests.py
```

Integration tests:

```bash
./tests/integration/run_all_integration_tests.sh
```

Both unit-test entry points accept an optional regular expression to select tests and
`--no-buffer` to show captured output.

## Test framework

`WxTestCase` in `tests/unit/wx_test.py` creates the requested PyTorch device for a test. CPU is the
default. Tests instantiated with `device_name="cuda"` are skipped when CUDA is unavailable.

`MpiTestCase` and `run_test_on_x_process` in `tests/unit/mpi_test.py` restrict a test to a requested
number of ranks. Optional tests are skipped when too few ranks are available. The returned
communicator is stored as `MpiTestCase.comm`; callers that create a communicator directly are
responsible for disconnecting it.

The suite runners combine results from all participating ranks so an error or failure on any rank
fails the test.

## Adding tests

- Start with a concrete failure the test must detect. Prefer an analytic solution, conservation
  law, independent calculation, or a regression for an observed defect.
- Do not test attribute storage, enumerate registry contents, reproduce a production method
  line by line, or check only that an output is finite when an accuracy check is possible.
- Use subtests for variations of one numerical question. Keep separate tests for distinct
  failure modes; reducing the test count alone is not a measure of quality.
- When practical, deliberately introduce the suspected error temporarily and verify that the
  test fails. Keep these experiments outside the maintained suite.
- Name test modules `test_<subject>.py` and methods `test_<behavior>`.
- Prefer an existing runner and register new test cases in `tests/unit/run_tests.py` or
  `tests/unit/run_mpi_tests.py`.
- Add integration cases to `tests/integration/run_all_integration_tests.sh`.
- Test both CPU and CUDA when behavior is device-sensitive.
- Keep required test data under `tests/data/` and use paths relative to the repository root.

## Euler regression coverage

The single-process suite registers metric, interface, operator, tensor contraction,
and analytic Jacobian tests. `rhs/test_euler_operator.py` uses a frozen periodic
5D chart to check scalar transport, acoustic and shear modes, polynomial momentum,
and stability. The shear refinement test assembles the production analytic Jacobian
and compares its exact semidiscrete evolution with a translated sine wave.
Its shared fixture is `rhs/euler_frozen_fixture.py`; it is test
support, not a separate experiment driver.
`jacobian/test_euler_partitions.py` checks the DFR partition identity, analytic
partition derivatives, and column solves with horizontal-velocity coupling.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
python3 tests/unit/run_tests.py 'Euler|Metric5D|SpatialContraction'
```

The MPI suite additionally checks panel seams, terrain reference scaling,
conservation, deep/shallow partition derivatives and column solves, and an EPI2/PMEX step:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
mpirun -n 6 python3 tests/unit/run_mpi_tests.py 'EulerRhs|Metric5DSeam'
```

For GPU runs, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Keep regression tests when they protect an adopted numerical method, even if
originally written while diagnosing a failure. Temporary optimization comparisons
and scripts that monkeypatch production methods do not belong in the maintained
suite.

KIOPS and PMEX are checked against an independent SciPy augmented matrix exponential,
including a singular, nonnormal matrix and multiple phi terms. FGMRES is checked against
a dense direct solve and its actual residual, including restarted iterations.

Integration reference comparisons use global L2 errors for each equation separately.
Every equation must satisfy the tolerance; a zero reference uses absolute L2 error.
Non-finite values on any rank fail the comparison collectively. Historical reference
states must be compatible with the current metric and state representation.
