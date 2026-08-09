# Testing

WxFactory uses `unittest` with a small MPI-aware compatibility layer. Run commands from the
repository root.

## Test data

Reference states and configurations live in the `tests/data` submodule. Fetch it before running
anything, or the tests that read fixtures will fail with a missing-directory error:

```bash
git submodule update --init tests/data
```

Note that `tests/data/temp`, which some tests create for their scratch output, will block the
initial checkout if it is present before the submodule is cloned.

## Complete test suite

```bash
./tests/run_all_tests.sh
```

This runs the single-process unit tests, the 24-rank MPI unit tests, and the configured integration
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

- Name test modules `test_<subject>.py` and methods `test_<behavior>`.
- Prefer an existing runner and register new test cases in `tests/unit/run_tests.py` or
  `tests/unit/run_mpi_tests.py`.
- Add integration cases to `tests/integration/run_all_integration_tests.sh`.
- Test both CPU and CUDA when behavior is device-sensitive.
- Keep required test data under `tests/data/` and use paths relative to the repository root.
