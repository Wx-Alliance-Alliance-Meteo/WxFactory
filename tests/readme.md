# Testing

A growing set of tests is available that cover more and more of WxFactory's functionality. The testing infrastructure is under active development and the way tests are organized is likely to change as progress continues on the project.

## Manual execution

A script is provided to run most availables tests. From the project root directory, run
```
./tests/run_all_tests.sh
```

The tests are divided into 3 categories, which can be run individually:
### Single-process unit tests:

`./tests/unit/run_tests.py`

These tests include some that run on a GPU. GPU tests are skipped if we are unable to initialize the GPU portion of WxFactory.

### Multi-process unit tests

`mpirun -n [number of processes] ./tests/unit/run_mpi_tests.py`

They need at least 6 processes

### Integration tests

`./tests/integration/run_all_integration_tests.sh`

These tests run a certain configuration for several time steps and verify that the result corresponds to a reference.

Additionally, there is a set of tests that tries to run a variety of configurations simply to check that
they don't crash, without verifying the result: `./tests/integration/quick_test.sh`.
These are not included in the `run_all_tests.sh` script; they will eventually become part of the regular
integration tests.


## Test framework

We built a compatibility layer between MPI and Unittest for running tests that requires multiple processes. The layer include a test runner and a utility function to artificially reduce the number of processes for a given test (`tests/mpi_test.py`).

### How to use the runner

When you create a new entrypoint for a test (ex: `run_tests.py` or `run_mpi_test.py`), you need this snippet of code :
```python
from mpi_test import TestRunner

...

def main():
    # This is the runner. You just need to replace the original Unittest runner to convert a standard test to a MPI test
    runner = MpiRunner()
    runner.run("""Your tests go here""")
```

### How to convert a test

To convert a test to use the MPI layer and that need a defined number of processes, add the following code to a test case (either in the test or the setup) :

```python
from mpi_test import run_test_on_x_process

...

class Test(...):
...
    def test_...(self):
        comm = run_test_on_x_process(self, """Number of required processes here""")
        ...
```

The function either give you a new communicator to give to your function to test or skip the test for the extra processes. If you don't have enough processes to run the test, the test is skipped on every process. Don't forget to disconnect the communicator at the end of the test.

### Utility code

#### ndarray_generator

This file contains utility functions to generate matrixes and vectors per device. The return list has the same size as the device list. Each item return is mapped to its corresponding device (the array is on the device with the same index).

#### cpu_test

In the file, there is a test case class that you can inherit your test case from. This test case create a CPU_Device when setting up the test.

#### gpu_test

In the file, there is a test case class that you can inherit your test case from. This test case create a CPU_Device and a CUDA_Device when setting up the test. It also skip the test when no cuda device can be found.

## Test contribution

There are no conventions yet, but here are our recommendations :
* Each test file (file that contains tests) should start with `test_{name of the function}`
* Each test case should start with `test_{whatever you are testing}`
* If possible, use already created entrypoints
* To add an integration test, add a line in the `run_all_integration_tests.sh`
