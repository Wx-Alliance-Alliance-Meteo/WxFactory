# Manual execution

To manually run the tests, go to the root folder of the project. Then run :

* `python ./tests/run_tests.py` to run the non MPI tests. These tests contains cuda dependant test cases. However, you can still run most of the cases without a GPU. 
* `mpirun -n {number_of_process} python ./tests/run_mpi_tests.py` to run the MPI tests with `number_of_process` processes. We recommend a least 6.
* `./tests/quick_test.sh` to run, ironically, a test on every possible configuration of the simulator.
* `./tests/run_all_integration_tests.sh` to run all integration tests

# Test framework

We built a compatibility layer between MPI and Unittest for running tests that requires multiple processes. The layer include a test runner and a utility function to artificially reduce the number of processes for a given test (`tests/mpi_test.py`).

## How to use the runner

When you create a new entrypoint for a test (ex: `run_tests.py` or `run_mpi_test.py`), you need this snippet of code :
```python
from mpi_test import TestRunner

...

def main():
    # This is the runner. You just need to replace the original Unittest runner to convert a standard test to a MPI test
    runner = MpiRunner()
    runner.run("""Your tests go here""")
```

## How to convert a test

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

## Utility code

### ndarray_generator

This file contains utility functions to generate matrixes and vectors per device. The return list has the same size as the device list. Each item return is mapped to its corresponding device (the array is on the device with the same index).

### cpu_test

In the file, there is a test case class that you can inherit your test case from. This test case create a CPU_Device when setting up the test.

### gpu_test

In the file, there is a test case class that you can inherit your test case from. This test case create a CPU_Device and a CUDA_Device when setting up the test. It also skip the test when no cuda device can be found.

# Test contribution

There are no convention yet, but here are our recommendations :
* Each test file (file that contains tests) should start with `test_{name of the function}`
* Each test case should start with `test_{whatever you are testing}`
* If possible, use already created entrypoints
* To add an integration test, add a line in the `run_all_integration_tests.sh`
