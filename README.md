# WxFactory
WxFactory is a numerical model that solves the Euler equations on a cubed-sphere grid.
It is used to test large-scale modelling of the atmosphere on multiple GPUs.
Coordination between GPUs is managed using MPI. The GPUs themselves are mostly used through the CuPy python module,
with performance critical parts of the code written in CUDA.

## Requirements

* Python >= 3.11
* MPI (CUDA-aware)

### To take advantage of compiled code
* CUDA toolkit
    * CUDA runtime
    * `nvcc` compiler
* A C++ compiler

### Python packages
* `numpy`       Scientific tools for Python
* `scipy`       Python-based ecosystem of open-source software for mathematics, science, and engineering
* `sympy`       Python library for symbolic mathematics
* `mpi4py`      Python interface for MPI
* `pybind11`    Library to expose C++/Python types to each other
* `netcdf4`     Python/NumPy interface to the netCDF C library (MPI version)
* `matplotlib`  A python plotting library, making publication quality plots
* `setuptools`  To compile C++/CUDA portions of WxFactory
* `cupy`        To be able to run on GPU (can install `cupy-cuda11x` or `cupy-cuda12x` for precompiled module)

#### For validation
* `requests` To be able to download reference results
* `tqdm` Viewing download progress when validating results

#### Optional
* `cartopy`  A cartographic python library with matplotlib support for visualisation
* `snakeviz` A tool for visualizing profiling output
* `netcdf4` Library to handle netCDF files. There is an MPI version of it, if you want parallel output
* `sqlite` To be able to store solver stats.
* `Sphinx`      Library to build the documentation
* `myst-parser` Library to parse markdown files for documentation

## Running the benchmark

- Edit `tests/benchmark_gen8/config_8th_deg.ini` to specify a valid output directory (the `output_dir` option)
- Run 
    ```
    mpirun -n [##] ./WxFactory ./tests/benchmark_gen8/config_8th_deg.ini
    ```
    where `[##]` is the number of MPI processes that will be used. While `WxFactory` can run with more processes than
    there are GPUs available, performance is usually better when running exactly one process per GPU.

- `WxFactory` can only be run with specific numbers of processes. To know what numbers are possible, you can run
    ```
    ./WxFactory --allowed-proc-count ./tests/benchmark_gen8/config_8th_deg.ini
    ```

### Validating the results

To verify that the results of the run are correct, run
```
./scripts/validate_benchmark.py [directory where results are stored]
```

This will download the reference solution and compare the current results with them. The `--max-concurrent` option of
the validation script can be used to speed up the comparison, but running all of them simultaneously may require 150+ GB
of RAM.


# More documentation

[Available here](README_detailed.md)
