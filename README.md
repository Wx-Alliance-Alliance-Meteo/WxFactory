# WxFactory
Research numerical weather model. The name is inspired by [Richardson’s Fantastic Forecast Factory](https://www.emetsoc.org/resources/rff/)

## Documentation list

- [Testing](./tests/readme.md)
- [Contributing](./doc/contribute.md)
- [References](./doc/references.md)

## Requirements

WxFactory was built for Python 3.11 (at least).  It also requires an MPI implementation.

### Python packages
* Python version at least 3.11
* `numpy` Scientific tools for Python
* `scipy` Python-based ecosystem of open-source software for mathematics, science, and engineering
* `sympy` Python library for symbolic mathematics
* `mpi4py` Python interface for MPI
* `pybind11` Library to expose C++/Python types to each other
* `netcdf4` Python/NumPy interface to the netCDF C library (MPI version)
* `matplotlib` A python plotting library, making publication quality plots

### Other libraries
* `netcdf4` Library to handle netCDF files. There is an MPI version of it, if you want parallel output
* `sqlite` To be able to store solver stats.

### Optional
* `cartopy`  A cartographic python library with matplotlib support for visualisation
* `tqdm`     Progress bar when generating matrices
* `snakeviz` A tool for visualizing profiling output

Python packages can be installed with the package management system of your
Linux distribution or with `pip`.

## Running WxFactory

```
# With the cubed sphere as a grid:
mpirun -n 6 ./WxFactory config/case6.ini

# With the 2D cartesian grid:
./WxFactory config/gaussian_bubble.ini
```

## Profiling WxFactory

You can generate an execution profile when running WxFactory by adding the `--profile` flag to the main command. For example:
```
mpirun -n 6 python3 ./WxFactory --profile config/case6.ini
```

This will generate a set of `profile_####.out` files, one for each launched process, that can be viewed with `snakeviz`. _You need to be able to open a browser window from the terminal to use this command_:
```
snakeviz ./profile_0000.out
```

## If you find this project useful, please cite:
Gaudreault, S., Charron, M., Dallerit, V., & Tokman, M. (2022). High-order numerical solutions to the shallow-water equations on the rotated cubed-sphere grid. Journal of Computational Physics, 449, 110792. [https://doi.org/10.1016/j.jcp.2021.110792](https://doi.org/10.1016/j.jcp.2021.110792)
