# WxFactory

A research numerical weather prediction framework inspired by [Richardson's Fantastic Forecast Factory](https://www.emetsoc.org/resources/rff/).

## Documentation

Full documentation is [available here](http://hpfx.collab.science.gc.ca/~sdyn001/WxFactory).

- [Configuration options](doc/config_options.md)
- [Testing](./tests/readme.md)
- [Contributing](./doc/contribute.md)
- [References](./doc/references.md)

## Requirements

Python 3.11 or later and an MPI implementation are required.

Necessary Python packages can be installed using:

```
pip install -r requirements.txt
```

For GPU support, also install the CuPy variant matching your CUDA version (`cupy-cuda11x` or `cupy-cuda12x`).

## Installation

```
pip install -e .
```

## Running WxFactory

```
# Cubed-sphere grid (requires a multiple of 6 processes):
mpirun -n 6 wxfactory config/case6.ini

# 2D Cartesian grid:
wxfactory config/gaussian_bubble.ini
```

## Profiling

Add `--profile` to generate per-process profile files (`prof_0000.out`, etc.), then view them with `snakeviz`:

```
mpirun -n 6 wxfactory --profile config/case6.ini
snakeviz prof_0000.out
```

## Citation

If you find this project useful, please cite:

Gaudreault, S., Charron, M., Dallerit, V., & Tokman, M. (2022). High-order numerical solutions to the shallow-water equations on the rotated cubed-sphere grid. *Journal of Computational Physics*, 449, 110792. https://doi.org/10.1016/j.jcp.2021.110792

Gaudreault, S., Subich, C., Panday, S., Charron, M., Magnoux, V., Dallerit, V., & Tokman, M. (2025). Application of High-Order Direct Flux Reconstruction and Stiffness-Resilient Time Integration to Simulations of Idealized Atmospheric Flows. *International Journal for Numerical Methods in Fluids*. https://doi.org/10.1002/fld.70046
