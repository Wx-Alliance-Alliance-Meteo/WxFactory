# WxFactory

A research numerical weather prediction framework inspired by [Richardson's Fantastic Forecast Factory](https://www.emetsoc.org/resources/rff/).

## Documentation

The source documentation is maintained in this repository:

- [Configuration options](doc/config_options.md)
- [Documentation map](doc/README.md)
- [Testing](tests/readme.md)
- [Contributing](doc/contribute.md)
- [References](doc/references.md)

## Requirements

Python 3.11 or later and an MPI implementation are required. WxFactory uses PyTorch for both CPU
and GPU execution.

Necessary Python packages can be installed using:

```
pip install -r requirements.txt
```

For GPU execution, install a CUDA-enabled PyTorch build compatible with the system CUDA driver.
The standard PyTorch installation instructions select the appropriate wheel.

## Installation

```
pip install -e .
```

## Running WxFactory

The `wxfactory` command below is created by `pip install -e .` (see [Installation](#installation)).
If you have not installed the package and want to run straight from the source tree, use the
`./WxFactory` launcher script instead (e.g. `mpirun -n 6 ./WxFactory config/case6.ini`).

```
# Cubed-sphere grid (requires a multiple of 6 processes):
mpirun -n 6 wxfactory config/case6.ini

# 3D Cartesian grid:
wxfactory config/gaussian_bubble.ini
```

For example, to run DCMIP 2-1 on the GPU with six MPI processes:

```bash
mpirun -n 6 wxfactory config/dcmip21_EPI2.ini
```

For DCMIP 2-2, the sheared mountain-wave case, use `config/dcmip22_EPI2.ini`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True mpirun -n 6 wxfactory config/dcmip22_EPI2.ini
python scripts/plot_dcmip22.py results/dcmip22_epi2.nc -o results/dcmip22 --pdf
```

The configuration uses a non-rotating planet of radius 6371220/500 m, a 30 km lid,
and 60 solution points per coordinate direction (20 elements with 3 points each).
It runs for 7200 s with output every 100 s. The node spacing follows Gauss–Legendre
quadrature and the SLEVE mapping; it is not the uniform 500 m level spacing of the
finite-volume reference grid in the DCMIP document.

The initial temperature varies with latitude, and the equatorial zonal wind increases
from 20 m/s at sea level to 80 m/s at the lid. The sponge above 20 km relaxes to the
same sheared wind. These are the dry shallow-atmosphere profiles in DCMIP-2012,
Eqs. (80)–(85), also implemented in GEM's `test2_schaer_mountain` reference routine.

The plotting script writes temperature perturbation, physical vertical velocity, and
zonal-wind perturbation sections on native and constant-height levels at 2400, 3600,
and 7200 s, plus an equatorial JSON summary of minima, maxima and unweighted RMS.
It subtracts the latitude-dependent background before equatorial interpolation.
Use `--times 0 100` to inspect initialization and early output. The shared
`plot_dcmip21.py` also detects case 22 from NetCDF metadata.

## Profiling

Add `--profile` to generate per-process profile files (`prof_0000.out`, etc.), then view them with `snakeviz`:

```
mpirun -n 6 wxfactory --profile config/case6.ini
snakeviz prof_0000.out
```

Run `wxfactory config/case6.ini --config-options` to print the accepted configuration schema, or
append `md` to emit the table in Markdown format.

## Citation

If you find this project useful, please cite:

Gaudreault, S., Charron, M., Dallerit, V., & Tokman, M. (2022). High-order numerical solutions to the shallow-water equations on the rotated cubed-sphere grid. *Journal of Computational Physics*, 449, 110792. https://doi.org/10.1016/j.jcp.2021.110792

Gaudreault, S., Subich, C., Panday, S., Charron, M., Magnoux, V., Dallerit, V., & Tokman, M. (2026). Application of High-Order Direct Flux Reconstruction and Stiffness-Resilient Time Integration to Simulations of Idealized Atmospheric Flows. *International Journal for Numerical Methods in Fluids*, 98(4), 448–468. https://doi.org/10.1002/fld.70046
