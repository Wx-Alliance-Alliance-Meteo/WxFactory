# How to contribute

The two branches that may be used by everyone are `main` and `dev`
 - `main`: Stable branch. All tests are always supposed to pass on this branch
 - `dev`: Active development branch. This one contains the most recent contributions, but may not be working in every case.

1. Create a new branch, starting from `main` or from `dev`, with a name that fits your project.
2. During your work, commit frequently in your branch.
3. Write tests to verify that your code works. Please include these tests in the commits (we have a directory just for that!)
4. Open a pull request from your branch to the branch from which you started.

## Configuration options

The available configuration options are described in a single place: the schema file
`config/config-format.json`. To keep `config.<option>` autocompletion and static type checking
working, `wx_factory/common/configuration.py` carries a block of generated type annotations.

If you add, remove, or change the type of an option in the schema, regenerate that block:

```
python -m wx_factory.common.config_hints --write
```

and commit the change. A unit test (and the `checks` CI workflow) runs
`python -m wx_factory.common.config_hints --check` and will fail if the annotations are stale.

Regenerate the human-readable option table from the same schema after changing it:

```
wxfactory config/case1.ini --config-options md > doc/config_options.md
```

The command requires a valid configuration path because it shares the normal command-line parser.

## Numerical precision

Initial grid coordinates and metric terms, Gauss–Legendre quadrature, DFR operators, and modal
filters are constructed in `float64`. In `precision = mixed` mode, completed arrays are cast once
to `float32`; the model state and most runtime arrays also use `float32`. Accuracy-sensitive solver
operations—including selected reductions, small bookkeeping matrices, and iterative-refinement
residuals—use `float64`. The `precision = double` mode retains `float64` throughout. Preserve the
double-build/working-precision boundary when adding static derived spatial operators: it avoids
baking construction roundoff into single-precision coefficients without increasing runtime
storage. State-dependent fluxes, work arrays, and metrics rebuilt after a time-dependent geometry
update normally remain in the configured working precision.

## Extending WxFactory

Several of the choices WxFactory makes at start-up (which time integrator, which right-hand side,
...) are driven by a *registry*: a table mapping a name (or a combination of options) to a factory
that builds the object. Adding a new option means adding one factory and one registry entry — you
do not need to edit a chain of `if`/`elif` tests scattered across the code.

### Add a new time integrator

There are two registries, because there are two kinds of integrator:

* **regular** schemes advance one right-hand side, `rhs.full`. They go in `REGISTRY` and run on
  every configuration.
* **partitioned** schemes advance `rhs.explicit` and `rhs.implicit` separately — `imex2`,
  `partrosexp2`, `strang_epi2_ros2`, `strang_ros2_epi2`. They go in `PARTITIONED_REGISTRY`, and
  they only run where that partition exists.

Steps:

1. Add your integrator class in `wx_factory/integrators/`, subclassing `Integrator`. Accept a
   `context` keyword and pass it to `super().__init__`, so the scheme runs on the device the
   simulation was configured for.
2. In that file, add the scheme to whichever registry fits. A regular scheme:
   ```python
   REGISTRY = {"my_scheme": lambda cfg, rhs, prec, ctx: MyScheme(cfg, rhs.full, context=ctx)}
   ```
   A partitioned one:
   ```python
   REGISTRY: dict = {}

   PARTITIONED_REGISTRY = {
       "my_imex": lambda cfg, rhs, prec, ctx: MyImex(cfg, rhs.explicit, rhs.implicit, context=ctx),
   }
   ```
   A module may define both. A name may not appear in both registries; `integrators/__init__.py`
   raises at import time if it does.
3. Import the module in `wx_factory/integrators/__init__.py` and add it to the list of modules whose
   registries are merged.
4. If the scheme needs its own configuration options, add them to `config/config-format.json` and
   regenerate the type hints (see above). Gate them on the integrator name with a `dependency`
   block, as `os22_parameter` does, so they are only read — and only required — when that scheme is
   selected. `time_integrator` itself has no `selectables` list: the registries are the authority,
   and an unknown name produces an error listing what is available.

`resolve` refuses a partitioned scheme when the configuration provides no partition (shallow water,
2D advection, and Euler with `advection_only = on`), and reports the reason the RHS gave. Generic
splittings (`lie`, `strang`, `os22`) resolve their sub-integrators through the same function, so a
partitioned sub-integrator is rejected there too.

### Add a new exponential solver

Exponential integrators select their inner solver through the registry in
`wx_factory/solvers/exponential_solver.py`. The `exponential_solver` configuration option is
validated by this registry, so a new solver does not require changes to the integrators or the
configuration schema.

1. Write an adapter that accepts an `ExponentialSolverRequest` and returns an
   `ExponentialSolverResult`.
2. Register the adapter under its configuration name:
   ```python
   @register_exponential_solver("my_solver")
   def _solve_my_solver(request: ExponentialSolverRequest) -> ExponentialSolverResult:
       value, stats = my_solver(
           request.tau_out,
           request.operator,
           request.vectors,
           tol=request.tolerance,
           device=request.device,
       )
       return ExponentialSolverResult(
           value=value,
           raw_stats=stats,
           iterations=stats.iterations,
           rejected_steps=stats.rejected_steps,
           local_error=stats.local_error,
           final_krylov_size=stats.final_krylov_size,
       )
   ```
3. Translate solver-specific arguments and statistics inside the adapter. Integrators should only
   construct the common request and consume the normalized result.
4. Validate solver-specific limitations in the adapter. For example, an implementation that
   supports only one output time should reject a request containing several times with a clear
   error.
5. Add registry, argument-translation, statistics, and capability tests under
   `tests/unit/solvers/`.

### Add a new equation set / right-hand side

The RHS depends on a pair: the equation set and the geometry it runs on. These are registered in
`wx_factory/rhs/rhs_selector.py`.

1. Add your `PDE` and `RHS` classes.
2. Register a factory for the `(equations, geometry class)` combination:
   ```python
   @register_rhs("my_equations", CubedSphere3D)
   def _build(ctx: RhsContext) -> RhsBundle:
       pde = MyPDE(ctx.geom, ctx.param, ctx.metric)
       full = MyRhs(pde, ctx.geom, ctx.operators_real, ...)
       return RhsBundle(
           full=full,
           shape=ctx.fields_shape,
           partition_reason="the partitioned right-hand side is not implemented for my equations",
       )
   ```
   `ctx` (an `RhsContext`) carries everything a factory might need: geometry, operators, metric,
   topography, process topology, config, and the state-vector shape.

   If the equations do support a partitioned integrator, pass `explicit=` and `implicit=` instead
   (both, or neither — `RhsBundle` rejects half a partition) and drop `partition_reason`. Otherwise
   write one sentence saying why there is no partition: it is what the user sees when they select a
   partitioned integrator, so make it say what would have to change. The same bundle may answer
   differently depending on its configuration — the Euler factories provide the partition normally
   but withhold it under `advection_only`, which freezes the dynamics.
3. Add `"my_equations"` to the `equations` option's `selectables` in `config/config-format.json`
   and regenerate the config type hints (see above).

### Add a new grid (geometry)

The geometry depends on a pair: the grid type and the equation set. These are registered in
`wx_factory/geometry/registry.py`.

1. Add your `Geometry` subclass.
2. Register a factory for the `(grid_type, equations)` combination:
   ```python
   @register_geometry("my_grid", "euler")
   def _build(ctx: GeometryContext) -> Geometry:
       return MyGrid(ctx.num_elements_horizontal, ctx.num_solpts, ..., ctx.device)
   ```
   `ctx` (a `GeometryContext`) carries the config, device, MPI communicator, and the derived grid
   sizes. If your grid needs a process topology (like the cubed sphere), create it in the factory;
   `Simulation` retrieves it back from `geometry.process_topology`.
3. Add `"my_grid"` to the `grid_type` option's `selectables` in `config/config-format.json` and
   regenerate the config type hints.

### Add a new output format

Output managers are selected by the geometry's `output_family` and the `output_format` config
option, and are registered in `wx_factory/output/registry.py`.

1. Add your `OutputManager` subclass.
2. Register a factory for the `(output_family, output_format)` combination:
   ```python
   @register_output("cubesphere", "my_format")
   def _build(ctx: OutputContext) -> OutputManager:
       return MyOutput(ctx.config, ctx.geometry, ctx.operators, ctx.device, ...)
   ```
   Register with `output_format=None` if the output does not depend on the format (as the
   Cartesian output does). A geometry advertises its family through the `output_family` class
   attribute.
3. Add `"my_format"` to the `output_format` option's `selectables` in `config/config-format.json`
   and regenerate the config type hints.

### Add a new step hook

A step hook post-processes the state after every time step. Hooks are registered in
`wx_factory/step_hooks/registry.py`. Unlike the other choices, several hooks may apply at once, so
each registers a *provider* that returns a hook instance when it applies or `None` otherwise.

1. Add your `StepHook` subclass (implement `process(Q, t)`).
2. Register a provider, choosing the phase it is resolved in — `PHASE_GEOMETRY` (only the geometry
   and config are available; runs before the initial state is built) or `PHASE_STATE` (the metric
   and operators are available):
   ```python
   @register_step_hook("my_hook", phase=PHASE_STATE)
   def _provider(ctx: StepHookContext):
       if ctx.config.case_number == 42:
           return MyHook(ctx.geometry, ctx.metric, ctx.operators, ctx.config)
       return None
   ```

### Add a new preconditioner

Preconditioners are selected by the `preconditioner` config option and registered in
`wx_factory/precondition/`. There are currently no built-in preconditioners (the historical ones
were removed because they were broken); `preconditioner = none` means no preconditioning.

1. Add a class that subclasses `Preconditioner` (from `wx_factory/precondition/preconditioner.py`).
   Implement `__apply__(vec, x0, verbose)` (how it acts on a vector), and override `prepare(dt, Q)`
   if it needs to refresh internal state at the start of each time step.
2. Register a factory for its config name:
   ```python
   @register_preconditioner("my_precond")
   def _build(ctx: PreconditionerContext) -> Preconditioner:
       return MyPreconditioner(ctx.rhs, ctx.geometry, ctx.config, ...)
   ```
   `ctx` (a `PreconditionerContext`) carries the config, device, geometry, operators, the RHS
   bundle, metric, topography, process topology, and the state-vector shape.
3. Add `"my_precond"` to the `preconditioner` option's `selectables` in `config/config-format.json`
   and regenerate the config type hints. Linear-solver-based integrators will then receive it
   through `self.preconditioner` and call `prepare` / apply it automatically.
