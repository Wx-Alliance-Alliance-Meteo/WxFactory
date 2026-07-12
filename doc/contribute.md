# How to contribute

The two branches that may be used by everyone are `main` and `dev`
 - `main`: Stable branch. All tests are always supposed to pass on this branch
 - `dev`: Active development branch. This one contains the most recent contributions, but may not be working in every case.

1. Create a new branch, starting from `main` or from `dev`, with a name that fits your project.
2. During your work, commit frequently in your branch.
3. Write tests to verify that your code works. Please include these tests in the commits (we have a directory just for that!)
4. Open a merge request from your branch to a the one from which you started.

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

## Extending WxFactory

Several of the choices WxFactory makes at start-up (which time integrator, which right-hand side,
...) are driven by a *registry*: a table mapping a name (or a combination of options) to a factory
that builds the object. Adding a new option means adding one factory and one registry entry — you
do not need to edit a chain of `if`/`elif` tests scattered across the code.

### Add a new time integrator

1. Add your integrator class in `wx_factory/integrators/`, subclassing `Integrator`.
2. In that file, add a `REGISTRY` dict mapping the config name to a factory:
   ```python
   REGISTRY = {"my_scheme": lambda cfg, rhs, prec, dev: MyScheme(cfg, rhs.full, device=dev)}
   ```
   Partitioned schemes use `rhs.explicit` / `rhs.implicit` instead of `rhs.full`.
3. Import the module in `wx_factory/integrators/__init__.py` so its `REGISTRY` is merged.
4. Add the name to the `time_integrator` option's `selectables` in `config/config-format.json`.

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
       return RhsBundle(full=full, shape=ctx.fields_shape)
   ```
   `ctx` (an `RhsContext`) carries everything a factory might need: geometry, operators, metric,
   topography, process topology, config, and the state-vector shape. If the equations support a
   partitioned integrator, also pass `explicit=` / `implicit=` to `RhsBundle`.
3. Add `"my_equations"` to the `equations` option's `selectables` in `config/config-format.json`
   and regenerate the config type hints (see above).

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
