# Documentation map

The maintained user and contributor documentation is:

- [`../README.md`](../README.md): installation and execution
- [`config_options.md`](config_options.md): generated configuration schema
- [`contribute.md`](contribute.md): extension points and development workflow
- [`references.md`](references.md): scientific references
- [`../tests/readme.md`](../tests/readme.md): test suites and test framework

The LaTeX files in this directory are mathematical derivations, presentations, and historical
design notes. `cubed-sphere_summary.tex` and `flux_jacobian.tex` are mathematical references, not
line-by-line specifications of the implementation. `bulle.tex`, `gef.tex`, and `cubesphere.tex`
describe earlier solver/preconditioner experiments and are retained as archival material; they are
not descriptions of the current implementation. The diagrams under `img/` accompany those notes
and presentations and should likewise not be used as an authoritative description of current
tensor storage.

The Sphinx project under `sphinx/` assembles the maintained Markdown pages and generated API
documentation.
