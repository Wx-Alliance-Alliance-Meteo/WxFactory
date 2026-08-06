"""Pluggable lateral (horizontal) boundary treatment for the 3D Euler halo exchange.

The RHS and the analytic J2 exchange horizontal boundary traces through a single call,
``start_exchange_euler_3d(south, north, west, east, boundary_sn, boundary_we, flip_dim) -> request``,
whose ``request.wait()`` returns the four ghost traces ``(dq_s, dq_n, dq_w, dq_e)``. What "the
neighbour's trace" is depends only on the lateral boundary condition:

- **donor_cell** — the neighbouring tile's trace, with the cubed sphere's per-edge contravariant
  rotation. This is exactly :class:`~wx_factory.process_topology.ProcessTopology`, left untouched so
  the cubed-sphere path stays bit-identical; it is the provider for the cube.
- **wall** — a solid reflecting wall: the tile's own boundary trace with the wall-normal momentum
  negated (tangential momenta and scalars unchanged). No MPI.
- **periodic** — wrap the tile to its opposite edge (south halo <- north boundary, etc.), identity
  vector transform. No MPI (single tile).

``wall`` and ``periodic`` are single-tile, local (no communication), so their request resolves
immediately. They are selected for cartesian grids by the ``lateral_boundary`` config option; the
cubed sphere always uses ``donor_cell``.

NOTE: the exact half/orientation of the reflected/wrapped trace against the DFR halo reconstruction
is validated when a flat case is run end to end (unification phase 4); the mapping here is the
geometrically intended one.
"""

from ..common.definitions import idx_rho_u1, idx_rho_u2


class _ImmediateRequest:
    """A request whose result is already known (local exchange, no MPI). Mirrors ExchangeRequest."""

    def __init__(self, ghosts):
        self._ghosts = ghosts

    def wait(self, timeout=10.0):
        return self._ghosts


class SingleTileLateralExchange:
    """Local lateral boundary treatment for a single cartesian tile: ``wall`` or ``periodic``.

    Implements the same ``start_exchange_euler_3d`` contract as ProcessTopology, but computes the
    ghost traces locally from the tile's own boundary data -- no neighbour, no MPI.
    """

    def __init__(self, mode: str):
        if mode not in ("wall", "periodic"):
            raise ValueError(f"SingleTileLateralExchange mode must be 'wall' or 'periodic', got {mode!r}")
        self.mode = mode

    def start_exchange_euler_3d(self, south, north, west, east, boundary_sn=None, boundary_we=None, flip_dim=None):
        if self.mode == "periodic":
            # Wrap: the south halo sees the north boundary, west sees east, and vice versa. The two
            # flat tiles share one coordinate frame, so the contravariant momenta pass unchanged.
            ghosts = (north, south, east, west)
        else:  # wall
            # Reflect the tile's own boundary with the wall-normal momentum negated. South/north are
            # x2 faces (normal = rho_u2); west/east are x1 faces (normal = rho_u1).
            ghosts = (
                self._reflect(south, idx_rho_u2),
                self._reflect(north, idx_rho_u2),
                self._reflect(west, idx_rho_u1),
                self._reflect(east, idx_rho_u1),
            )
        return _ImmediateRequest(ghosts)

    @staticmethod
    def _reflect(trace, normal_idx):
        """Copy a boundary trace with its wall-normal momentum component negated."""
        out = trace.copy()
        out[normal_idx] = -out[normal_idx]
        return out


class FlatTileTopology:
    """A single-tile 'process topology' for a flat cartesian slab (one MPI rank, no decomposition).

    Provides the subset of the ProcessTopology surface the 3D geometry construction, the metric build,
    and the RHS exchange use: a trivial single-tile decomposition (one panel, one row/column) and the
    local lateral boundary exchange (wall or periodic). This is what lets Cartesian3D run at np=1
    without the cubed sphere's 6-panel MPI topology. Multi-rank cartesian decomposition is future work.
    """

    def __init__(self, context, lateral_boundary: str):
        self.context = context
        self._comm = context.comm
        self.size = context.comm.size
        # Trivial single-tile decomposition: the whole horizontal domain is this one tile.
        self.num_lines_per_panel = 1
        self.num_pe_per_panel = 1
        self.my_panel = 0
        self.my_row = 0
        self.my_col = 0
        self.my_rank_in_panel = 0
        self.panel_comm = context.comm
        self._exchange = SingleTileLateralExchange(lateral_boundary)

    def start_exchange_euler_3d(self, *args, **kwargs):
        return self._exchange.start_exchange_euler_3d(*args, **kwargs)

    def start_exchange_vectors(
        self, south, north, west, east, boundary_sn=None, boundary_we=None, flip_dim=None, covariant=False
    ):
        # The only caller at np=1 is the metric-derivative fix-up, on a flat tile where those
        # derivatives are spatially constant, so a periodic wrap (identity across the tile) is exact
        # regardless of the physical boundary mode. Vectors pass unchanged (one flat coordinate frame).
        return _ImmediateRequest((north, south, east, west))

    # --- single-tile I/O: no gather/scatter needed (one rank holds the whole field) ---
    def distribute_cube(self, field, *args, **kwargs):
        return field

    def gather_cube(self, field, *args, **kwargs):
        return field

    def gather_tiles_to_panel(self, field, *args, **kwargs):
        return field
