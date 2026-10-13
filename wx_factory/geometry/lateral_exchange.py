"""Single-tile Cartesian boundary exchanges use one shared coordinate frame.

Wall and periodic traces resolve locally; cubed-sphere exchanges use ProcessTopology.
"""

from ..common.definitions import idx_v1, idx_v2


class _ImmediateRequest:
    """Match the MPI request interface for an exchange requiring no communication."""

    def __init__(self, ghosts):
        self._ghosts = ghosts

    def wait(self, timeout=10.0):
        return self._ghosts


class SingleTileLateralExchange:
    """Wall and periodic traces for a single Cartesian tile, without MPI."""

    def __init__(self, mode: str):
        if mode not in ("wall", "periodic"):
            raise ValueError(f"SingleTileLateralExchange mode must be 'wall' or 'periodic', got {mode!r}")
        self.mode = mode

    def start_exchange_euler_3d(
        self, south, north, west, east, boundary_sn=None, boundary_we=None, flip_dim=None, covariant=False
    ):
        if self.mode == "periodic":
            ghosts = (north, south, east, west)
        else:
            ghosts = (
                self._reflect(south, idx_v2),
                self._reflect(north, idx_v2),
                self._reflect(west, idx_v1),
                self._reflect(east, idx_v1),
            )
        return _ImmediateRequest(ghosts)

    @staticmethod
    def _reflect(trace, normal_idx):
        """Negate the wall-normal velocity; Cartesian faces have no tangential metric coupling."""
        out = trace.copy()
        out[normal_idx] = -out[normal_idx]
        return out


class FlatTileTopology:
    """Adapt the cubed-sphere layout and I/O interfaces to a single Cartesian rank."""

    def __init__(self, context, lateral_boundary: str):
        self.context = context
        self._comm = context.comm
        self.size = context.comm.size
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

    def distribute_cube(self, field, *args, **kwargs):
        return field

    def gather_cube(self, field, *args, **kwargs):
        return field

    def gather_tiles_to_panel(self, field, *args, **kwargs):
        return field
