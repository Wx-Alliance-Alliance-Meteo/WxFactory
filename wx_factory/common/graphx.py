import matplotlib.pyplot
import numpy
from torch import Tensor


def image_field(
    geom,
    field: Tensor,
    filename: str,
    vmin: float,
    vmax: float,
    n: int,
    label: str = "K",
    colormap: str = "jet",
):
    domain_width = geom.x1 - geom.x0
    domain_height = geom.z1 - geom.z0
    aspect_ratio = domain_width / domain_height

    fig_width = 12
    fig_height = min(8, max(3.5, fig_width / aspect_ratio + 1))

    fig, ax = matplotlib.pyplot.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    cmap = ax.contourf(
        geom.X1_cartesian.cpu().numpy(),
        geom.X3_cartesian.cpu().numpy(),
        field.cpu().numpy(),
        cmap=colormap,
        levels=numpy.linspace(vmin, vmax, n),
        extend="both",
    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")

    # Saturate outside the scale without introducing a new color at either limit.
    levels = cmap.levels
    cmap.cmap = cmap.cmap.copy()
    cmap.cmap.set_under(cmap.cmap(cmap.norm((levels[0] + levels[1]) / 2)))
    cmap.cmap.set_over(cmap.cmap(cmap.norm((levels[-2] + levels[-1]) / 2)))
    cmap.changed()

    cbar = fig.colorbar(cmap, ax=ax, orientation="vertical")
    cbar.set_label(label)

    fig.savefig(filename, dpi=150)
    matplotlib.pyplot.close(fig)
