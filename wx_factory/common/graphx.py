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

    # Base height in inches, width scaled by aspect ratio
    fig_height = 10
    fig_width = fig_height * aspect_ratio

    fig, ax = matplotlib.pyplot.subplots(figsize=(fig_width, fig_height))

    cmap = matplotlib.pyplot.contourf(
        geom.X1_cartesian.cpu().numpy(),
        geom.X3_cartesian.cpu().numpy(),
        field.cpu().numpy(),
        cmap=colormap,
        levels=numpy.linspace(vmin, vmax, n),
        extend="both",
    )
    ax.set_aspect("equal", "box")

    cbar = fig.colorbar(cmap, ax=ax, orientation="vertical")
    cbar.set_label(label)

    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close(fig)
