from mpi4py import MPI
import numpy
import pickle
import matplotlib.pyplot

plot_index = 0


def plot_field_from_file(geom_prefix, field_prefix):
    rank = MPI.COMM_WORLD.Get_rank()
    suffix = "{:04d}.dat".format(rank)
    geom_filename = geom_prefix + suffix
    field_filename = field_prefix + suffix

    geom = pickle.load(open(geom_filename, "rb"))
    field = pickle.load(open(field_filename, "rb"))

    plot_field(geom, field[0, :, :] ** 2)


def plot_array(array, filename=None, comm=MPI.COMM_WORLD, background_value=0):
    rank = comm.Get_rank()

    all_arrays = comm.gather(array, root=0)

    if rank == 0:
        print(f"Doing the plotting", flush=True)

        z = numpy.empty_like(all_arrays[0])
        z[...] = background_value
        c1 = numpy.vstack((z, numpy.flipud(all_arrays[3]), z))
        c2 = numpy.vstack((numpy.flipud(all_arrays[4]), numpy.flipud(all_arrays[0]), numpy.flipud(all_arrays[5])))
        c3 = numpy.vstack((z, numpy.flipud(all_arrays[1]), z))
        c4 = numpy.vstack((z, numpy.flipud(all_arrays[2]), z))
        common = numpy.hstack((c1, c2, c3, c4))

        matplotlib.pyplot.clf()
        ax = matplotlib.pyplot.gca()

        im = ax.imshow(common, interpolation="nearest")
        cbar = ax.figure.colorbar(im, ax=ax)

        matplotlib.pyplot.tight_layout()

        if filename is None:
            matplotlib.pyplot.show()
        else:
            matplotlib.pyplot.savefig(filename)

    comm.Barrier()


def image_field(
    geom: "Cartesian2D",
    field: numpy.ndarray,
    filename: str,
    vmin: float,
    vmax: float,
    n: int,
    label: str = "K",
    colormap: str = "jet",
):
    device = geom.device

    domain_width = geom.x1 - geom.x0
    domain_height = geom.z1 - geom.z0
    aspect_ratio = domain_width / domain_height

    # Base height in inches, width scaled by aspect ratio
    fig_height = 10
    fig_width = fig_height * aspect_ratio

    fig, ax = matplotlib.pyplot.subplots(figsize=(fig_width, fig_height))

    cmap = matplotlib.pyplot.contourf(
        device.to_host(geom.X1_cartesian),
        device.to_host(geom.X3_cartesian),
        device.to_host(field),
        cmap=colormap,
        levels=numpy.linspace(vmin, vmax, n),
        extend="both",
    )
    ax.set_aspect("equal", "box")

    cbar = fig.colorbar(cmap, ax=ax, orientation="vertical")
    cbar.set_label(
        label,
    )

    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close(fig)

    return


def print_mountain(
    x1,
    x3,
    mountain: numpy.ndarray,
    normals_x: numpy.ndarray = None,
    normals_z: numpy.ndarray = None,
    filename: str = None,
):
    if not ((normals_x is None) == (normals_z is None)):
        raise ValueError(f"Either provide both normal arrays or none of them")

    fig, ax_mtn = matplotlib.pyplot.subplots()
    ax_mtn.scatter(x1, x3, c=mountain, marker="s")

    length_factor = 30.0
    if normals_x is not None:
        for i in range(mountain.shape[0]):
            for j in range(mountain.shape[1]):
                if (mountain[i, j] > 0) and (normals_x[i, j] ** 2 + normals_z[i, j] ** 2 > 1e-1):
                    ax_mtn.arrow(
                        x1[i, j],
                        x3[i, j],
                        normals_x[i, j] * length_factor,
                        normals_z[i, j] * length_factor,
                        head_width=40.0,
                        head_length=10.0,
                    )

    if filename is None:
        filename = "mountain.png"

    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close(fig)

    return


def print_residual_per_variable(geom, field, filename=None):

    num_levels = 20

    fig, axes = matplotlib.pyplot.subplots(2, 2, sharex=True, sharey=True)

    def plot_var(ax, vals, title):
        minval = vals.min() - 1e-15
        maxval = vals.max() + 1e-15
        if minval < 0.0 and maxval > 0.0:
            ratio = maxval / (maxval - minval)
            num_pos = int(numpy.rint((num_levels) * ratio))
            num_neg = num_levels - num_pos
            minval = -maxval / num_pos * num_neg

        levels = numpy.linspace(minval, maxval, num_levels)

        cmap = ax.contourf(geom.X1, geom.X3, vals, levels=levels)
        ax.set_title(title)
        cbar = fig.colorbar(cmap, ax=ax, orientation="vertical", format="%8.1e")
        cbar.set_label("Residual")

    plot_var(axes[0][0], field[RHO], "Rho")
    plot_var(axes[0][1], field[RHO_THETA], "Rho-theta")
    plot_var(axes[1][0], field[RHO_U], "Rho-u")
    plot_var(axes[1][1], field[RHO_W], "Rho-w")

    global plot_index

    fn = filename
    if fn is None:
        fn = f"res_plot/residual{plot_index:04d}.png"
    matplotlib.pyplot.savefig(fn)
    matplotlib.pyplot.close(fig)

    plot_index += 1

    return
