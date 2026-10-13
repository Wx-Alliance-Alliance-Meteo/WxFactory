def physical_winds(state, geometry, metric):
    """Return relative eastward, northward, and vertical winds in m/s."""
    return metric.physical_winds(state[1:4])


def state_representation(config):
    expected = "relative_covariant" if config.equations == "euler" else "conservative"
    return getattr(config, "state_representation", expected)
