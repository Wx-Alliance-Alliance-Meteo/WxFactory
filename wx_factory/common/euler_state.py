def physical_winds(state, geometry, metric):
    """Return relative eastward, northward, and vertical winds in m/s."""
    return metric.physical_winds(state[1:4])


def state_representation(config):
    from .shallow_water_state import prescribed_wind

    expected = (
        "prescribed_velocity"
        if config.equations == "shallow_water" and prescribed_wind(config)
        else "relative_covariant"
    )
    return getattr(config, "state_representation", expected)
