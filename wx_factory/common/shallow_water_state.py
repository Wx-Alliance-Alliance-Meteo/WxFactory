"""Conversions between the shallow-water state and relative velocity."""


def prescribed_wind(config):
    return config.case_number in (0, 1, -1)


def contravariant_velocity(state, metric, config):
    if prescribed_wind(config):
        return state[1:3]
    return metric.volume.raise_velocity(state[1:3])
