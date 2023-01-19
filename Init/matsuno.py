#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:03:39 2019
@author: shlomi
"""
Earth = {
        'angular_frequency': 7.29212e-5,
        'gravitational_acceleration': 9.80616,
        'mean_radius': 6371220.,
        'layer_mean_depth': 30.
        }


def _unpack_parameters(parameters, key):
    """Unpacks the values from a dictionary containing various parameters for
    pymaws

    Parameters
    ----------
    parameters : dict
        planetary parameters dict with keys:
            angular_frequency: float, (rad/sec)
            gravitational_acceleration: float, (m/sec^2)
            mean_radius: float, (m)
            layer_mean_depth: float, (m)
    key : String
        key with the one of the names of the above keys.
    Returns
    -------
    value : Float, scalar
            one of the above values
    """
    if not isinstance(key, str):
        raise TypeError(str(key) + ' should be string...')
    if not isinstance(parameters, dict):
        raise TypeError('parameters should be dictionary...')
    if key not in parameters:
        raise KeyError(str(key) + ' not in parameters!')
    value = parameters[key]
    return value


def _eval_omega(k, n, parameters=Earth):
    """
    Evaluates the wave-frequencies for a given wave-number and wave-mode.
    For further details see Eqs. (2)-(5) in  the text

    Parameters
    ----------
    k : Integer, scalar
        spherical wave-number (dimensionless). must be >= 1.
    n : Integer, scaler
        wave-mode (dimensionless). must be >=1.
    parameters: dict
        planetary parameters dict with keys:
            angular_frequency: float, (rad/sec)
            gravitational_acceleration: float, (m/sec^2)
            mean_radius: float, (m)
            layer_mean_depth: float, (m)
        Defualt: Earth's parameters defined above

    Returns
    -------
    omega : Float, dict
            wave frequency in rad/sec for each wave-type(Rossby, EIG, WIG)

    Notes
    -----
    This function supports k>=1 and n>=1 inputs only.
    Special treatments are required for k=0 and n=-1,0/-.

    """
    import numpy as np
    # make sure k and n are scalers:
    if not np.isscalar(n):
        raise TypeError('n must be scalar')
    if not np.isscalar(k):
        raise TypeError('k must be scalar')
    # make sure input is integer:
    if not isinstance(k, (int, np.integer)):
        raise TypeError('k should be integer, i.e., ' + str(int(k)))
    if not isinstance(n, (int, np.integer)):
        raise TypeError('n should be integer, i.e., ' + str(int(n)))
    # check for k>=1 and n>=1
    if k < 1:
        raise ValueError('pymaws supports only k>=1 for now...')
    if n < 1:
        raise ValueError('pymaws supports only n>=1 for now...')
    # unpack dictionary into vars:
    OMEGA = _unpack_parameters(parameters, 'angular_frequency')
    G = _unpack_parameters(parameters, 'gravitational_acceleration')
    A = _unpack_parameters(parameters, 'mean_radius')
    H0 = _unpack_parameters(parameters, 'layer_mean_depth')
    # evaluate Eq. (4) the text
    omegaj = np.zeros((1, 3))
    delta0 = 3. * (G * H0 * (k / A)**2 + 2. * OMEGA *
                   (G * H0)**0.5 / A * (2 * n + 1))
    delta4 = -54. * OMEGA * G * H0 * k / A**2

    for j in range(1, 4):
        deltaj = (delta4**2 - 4. * delta0**3 + 0. * 1j)**0.5
        deltaj = (0.5 * (delta4 + deltaj))**(1. / 3.)
        deltaj = deltaj * np.exp(2. * np.pi * 1j * j / 3.)
        # evaluate Eq. (3) the text
        omegaj[0, j - 1] = np.real(-1. / 3. * (deltaj + delta0 / deltaj))
    # put all wave-types in dict:
    # (Eq. (5) in the text)
    omega = {'Rossby': -np.min(np.abs(omegaj)),
             'WIG': np.min(omegaj),
             'EIG': np.max(omegaj)}
    return omega


def _eval_hermite_polynomial(x, n):
    """
    Evaluates the normalized Hermite polynomial of degree n at point/s x
    using the three-term recurrence relation. For further details see Eq. (7)
    in the text.

    Parameters
    ----------
    x : Float, array_like or scalar
        list or array of points where the evalution takes place.
    n : Integer, scaler
        degree of the Hermite polynomial.

    Returns
    -------
    H_n : Float, array_like or scalar
            Evaluation of the normalized Hermite polynomial.

    Notes
    -----

    """
    import numpy as np
    # make sure n is integer and scaler:
    if not np.isscalar(n):
        raise TypeError('n must be scalar')
    if not isinstance(n, (int, np.integer)):
        raise TypeError('n should be integer, i.e., ' + str(int(n)))
    # make sure x is an array(or scalar):
    x = np.asarray(x)
    # main evaluation:
    if n < 0:
        H_n = np.zeros(x.shape)
    elif n == 0:
        H_n = np.ones(x.shape) / np.pi**0.25
    elif n == 1:
        H_n = (4.0 / np.pi)**0.25 * x
    elif n >= 2:
        H_n = ((2.0 / n)**0.5 * x * _eval_hermite_polynomial(x, n - 1) -
               ((n - 1) / n)**0.5 * _eval_hermite_polynomial(x, n - 2))
    return H_n


def _eval_meridional_velocity(lat, Lamb, n=1, amp=1e-5):
    """
    Evaluates the meridional velocity amplitude at a given latitude point and
    a given wave-amplitude. See Eq.(6a) in the text.

    Parameters
    ----------
    lat : Float, array_like or scalar
          latitude(radians)
    Lamb: Float, scalar
          Lamb's parameter. ~ 2935 for Earth's parameters
    n : Integer, scaler
        wave-mode (dimensionless)
        Default : 1
    amp : Float, scalar
          wave amplitude(m/sec)
          Default : 1e-5

    Returns
    -------
    vel : Float, array_like or scalar
          Evaluation of the meridional velocity.

    Notes
    -----
    This function supports n>=1 inputs only.
    Special treatments are required for n=-1,0/-.

    """
    import numpy as np
    if not np.isscalar(amp):
        raise TypeError('amp must be scalar')
    # re-scale latitude
    y = Lamb**0.25 * lat

    # Gaussian envelope
    ex = np.exp(-0.5 * y**2)

    vel = amp * ex * _eval_hermite_polynomial(y, n)

    return vel


def _eval_field_amplitudes(lat, k=5, n=1, amp=1e-5, field='v',
                           wave_type='Rossby', parameters=Earth):
    """
    Evaluates the latitude dependent amplitudes at a given latitude point.

    Parameters
    ----------
    lat : Float, array_like or scalar
          latitude(radians)
    k : Integer, scalar
        spherical wave-number (dimensionless)
        Default : 5
    n : Integer, scaler
        wave-mode (dimensionless)
        Default : 1
    amp : Float, scalar
          wave amplitude(m/sec)
          Default : 1e-5
    field : str
            pick 'phi' for geopotential height,
            'u' for zonal velocity and 'v' for meridional velocity
            Defualt : 'v'
    wave_type: str
        choose Rossby waves or WIG waves or EIG waves.
        Defualt: Rossby
    parameters: dict
        planetary parameters dict with keys:
            angular_frequency: float, (rad/sec)
            gravitational_acceleration: float, (m/sec^2)
            mean_radius: float, (m)
            layer_mean_depth: float, (m)
        Defualt: Earth's parameters defined above
    Returns
    -------
    Either u_hat(m/sec), v_hat(m/sec) or p_hat(m^2/sec^2) : Float, array_like
    or scalar Evaluation of the amplitudes for the zonal velocity,
    or meridional velocity or the geopotential height respectivly.

    Notes
    -----
    This function supports k>=1 and n>=1 inputs only.
    Special treatments are required for k=0 and n=-1,0/-.

    """
    if not isinstance(wave_type, str):
        raise TypeError(str(wave_type) + ' should be string...')
    # unpack dictionary into vars:
    OMEGA = _unpack_parameters(parameters, 'angular_frequency')
    G = _unpack_parameters(parameters, 'gravitational_acceleration')
    A = _unpack_parameters(parameters, 'mean_radius')
    H0 = _unpack_parameters(parameters, 'layer_mean_depth')
    # Lamb's parameter:
    Lamb = (2. * OMEGA * A)**2 / (G * H0)
    # evaluate wave frequency:
    all_omegas = _eval_omega(k, n, parameters)
    # check for validity of wave_type:
    if wave_type not in all_omegas:
        raise KeyError(wave_type + ' should be Rossby, EIG or WIG...')
    omega = all_omegas[wave_type]
    # evaluate the meridional velocity amp first:
    v_hat = _eval_meridional_velocity(lat, Lamb, n, amp)
    # evaluate functions for u and phi:
    v_hat_plus_1 = _eval_meridional_velocity(lat, Lamb, n + 1, amp)
    v_hat_minus_1 = _eval_meridional_velocity(lat, Lamb, n - 1, amp)
    # Eq. (6a) in the text
    if field == 'v':
        return v_hat
    # Eq. (6b) in the text
    elif field == 'u':
        u_hat = (- ((n + 1) / 2.0)**0.5 * (omega / (G * H0)**0.5 + k / A) *
                 v_hat_plus_1 - ((n) / 2.0)**0.5 * (omega / (G * H0)**0.5 -
                                                    k / A) * v_hat_minus_1)
        # pre-factors
        u_hat = G * H0 * Lamb**0.25 / \
            (1j * A * (omega**2 - G * H0 * (k / A)**2)) * u_hat
        return u_hat
    # Eq. (6c) in the text
    elif field == 'phi':
        p_hat = (- ((n + 1) / 2.0)**0.5 * (omega + (G * H0)**0.5 * k / A) *
                 v_hat_plus_1 + ((n) / 2.0)**0.5 * (omega - (G * H0)**0.5 *
                                                    k / A) * v_hat_minus_1)
        p_hat = G * H0 * Lamb**0.25 / \
            (1j * A * (omega**2 - G * H0 * (k / A)**2)) * p_hat
        return p_hat
    else:
        raise KeyError('field must be u, v or phi')


def eval_field(lat, lon, time, k=5, n=1, amp=1e-5, field='phi',
               wave_type='Rossby', parameters=Earth):
    """
    Evaluates the analytic solutions of either the zonal or meridional velocity
    or the geopotential height on at given latitude, longitude and time.

    Parameters
    ----------
    lat : Float, scalar
          latitude(radians)
    lon : Float, scalar
          longitude(radians)
    time : Float, scaler
           time(sec), =0 if one wants initial conditions.
    k : Integer, scalar
        spherical wave-number (dimensionless)
        Default : 5
    n : Integer, scaler
        wave-mode (dimensionless)
        Default : 1
    amp : Float, scalar
          wave amplitude(m/sec)
          Defualt : 1e-5
    field : str
            pick 'phi' for geopotential height,
            'u' for zonal velocity and 'v' for meridional velocity
            Defualt : 'phi'
    wave_type: str
        choose Rossby waves or WIG waves or EIG waves.
        Defualt: Rossby
    parameters: Float, dict
        planetary parameters dict with keys:
            angular_frequency: float, (rad/sec)
            gravitational_acceleration: float, (m/sec^2)
            mean_radius: float, (m)
            layer_mean_depth: float, (m)
        Defualt: Earth's parameters defined above
    Returns
    -------
    f : Float, scalar
        Evaluation of the the zonal velocity(m/sec),
        or meridional velocity(m/sec) or the geopotential height(m^2/sec^2)
        respectivly.

    Notes
    -----
    This function supports k>=1 and n>=1 inputs only.
    Special treatments are required for k=0 and n=-1,0/-.

    """
    import numpy as np

    # make sure lat, lon and time are scalars
    if not np.isscalar(lat):
        raise TypeError('lat must be scalar')
    if not np.isscalar(lon):
        raise TypeError('lon must be scalar')
    if not np.isscalar(time):
        raise TypeError('time must be scalar')

    # frequency
    all_omegas = _eval_omega(k, n, parameters)
    if wave_type not in all_omegas:
        raise KeyError(wave_type + ' should be Rossby, EIG or WIG...')
    omega = all_omegas[wave_type]

    # latitude-dependent amplitudes
    if field == 'phi':
        f_hat = _eval_field_amplitudes(lat, k, n, amp, 'phi', wave_type,
                                       parameters)
    elif field == 'u':
        f_hat = _eval_field_amplitudes(lat, k, n, amp, 'u', wave_type,
                                       parameters)
    elif field == 'v':
        f_hat = _eval_field_amplitudes(lat, k, n, amp, 'v', wave_type,
                                       parameters)

    # adding time and longitude dependence
    f = np.real(np.exp(1j * (k * lon - omega * time)) * f_hat)

    if field == 'phi':
       f += parameters['gravitational_acceleration']*parameters['layer_mean_depth']

    return f
