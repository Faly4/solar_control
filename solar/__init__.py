import os

import numpy
import numpy as np
import pandas
import pandas as pd
from dev_tools import div_zeros, lim_ang, poly_ord_3rd, second_compute, simple_da, yr_float
from matplotlib import pyplot as plt

location = os.path.dirname(os.path.realpath(__file__))
my_file0 = os.path.join(location, 'data', 'coeff0y.npy')
my_file1 = os.path.join(location, 'data', 'coeff0.npy')
my_file2 = os.path.join(location, 'data', 'delta_t.npy')
my_file3 = os.path.join(location, 'data', 'data_iut.pkl')
iut_data = pd.read_pickle(my_file3)


def aberr_corr(r0):
    """
    Calculate the aberration correction

    Parameters
    ----------
    r0: numpy.ndarray
        Earth radius vector in Astronomical Units (AU)

    Returns
    -------
    numpy.ndarray
        Aberration correction (°)
    """
    return -20.4898 / 3600 / r0


def appt_sun_long(θ2, Δψ, Δτ):
    r"""
    Calculate the apparent sun longitude.

    Parameters
    ----------
    θ2 : numpy.ndarray
        Geocentric longitude :math:`\theta_0` (rad)
    Δψ : numpy.ndarray
        Nutation in longitude (°)
    Δτ : numpy.ndarray
        Aberration correction (°)

    Returns
    -------
    numpy.ndarray
        Apparent sun longitude (rad)

    """
    return np.deg2rad(lim_ang(np.rad2deg(θ2) + Δψ + Δτ, 360))


def appt_sidereal_tm(jd, jc, Δψ, ε):
    r"""
    Calculate the apparent sidereal time at Greenwich at any given time.

    Parameters
    ----------
    jd : numpy.ndarray
        Julian Day
    jc : numpy.ndarray
        Julian century
    Δψ : numpy.ndarray
        Nutation in longitude (°)
    ε : numpy.ndarray
        True obliquity of the ecliptic ε (rad)

    Returns
    -------
    numpy.ndarray
        Apparent sidereal time at Greenwich (°)

    """
    v1 = poly_ord_3rd(- 1 / 38710000, 0.000387933, 0, 280.46061837, jc)
    ν0 = lim_ang(v1 + 360.98564736629 * (jd - 2451545), 360)
    return ν0 + Δψ * np.cos(ε)


def earth_term(χ, jme, param):
    r"""
    Calculate the Earth heliocentric longitude, latitude, or radius vector.

    Parameters
    ----------
    χ : numpy.ndarray
        Coefficient used computation for the earth  heliocentric longitude, latitude, or radius vector
    jme : numpy.ndarray
        Julian ephemeris millennium
    param : list
        List of used parameter. Example for heliocentric radius vector param = ['R0', 'R1', 'R2', 'R3', 'R4']

    Returns
    -------
    numpy.ndarray
        Earth heliocentric longitude, latitude, or radius vector (rad)
    """
    v = len(jme)
    mx0 = χ.item().get(param)
    a0 = mx0[:, 0]
    b0 = mx0[:, 1]
    c0 = mx0[:, 2]
    u1 = np.shape(c0)[0]
    a00 = np.tile(a0, (v, 1))
    b00 = np.tile(b0, (v, 1))
    c00 = np.tile(c0, (v, 1))
    jme00 = np.transpose(np.tile(jme, (u1, 1)))
    k0 = a00 * np.cos(b00 + c00 * jme00)
    uni0 = np.ones(u1)
    return np.matmul(k0, uni0)


def earth_helio(χ, jme, param):
    r"""
    Calculate the Earth heliocentric longitude, latitude, and radius vector coefficient. It uses Earth Periodic Terms

    Parameters
    ----------
    jme : numpy.ndarray
        Julian ephemeris millennium
    χ : numpy.ndarray
        Earth Periodic Terms
    param : list
        List of used parameter. Example for heliocentric radius vector param = ['R0', 'R1', 'R2', 'R3', 'R4']

    Returns
    -------
    numpy.ndarray
        Earth heliocentric longitude, latitude, and radius vector coefficient (rad)
    """
    earth1data = np.zeros((6, len(jme)))
    s = -1
    for v in param:
        s = s + 1
        earth1data[s, :] = earth_term(χ, jme, v)
    k0 = earth1data[0, :]
    k1 = earth1data[1, :] * jme
    k2 = earth1data[2, :] * jme ** 2
    k3 = earth1data[3, :] * jme ** 3
    k4 = earth1data[4, :] * jme ** 4
    k5 = earth1data[5, :] * jme ** 5
    return (k0 + k1 + k2 + k3 + k4 + k5) / 1e8


def nutation_long_obl(jce):
    r"""
    Calculate the nutation in longitude Δε and the nutation in obliquity Δψ. The computation uses the periodic terms
    for the nutation in longitude and obliquity. This function use :func:`poly_ord_3rd`

    Parameters
    ----------
    jce : numpy.ndarray
        Julian Ephemeris Day

    Returns
    -------
    tuple
        * Δε (numpy.ndarray) - Nutation in obliquity (°)
        * Δψ (numpy.ndarray) - Nutation in longitude (°)
    """
    nut2pd2terms = np.load(my_file0, allow_pickle=True)
    x0 = poly_ord_3rd(1.0 / 189474.0, -0.0019142, 445267.11148, 297.85036, jce)
    x1 = poly_ord_3rd(-1.0 / 300000.0, -0.0001603, 35999.05034, 357.52772, jce)
    x2 = poly_ord_3rd(1.0 / 56250.0, 0.0086972, 477198.867398, 134.96298, jce)
    x3 = poly_ord_3rd(1.0 / 327270.0, -0.0036825, 483202.017538, 93.27191, jce)
    x4 = poly_ord_3rd(1.0 / 450000.0, 0.0020708, -1934.136261, 125.04452, jce)
    v = len(x0)
    u = len(nut2pd2terms[:, 0])
    y0 = np.tile(nut2pd2terms[:, 0], (v, 1))
    y1 = np.tile(nut2pd2terms[:, 1], (v, 1))
    y2 = np.tile(nut2pd2terms[:, 2], (v, 1))
    y3 = np.tile(nut2pd2terms[:, 3], (v, 1))
    y4 = np.tile(nut2pd2terms[:, 4], (v, 1))
    a0 = np.tile(nut2pd2terms[:, 5], (v, 1))
    b0 = np.tile(nut2pd2terms[:, 6], (v, 1))
    c0 = np.tile(nut2pd2terms[:, 7], (v, 1))
    d0 = np.tile(nut2pd2terms[:, 8], (v, 1))
    x00 = np.transpose(np.tile(x0, (u, 1)))
    x10 = np.transpose(np.tile(x1, (u, 1)))
    x20 = np.transpose(np.tile(x2, (u, 1)))
    x30 = np.transpose(np.tile(x3, (u, 1)))
    x40 = np.transpose(np.tile(x4, (u, 1)))
    jce = np.transpose(np.tile(jce, (u, 1)))
    m0 = y0 * x00 + y1 * x10 + y2 * x20 + y3 * x30 + y4 * x40
    m0 = np.deg2rad(m0)
    Δψ = np.matmul((a0 + b0 * jce) * np.sin(m0), np.ones(u)) / 36e6
    Δε = np.matmul((c0 + d0 * jce) * np.cos(m0), np.ones(u)) / 36e6
    return Δε, Δψ


def earth_helio_all(jme):
    """
    Compute heliocentric latitude and Earth heliocentric longitude based on Earth Periodic Terms. This
    function need function :func:`earth_helio`.

    Parameters
    ----------
    jme : numpy.ndarray
        Julian ephemeris millennium

    Returns
    -------
    tuple
        * b (numpy.ndarray) - Earth heliocentric latitude (rad)
        * l0 (numpy.ndarray) - Earth heliocentric longitude (rad)
        * r0 (numpy.ndarray) - Earth heliocentric radius vector in Astronomical Units (AU)

    """
    earth0data = np.load(my_file1, allow_pickle=True)
    l0 = earth_helio(earth0data, jme, ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'])
    b0 = -earth_helio(earth0data, jme, ['B0', 'B1'])  # β=-b
    r0 = earth_helio(earth0data, jme, ['R0', 'R1', 'R2', 'R3', 'R4'])
    return lim_ang(b0, 2 * np.pi), lim_ang(l0, 2 * np.pi), r0


def obl_eclip(jme, Δε):
    r"""
    Calculate the true obliquity of the ecliptic.
    Parameters
    ----------
    jme : numpy.ndarray
        Julian ephemeris millennium
    Δε : numpy.ndarray
        Nutation in obliquity (°)

    Returns
    -------
    numpy.ndarray
        True obliquity of the ecliptic ε (rad)
    """
    u5 = jme / 10
    ε0 = (84381.448 + u5 * (-4680.93 + u5 * (-1.55 + u5 * (1999.25 + u5 * (-51.38 + u5 * (
            -249.67 + u5 * (-39.05 + u5 * (7.12 + u5 * (27.87 + u5 * (5.79 + u5 * 2.45)))))))))) / 3600 + Δε
    return lim_ang(np.deg2rad(ε0), 2 * np.pi)


def geo_long(l0):
    r"""
    Compute the geocentric longitude.

    Parameters
    ----------
    l0 : numpy.ndarray
        Earth heliocentric longitude (rad)

    Returns
    -------
    numy.ndarray
        Geocentric longitude :math:`\theta_0` (rad)

    """
    return lim_ang(l0 + np.pi, 2 * np.pi)


def ephem_da(yr, mon, da, hr, Δt):
    """
    Calculate the Julian and Julian Ephemeris Day, Century, and Millennium

    Parameters
    ----------
    yr : np.ndarray
        Year of the considered date
    mon : np.ndarray
        Month of the considered date
    da : np.ndarray
        Day of the month of the considered date
    hr : np.ndarray
        Decimal hour of the date
    Δt : np.ndarray
        Earth delta t time in second for a specific date

    Returns
    -------
    tuple
        * jce (numpy.ndarray) - Julian Ephemeris Day
        * jc (numpy.ndarray) - Julian century
        * jd (numpy.ndarray) - Julian Day
        * jme (numpy.ndarray) - Julian ephemeris millennium
    """
    rx0 = np.array((yr > 1582).astype(float))
    n0 = np.array((mon > 2).astype(float))
    n1 = np.array((mon < 3).astype(float))
    yb = yr * n0 + (yr - 1) * n1
    mb = mon * n0 + (mon + 12) * n1
    a = np.floor(yb / 100)
    bx0 = (2 - a + np.floor(a / 4)) * rx0
    dx0 = da + hr / 24
    jd = np.floor(365.25 * (yb + 4716)) + np.floor(30.6001 * (mb + 1)) + dx0 + bx0 - 1524.5
    jde = jd + Δt / 86400
    jc = (jd - 2451545) / 36525
    jce = (jde - 2451545.0) / 36525
    jme = jce / 10
    return jc, jce, jd, jme


def geo_sun_r_ac(λ, ε, β):
    r"""
    Calculate the geocentric sun right ascension.


    Parameters
    ----------
    λ : numpy.ndarray
        Apparent sun longitude (rad)
    ε : numpy.ndarray
        True obliquity of the ecliptic ε (rad)
    β : numpy.ndarray
        Geocentric latitude (rad)


    Returns
    -------
    numpy.ndarray
        Geocentric sun right ascension α (rad)

    """
    α = lim_ang(np.arctan2(np.sin(λ) * np.cos(ε) - np.tan(β) * np.sin(ε), np.cos(λ)), 2 * np.pi)
    return α


def geo_sun_dec(λ, ε, β):
    r"""
    Calculate the geocentric sun declination


    Parameters
    ----------
    λ : numpy.ndarray
        Apparent sun longitude (rad)
    ε : numpy.ndarray
        True obliquity of the ecliptic ε (rad)
    β : numpy.ndarray
        Geocentric latitude (rad)

    Returns
    -------
    numpy.ndarray
        Geocentric sun declination :math:`\delta` (rad)
    """
    return np.arcsin(np.sin(β) * np.cos(ε) + np.cos(β) * np.sin(ε) * np.sin(λ))


def obs_lcl_hr_ang(α, ν, σ):
    r"""
    Calculate the observer local hour angle.


    Parameters
    ----------
    α : numpy.ndarray
        Geocentric sun right ascension (rad)
    ν : numpy.ndarray
        Apparent sidereal time at Greenwich (°)
    σ : float
        Observer geographical longitude


    Returns
    -------
    numpy.ndarray
        Observer local hour angle (°)
    """
    return lim_ang(ν + σ - np.rad2deg(α), 360)


def topo_el_angl_corr(press, temp, e0):
    r"""
    Calculate the topocentric zenith angle with atmospheric refraction correction

    Parameters
    ----------
    press : float
        Annual average local pressure in millibars (mbar)
    temp : float
        Annual average local temperature celsius (°C)
    e0 : numpy.ndarray
        Elevation angle without atmospheric refraction correction (°)

    Returns
    -------
    numpy.ndarray
        Topocentric zenith angle (rad)
    """
    Δe = press / 1010 * 283 / (273 + temp) * 1.02 / (60 * np.tan(np.deg2rad(e0 + 10.3 / (e0 + 5.11))))
    e = e0 + Δe
    return np.deg2rad(90 - e)


def topo_astro_az_ang(hrt, φ, δ1):
    r"""
    Calculate the topocentric astronomers azimuth angle Φ.

    Parameters
    ----------
    hrt : numpy.ndarray
        Topocentric local hour angle (rad)
    φ : float
        Geocentric latitude (rad)
    δ1 : numpy.ndarray
        Topocentric sun declination (rad)

    Returns
    -------
    numpy.ndarray
        Topocentric azimuth angle (rad)
    """
    Γ = np.arctan2(np.sin(hrt), np.cos(hrt) * np.sin(φ) - np.tan(δ1) * np.cos(φ))
    Φ = lim_ang(Γ + np.pi, 2 * np.pi)
    return Φ


def incid_angle_surf(hrt, φ, δ1, ω, θ, γ):
    r"""
    Calculate the incidence angle for a surface oriented in any
    direction :math:`I_0`


    Parameters
    ----------
    hrt : numpy.ndarray
        Topocentric local hour angle (rad)
    φ : float
        Geocentric latitude (rad)
    δ1 : numpy.ndarray
        Topocentric sun declination (rad)
    ω : float
        Slope of the surface measured from the horizontal plane (rad)
    θ : numpy.ndarray
        Topocentric zenith angle (rad)
    γ : float
        Surface azimuth rotation angle, measured from south to the projection of the surface normal on the horizontal
        plane, positive or negative if oriented west or east from south, respectively

    Returns
    -------
    numpy.ndarray
        Incidence angle for a surface oriented in any
    direction
    """
    Γ = np.arctan2(np.sin(hrt), np.cos(hrt) * np.sin(φ) - np.tan(δ1) * np.cos(φ))
    i0 = np.arccos(np.cos(θ) * np.cos(ω) + np.sin(ω) * np.sin(θ) * np.cos(Γ - γ))
    return i0


def topo_sun(α, φ, δ, hro, el, r0):
    r"""
    Compute Topocentric sun right ascension :math:`\alpha'`, Parallax in the sun right ascension Δα and Topocentric
    sun declination :math:`\delta'`.


    Parameters
    ----------
    α : numpy.ndarray
        Geocentric sun right ascension (rad)
    φ : float
        Geocentric latitude (rad)
    δ : numpy.ndarray
        Geocentric sun declination :math:`\delta` (rad)
    hro : numpy.ndata
        Observer local hour angle (rad)
    el : float
        Elevation in meter (m)
    r0 : numpy.ndarray
        Earth heliocentric radius vector in Astronomical Units (AU)


    Returns
    -------
    tuple
        * :math:`\alpha'` (numpy.ndarray) - Topocentric sun right ascension (rad)
        * Δα (numpy.ndarray) - Parallax in the sun right ascension (rad)
        * :math:`\delta'` (numpy.ndarray) - Topocentric sun declination (rad)
    """
    # equatorial horizontal parallax of the sun
    ξ = np.deg2rad(8.794 / 3600 / r0)
    # the term u
    u = np.arctan(0.99664719 * np.tan(φ))
    x = np.cos(u) + el / 6378140 * np.cos(φ)
    y = 0.99664719 * np.sin(u) + el / 6378140 * np.sin(φ)
    Δα = np.arctan2(-x * np.sin(ξ) * np.sin(hro), np.cos(δ) - x * np.sin(ξ) * np.cos(hro))
    α1 = α + Δα
    δ1 = np.arctan2((np.sin(δ) - y * np.sin(ξ)) * np.cos(Δα), np.cos(δ) - y * np.sin(ξ) * np.cos(hro))
    return α1, Δα, δ1


def topo_lcl_hour_angl(hro, Δα):
    r"""
    Calculate the topocentric local hour angle.

    Parameters
    ----------
    Δα : numpy.ndarray
        Parallax in the sun right ascension (rad)
    hro : numpy.ndata
        Observer local hour angle (rad)


    Returns
    -------
    numpy.ndarray
        Topocentric local hour angle (rad)
    """
    return hro - Δα


def earth_dlt_t(time_step, fig=False):
    r"""
    Compute earth delta_t from historical and forecast given by the U.S. Naval Observatory from year 1657 to
    2033.75.

    Parameters
    ----------
    time_step : tuple
        List of date with Python datetime format
    fig : bool
        if set True, plot U.S. Naval Observatory data

    Returns
    -------
    numpy.ndarray
        Earth Δt in seconds (s)
    """

    delta_t = np.load(my_file2)  # load delta_t data
    k0 = yr_float(time_step)  # convert year to float
    if fig:
        fig, ax = plt.subplots()
        ax.plot(delta_t[:, 0], delta_t[:, 1], color='black')
        ax.axvspan(np.min(delta_t[:, 0]), 1973, color='y', alpha=0.5, lw=0)
        ax.axvspan(1973, 2023, color='darkslategray', alpha=0.7, lw=0)
        ax.axvspan(2023, np.max(delta_t[:, 0]), color='r', alpha=0.5, lw=0)
        ax.legend(['Δt', 'historical: year/2', 'historical: month', 'predicted: year/4'])
        ax.axvline(x=1973, color='b', label='aniline - full height')
        plt.text(1973 + 1, np.mean(delta_t[:, 1]) / 7, '1973', rotation=90, verticalalignment='center')
        ax.axvline(x=2023, color='b', label='aniline - full height')
        plt.text(2023 + 1, np.mean(delta_t[:, 1]) / 7, '2023', rotation=90, verticalalignment='center')
        ax.title.set_text('Δt')
        ax.set_xlabel('year')
        ax.set_ylabel('Δt [seconds]')
        text0 = ax.text(np.min(delta_t[:, 0]) * 1.005, np.min(delta_t[:, 1]), 'U.S. Naval Observatory\n1657 to 2033.75',
                        horizontalalignment='left', fontsize=8)
        text0.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        ax.grid()
    return np.interp(k0, delta_t[:, 0], delta_t[:, 1])


def almanac(el, offset, press, temp, time_step, γ, σ, φ, ω):
    r"""
    Compute solar positions.

    Parameters
    ----------
    el : float
        Elevation in meter (m)
    offset : float
        Decimal hour from UTC time
    press : float
        annual average local pressure in millibars (mbar)
    temp : float
        Annual average local temperature Celsius (°C)
    time_step : tuple
        List of date with Python datetime format
    γ : float
        Surface azimuth rotation angle, measured from south to the projection of the surface normal on the horizontal
        plane, positive or negative if oriented west or east from south, respectively
    σ : float
        Observer geographical longitude (°)
    φ : float
        Geocentric latitude (rad)
    ω : float
        Slope of the surface measured from the horizontal plane (rad)


    Returns
    -------
    pandas.DataFrame
        * jd - Julian Day
        * β - Geocentric latitude (°)
        * l0 - Earth heliocentric longitude (°)
        * r0 (- Earth heliocentric radius vector in Astronomical Units (AU)
        * :math:`\theta_0` - Geocentric longitude (°)
        * Δε - Nutation in obliquity (°)
        * Δψ - Nutation in longitude (°)
        * ε - True obliquity of the ecliptic  in radian (°)
        * λ - Apparent sun longitude (°)
        * α - Geocentric sun right ascension (°)
        * :math:`\delta` - Geocentric sun declination  in radian (°)
        * θ - Topocentric zenith angle (°)
        * Φ - Topocentric azimuth angle (°)
        * :math:`I_0` - Incidence angle for a surface oriented in any direction (°)
    """
    Δt = earth_dlt_t(time_step)
    da, hr, yr, mon = simple_da(time_step, offset)
    jc, jce, jd, jme = ephem_da(yr, mon, da, hr, Δt)
    β, l0, r0 = earth_helio_all(jme)
    θ2 = geo_long(l0)
    Δε, Δψ = nutation_long_obl(jce)
    ε = obl_eclip(jme, Δε)
    Δτ = aberr_corr(r0)
    λ = appt_sun_long(θ2, Δψ, Δτ)
    ν = appt_sidereal_tm(jd, jc, Δψ, ε)
    α = geo_sun_r_ac(λ, ε, β)
    δ = geo_sun_dec(λ, ε, β)
    hro = np.deg2rad(obs_lcl_hr_ang(α, ν, σ))
    α1, Δα, δ1 = topo_sun(α, φ, δ, hro, el, r0)
    hrt = topo_lcl_hour_angl(hro, Δα)
    e0 = topo_el_angl(φ, δ1, hrt)
    θ = topo_el_angl_corr(press, temp, e0)
    Φ = topo_astro_az_ang(hrt, φ, δ1)
    i0 = incid_angle_surf(hrt, φ, δ1, ω, θ, γ)
    out0 = np.column_stack((jd, np.rad2deg(β), np.rad2deg(l0), r0, np.rad2deg(θ2), Δε, Δψ, np.rad2deg(ε), np.rad2deg(λ),
                            np.rad2deg(α), np.rad2deg(δ), np.rad2deg(θ), np.rad2deg(Φ), np.rad2deg(i0)))
    col_name = ['Julian_da', 'Geo_lat', 'Earth_helio_long', 'Earth_helio_rad_vec', 'Geo_long', 'Nutat_obl',
                'Nutat_long', 'True_obl_ecl', 'Appt_sun_long', 'Geo_sun_r_ac', 'Geo_sun_dec', 'Topo_z_ang',
                'Topo_az_ang', 'Incid_angl']
    return pd.DataFrame(out0, columns=col_name)


def topo_el_angl(φ, δ1, hrt):
    r"""
    Calculate the topocentric elevation angle without atmospheric refraction correction.

    Parameters
    ----------
    φ : float
        Geocentric latitude (rad)
    δ1 : numpy.ndarray
        Topocentric sun declination (rad)
    hrt : numpy.ndarray
        Topocentric local hour angle (rad)

    Returns
    -------
    numpy.ndarray
        Topocentric elevation angle without atmospheric refraction correction (°)
    """
    mx0 = np.arcsin(np.sin(φ) * np.sin(δ1) + np.cos(φ) * np.cos(δ1) * np.cos(hrt))
    return np.rad2deg(mx0)


def bird_clear_sky(time_step, offset, φ, σ, pssr, oz, wtr, τ500, τ380, ba, rg):
    """
    Compute Brid's clear sky model for radiaion.

    Parameters
    ----------
    time_step : pandas.DatetimeIndex
        Pandas.DatetimeIndex
    offset : float
        Time zone from UTC
    φ : float
        Latitude (°)
    σ : float
        Longitude (°)
    pssr : float
        Surface pressure (mBar)
    oz : float
        Amount of ozone in a vertical column from surface (cm)
    wtr : float
        Amount of precipitable water in a vertical column from surface (cm)
    τ500 : float
        Aerosol optical depth from surface in a vertical path at 0.5 μm wavelength
    τ380 : float
        Aerosol optical depth from surface in a vertical path at 0.38 μm wavelength
    ba : float
        Ratio of the forward-scattered irradiance to the total scattered irradiance due to aerosols
    rg : float
        Ground albedo

    Returns
    -------
    pandas.DataFrame
        * Day of year
        * Hour in decimal hour
        * Extraterrestrial radiation in Watt per meter square (W/m²)
        * Day angle (°)
        * Solar declination (°)
        * Equation of time (min)
        * Hour angle (°)
        * Angle between a line to the sun and the local zenith (°)
        * Air mass
        * Transmittance of Rayleigh scattering
        * Transmittance of ozone absorptance
        * Transmittance of absorptance of uniformly mixed gases (carbon dioxide and oxygen)
        * Transmittance of water vapor absorptance
        * Transmittance of aerosol absorptance and scattering
        * Transmittance of aerosol absorptance
        * rs
        * Direct Normal insolation (W/m²)
        * Direct solar irradiance on a horizontal surface (W/m²)
        * Solar irradiance on a horizontal surface from scattered light (W/m²)
        * Total (global) solar irradiance on a horizontal surface (W/m²)
        * Day of year and hour in decimal
        * Diffuse solar irradiance on a horizontal surface (W/m²)
    """
    t_aua = 0.2758 * τ380 + 0.35 * τ500
    doy = time_step.day_of_year.to_numpy(dtype=float)  # day of year
    da, hr, ya, month = simple_da(time_step, 0)
    xta_rdan = 1367 * (1.00011 + 0.034221 * np.cos(2 * np.pi * (doy - 1) / 365) + 0.00128 * np.sin(
        2 * np.pi * (doy - 1) / 365) + 0.000719 * np.cos(2 * (2 * np.pi * (doy - 1) / 365)) + 0.000077 * np.sin(
        2 * (2 * np.pi * (doy - 1) / 365)))
    da_ang = 6.283185 * (doy - 1) / 365
    dec = (0.006918 - 0.399912 * np.cos(da_ang) + 0.070257 * np.sin(da_ang) - 0.006758 * np.cos(
        2 * da_ang) + 0.000907 * np.sin(2 * da_ang) - 0.002697 * np.cos(3 * da_ang) + 0.00148 * np.sin(3 * da_ang)) * (
                  180 / np.pi)
    eqn_tm = (0.0000075 + 0.001868 * np.cos(da_ang) - 0.032077 * np.sin(da_ang) - 0.014615 * np.cos(
        2 * da_ang) - 0.040849 * np.sin(2 * da_ang)) * 229.18
    hr_ang = 15 * (hr - 12.5) + σ - offset * 15 + eqn_tm / 4
    z_ang = np.arccos(
        np.cos(dec / (180 / np.pi)) * np.cos(φ / (180 / np.pi)) * np.cos(hr_ang / (180 / np.pi)) + np.sin(
            dec / (180 / np.pi)) * np.sin(φ / (180 / np.pi))) * (180 / np.pi)
    idx_z0 = (z_ang < 89).astype(float)  # filter zenith angle at 89
    idx_z1 = (z_ang < 90).astype(float)  # filter zenith angle at 90
    z_ang1 = z_ang * idx_z0 + (np.ones(len(idx_z0)) - idx_z0) * 85
    air_mass = (1 / (np.cos(z_ang1 / (180 / np.pi)) + 0.15 / (93.885 - z_ang1) ** 1.25)) * idx_z0
    idx_air_mass = (air_mass > 0).astype(float)  # filter air mass
    t_rayliegh = np.exp(-0.0903 * (pssr * air_mass / 1013) ** 0.84 * (
            1 + pssr * air_mass / 1013 - (pssr * air_mass / 1013) ** 1.01)) * idx_air_mass
    t_oz = (1 - 0.1611 * (oz * air_mass) * (1 + 139.48 * (oz * air_mass)) ** -0.3034 - 0.002715 * (oz * air_mass) / (
            1 + 0.044 * (oz * air_mass) + 0.0003 * (oz * air_mass) ** 2)) * idx_air_mass
    t_gases = np.exp(-0.0127 * (air_mass * pssr / 1013) ** 0.26) * idx_air_mass
    t_wtr = (1 - 2.4959 * air_mass * wtr / (
            (1 + 79.034 * wtr * air_mass) ** 0.6828 + 6.385 * wtr * air_mass)) * idx_air_mass
    t_aerosol = np.exp(-(t_aua ** 0.873) * (1 + t_aua - t_aua ** 0.7088) * air_mass ** 0.9108) * idx_air_mass
    t_aa = (1 - 0.1 * (1 - air_mass + air_mass ** 1.06) * (
            1 - t_aerosol)) * idx_air_mass  # Transmittance of aerosol absorptance
    rs = (0.0685 + (1 - ba) * (1 - div_zeros(t_aerosol, t_aa))) * idx_air_mass
    id0 = (0.9662 * xta_rdan * t_aerosol * t_wtr * t_gases * t_oz * t_rayliegh) * idx_air_mass
    idnh = id0 * np.cos(z_ang / (180 / np.pi)) * idx_z1
    ias = (xta_rdan * np.cos(z_ang / (180 / np.pi)) * 0.79 * t_oz * t_gases * t_wtr * t_aa * (
            0.5 * (1 - t_rayliegh) + ba * (1 - (div_zeros(t_aerosol, t_aa)))) / (
                   1 - air_mass + air_mass ** 1.02)) * idx_air_mass
    gh = (idnh + ias) / (1 - rg * rs) * idx_air_mass
    d_tm = doy + (hr - 0.5) / 24
    diff_hz = gh - idnh
    out0 = np.column_stack((doy, hr, xta_rdan, da_ang, dec, eqn_tm, hr_ang, z_ang, air_mass, t_rayliegh, t_oz, t_gases,
                            t_wtr, t_aerosol, t_aa, rs, id0, idnh, ias, gh, d_tm, diff_hz))
    col_name = ['doy', 'hr', 'xta_rdan', 'da_ang', 'dec', 'eqn_tm', 'hr_ang', 'z_ang', 'air_mass', 't_rayliegh', 't_oz',
                't_gases', 't_wtr', 't_aerosol', 't_aa', 'rs', 'id0', 'idnh', 'ias', 'gh', 'd_tm', 'diff_hz']
    return pd.DataFrame(out0, columns=col_name)


def ecdf0(arr, q, b=500):
    """
    Return empirical at given quantile *q* with linear interpolation.

    Parameters
    ----------
    b : int
        Number of class for frequency construction. If not given, :math:`b=500`.
    q : numpy.ndarray
        Quantile value
    arr : numpy.ndarray
        Value to be converted to ecdf

    Returns
    -------
    ndarray
    """
    num_bins = np.min([b, 500, len(arr)])
    counts, bin_edges = np.histogram(arr, bins=num_bins)
    # print(counts)
    cdf = np.cumsum(counts) / sum(counts)
    csi5 = bin_edges[1:]
    return np.interp(q, cdf, csi5)


def ghi_clim(clear_sky, csi):
    l0 = len(clear_sky)
    d0 = csi.ndim
    if d0 < 2:
        v0 = np.array([clear_sky, ] * len(csi)).transpose()
        v1 = np.array([csi, ] * l0)
        out0 = np.multiply(v0, v1)
    else:
        d0, d1 = np.shape(csi)
        csi_2d = np.tile(csi, (np.ceil(len(clear_sky) / d0).astype(int), 1))
        # clear_sky0 = data_test["clear_sky"].values
        out0 = clear_sky[:, np.newaxis] * csi_2d[range(len(clear_sky)), :]
    return out0


def clim_csi(csi, q, b=500, csi0=1.2, frac=0.95, za=None, za0=89, clear_sky=None):
    """
    Compute climatology from empirical distribution of clear sky index based on given quantile *q*. This function
    rely on the :meth:`ecdf0`

    Parameters
    ----------
    csi : numpy.ndarray
        Clear sky index
    q : numpy.ndarray
        Quantile value
    b : int
        Number of class for frequency construction. If not given, :math:`b=500`.
    csi0 : float
        Maximum allowed value of the clear sky indexes
    frac : float
        Maximum fraction of extraterrestrial irradiation to earth surface
    za : numpy.ndarray
        Values of the zenith angle, can be None
    za0 : float
        Zenith angle value for filtering data
    clear_sky : numpy.ndarray
        Clear sky data
    Returns
    -------
    numpy.ndarray
        2D numpy array of the probabilistic outcome
    """
    if clear_sky is not None:
        csi_lim = np.minimum(csi0, 1367 * frac / np.max(clear_sky))
    else:
        csi_lim = np.minimum(csi0, 1.24 * frac)
    if za is not None:
        csi2 = csi[np.where(za < za0)[0]]
    else:
        csi2 = np.copy(csi)
    id1 = np.where(csi2 > csi_lim)[0]
    csi2[id1] = csi_lim
    csi2 = csi2[np.isfinite(csi2)]
    # csi2 = flt_dta(csi, csi0=csi_lim, za0=za0)  # filtering data
    return ecdf0(csi2, q, b=b)


def clim_csi_hour(time_step, csi, q, csi0=1.2, frac=0.95, za=None, za0=89, clear_sky=None):
    """
    Compute climatology from empirical distribution of clear sky index based on given quantile *q* and time steps. This
    function rely on the :meth:`ecdf0`

    Parameters
    ----------
    time_step : pandas.DatetimeIndex
        Date for the computation of identical time steps
    csi : numpy.ndarray
        Clear sky index
    q : numpy.ndarray
        Quantile value
    csi0 : float
        Maximum allowed value of the clear sky indexes
    frac : float
        Maximum fraction of extraterrestrial irradiation to earth surface
    za : numpy.ndarray
        Values of the zenith angle, can be None
    za0 : float
        Zenith angle value for filtering data
    clear_sky : numpy.ndarray
        Clear sky data
    Returns
    -------
    numpy.ndarray
        2D numpy array of the probabilistic outcome
    """
    if clear_sky is not None:
        csi_lim = np.minimum(csi0, 1367 * frac / np.max(clear_sky))
    else:
        csi_lim = np.minimum(csi0, 1.24 * frac)
    st0 = second_compute(time_step)
    mp0 = np.unique(st0)
    cdf = np.zeros((len(mp0), len(q)), dtype=float)
    for u0 in range(len(mp0)):
        k0 = np.where(st0 == mp0[u0])[0]
        csi2 = csi[k0]
        id1 = np.where(csi2 > csi_lim)[0]
        csi2[id1] = csi_lim
        if za is not None:
            csi2 = csi2[np.where(za[k0] < za0)[0]]
        csi1 = csi2[np.isfinite(csi2)]
        if len(csi1) == 0:
            cdf[u0, :] = np.zeros(len(q))
        else:
            cdf[u0, :] = ecdf0(csi1, q, len(q))
    return cdf, mp0
