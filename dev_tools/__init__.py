import numpy as np
import pandas as pd


def div_zeros(a, b):
    """
    Return 0 if division is by zeros

    Parameters
    ----------
    a : numpy.ndarray
    b : numpy.ndarray
        Array with zeros value

    Returns
    -------
    numpy.ndarray
    """
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def simple_da(time_step, offset):
    r"""
    Convert Python date time to separate value

    Parameters
    ----------
    time_step : tuple
        List of date with Python datetime format
    offset : float
        Decimal hour from UTC time

    Returns
    -------
    tuple
        * day (numpy.ndarray) - Days from the dates
        * hour (numpy.ndarray) - Hours from the dates
        * year (numpy.ndarray) - Years from the dates
        * month (numpy.ndarray) - Months from the dates
    """
    da = np.array(pd.to_datetime(time_step).day)
    d1 = np.array(pd.to_datetime(time_step).hour, dtype=float)
    d2 = np.array(pd.to_datetime(time_step).minute, dtype=float)
    d3 = np.array(pd.to_datetime(time_step).second, dtype=float)
    hr = d1 + (d2 / 60) + d3 / 3600 + offset
    yr = np.array(pd.to_datetime(time_step).year, dtype=float)
    mon = np.array(pd.to_datetime(time_step).month, dtype=float)
    return da, hr, yr, mon


def poly_ord_3rd(a, b, c, d, x):
    r"""
    Compute value of polynomial of 3rd order

    .. math:: y =a x^3 + b x^2 + c x + d


    Parameters
    ----------
    a : float
    b : float
    c : float
    d : float
    x : numpy.ndarray


    Returns
    -------
    numpy.ndarray
    """
    return ((a * x + b) * x + c) * x + d


def lim_ang(angle, period):
    r"""
    Set angle value in range :math:`[0, period]`

    Parameters
    ----------
    angle : numpy.ndarray
        Input angle
    period : float
        Period of angle: in degree = 360°, in radian = :math:`2\pi`


    Returns
    -------
    numpy.ndarray
        Angle within the range [0, period]
    """
    return angle - np.floor(angle / period) * period


def yr_float(time_step):
    """
    Compute decimal year from Python datetime

    Parameters
    ----------
    time_step : tuple
        List of date with Python datetime format

    Returns
    -------
    numpy.ndarray
        Decimal year
    """
    d1 = time_step.tz_localize(None)
    d0 = np.array(d1.year)
    df2 = pd.DataFrame({'year': d0, 'month': 1, 'day': 1})
    df3 = pd.DataFrame({'year': d0 + 1, 'month': 1, 'day': 1})
    k2 = pd.to_datetime(df2)
    k3 = pd.to_datetime(df3)
    t2 = k2.astype('int64') // 1e9
    t3 = k3.astype('int64') // 1e9
    t0 = t3 - t2
    df = pd.DataFrame({'date': d1})
    df['date'] = pd.to_datetime(df['date'])
    f5 = df['date'].astype('int64') // 1e9
    t1 = f5 - t2
    return np.array(d0 + t1 / t0)


def second_compute(time_step):
    h0 = time_step.dt.strftime('%H').astype(int)
    m0 = time_step.dt.strftime('%M').astype(int)
    s0 = time_step.dt.strftime('%S').astype(int)
    st0 = np.round(np.array(h0 * 3600 + m0 * 60 + s0))
    return st0


def flt_dta(csi, za=None, csi0=1.5, za0=89):
    """
    Return valid value of the csi

    Parameters
    ----------
    csi : numpy.ndarray
        Value of clear sky index (csi), can contains *inf* or *nan*.
    za : numpy.ndarray
        Value of the zenith angle corresponding to the *csi*. **za** is an optional parameter.
    csi0 : float
        Maximum value of clear sky index. If not given, :math:`csi_0=1.5`.
    za0 : float
        csi filtering zenith angle. If not given, :math:`89°`

    Returns
    -------
    numpy.ndarray
        Valid value of **csi**

    """
    if za is not None:
        csi2 = csi[np.where(za < za0)[0]]
    else:
        csi2 = np.copy(csi)
    id1 = np.where(csi2 > csi0)[0]
    csi2[id1] = csi0
    csi2 = csi2[np.isfinite(csi2)]
    return csi2
