import numpy


def path0x(dp8out, states, init8sta):
    """
    Extract the optimal path from the output of the **Dynamic Programming solver** :func:`dmx0g` by setting the initial
    state.

    Parameters
    ----------
    dp8out : dict
        Dictionary with 2 keys from :func:`dmx0g` output:
            * key[0]: 2D numpy array of returns
            * key[1]: 2D numpy array of paths
    states : numpy.ndarray
        1D numpy array of the all possible state from the discretization.
    init8sta : float
        initial state of the decision variable

    Returns
    -------
    dict
           Dictionary with 2 keys:
            * key[0]: 1D numpy array of optimal states
            * key[1]: float of optimal return
    """
    path0 = dp8out["1"]
    return0 = dp8out["0"]
    print(numpy.shape(return0))
    l1, dx1 = numpy.shape(path0)
    path1 = numpy.zeros((l1, dx1), dtype=int)
    gx1 = numpy.flip(states)
    for um in range(l1):
        m1 = um
        for un in range(dx1):
            m1 = path0[m1, un]
            path1[um, un] = m1
    path1 = numpy.column_stack((list(range(l1)), path1))
    z1 = numpy.concatenate(numpy.where(gx1 == init8sta))[0]
    z2 = path1[z1, :]
    out1 = dict()
    out1["0"] = gx1[z2]
    out1["1"] = return0[z2[0], 1]
    return out1


def inv_eff(pwr8in, inv8rt8pwr):
    r"""
    Calculate the efficiencies (:math:`\eta`) of the inverter  from the input power and the rated
    power of the inverter. The inverter efficiency follow a quadratic equation (:math:`\eta = a.x^2+ b.x+c`)

    .. math::
            \eta = inv_{eff}(P_{out},P_{inv})

    Parameters
    ----------
    pwr8in : numpy.ndarray
        Power at the input of the inverter
    inv8rt8pwr : float
        Rated power of the used inverter


    Returns
    -------
    numpy.ndarray
        Inverter efficiency (:math:`\eta`)
    """
    in0 = numpy.abs(pwr8in / inv8rt8pwr)
    n1 = 1.0 - numpy.divide(0.0094 + 0.043 * in0 + 0.04 * numpy.power(in0, 2), in0, out=numpy.ones_like(in0),
                            where=in0 != 0)
    k0 = numpy.where(n1 <= 0)
    n1[k0] = 0
    return n1


def inv_in(pwr8out, inv8rt8pwr):
    r"""
    Calculates the power at the input of the inverter from the power at the output and the rated power of the inverter.

    .. math:: P_{in}= inv_{eff}^{-1}(P_{out},P_{inv}) \times P_{out}


    Parameters
    ----------
    pwr8out : numpy.ndarray
        Power at the output of the inverter
    inv8rt8pwr : float
        Rated power of the used inverter


    Returns
    -------
    numpy.ndarray
        Power input to the inverter (:math:`P_{in}`)
    """
    a1 = 0.04 / inv8rt8pwr
    b1 = 0.043 - 1
    c1 = 0.0094 * inv8rt8pwr + pwr8out
    del0 = numpy.multiply(a1, c1) * (-4) + numpy.power(b1, 2)
    x1 = (-b1 - numpy.sqrt(del0)) / (2 * a1)
    x2 = (-b1 + numpy.sqrt(del0)) / (2 * a1)
    x0 = numpy.minimum(x1, x2)
    k0 = numpy.where(pwr8out <= 0)
    x0[k0] = 0
    return x0


def rmse(observations, forecast, option=0):
    r"""
    Compute the Root Mean Square Error (RMSE) between observations (:math:`y_1`) and  forecasts (:math:`\hat{y}_i`).

    .. math::
        RMSE=\sqrt{\sum_{i=1}^n  \frac{(\hat{y}_i-y_i)^2}{n}}

    Parameters
    ----------
    observations : numpy.ndarray
    forecast : numpy.ndarray
    option : int
        If :math:`option=0` the RMSE is computed from non-zero observations (:math:`y_i \neq 0`)

    Returns
    -------
    float
        RMSE
    """
    a = numpy.concatenate(observations)
    b = numpy.concatenate(forecast)
    id1 = range(len(a))
    if option == 0:
        id1 = numpy.where(a > 0)[0]
    y0 = numpy.sqrt(sum(numpy.power(numpy.add(b[id1], - a[id1]), 2) / len(id1)))
    return y0


def dmx0g(states, dx0, cost_function, *args):
    r"""
    Backward Dynamic Programming solver from state variation. The solution is computed from last to first state.
    This code generates a mapping of all solutions by constructing 2 matrices: return and path. In the case where
    several solutions are found, the solution is chosen by minimizing the second criteria.

    For example for storage management, the cost function returns 2 keys: key[0]=cost, key[1]=network energy.

    If there are 3 values in key[0] with the same minimal cost, the solution chosen among the 3 will correspond to
    the one whose energy drawn from the network is the minimal.

    Parameters
    ----------
    states : numpy.ndarray
        1D numpy array of the all possible state from the discretization.
    dx0 : int
        Length of the window of optimization
    cost_function : function
        * The cost function must have at least 2 parameter, return a dictionary with 2 keys and be in this form:
            def cost(df0, m0, \*args):
                * df0: the variation of the state
                * m0: step, for example if the length of the window of optimization is 24 time steps.
                    * From time 1 :math:`\to` 2, :math:`m0=1`
                    * From time 23 :math:`\to` 24, :math:`m0=23`
        * The cost function must return a dictionary with 2 keys
            * key[0]: Cost of variation of the state
            * key[1]: Parameter for choosing the solution if multiple solution is found
    args : arg
        Additional or optional arguments for the **cost_function**


    Returns
    -------
    dict
        Dictionary with 2 keys as output:
            * key[0]: 2D numpy array of returns
            * key[1]: 2D numpy array of paths
    """
    lx0 = len(states)
    r2 = numpy.ones((lx0, dx0), dtype=float)  # first return table
    x2 = numpy.zeros((lx0, dx0), dtype=int)  # create indexes table
    m2 = numpy.ones((dx0, 1), dtype=int)
    m2[dx0 - 1] = 0
    for uk in range(dx0 - 1, -1, -1):
        for vk in range(lx0):
            dsoc0 = states[vk] - states
            c0x = cost_function(dsoc0, uk, *args)
            c0 = c0x["0"]
            c1 = numpy.concatenate(r2[:, uk + 1 * m2[uk] - 1]) + c0
            r2[vk, uk] = numpy.min(c1)  # affect min value in table
            id2 = numpy.concatenate(numpy.where(c1 == numpy.min(c1)))  # find min
            pew1 = c0x["1"][id2]
            k1 = numpy.concatenate(numpy.where(pew1 == numpy.min(pew1)))[0]  # find min
            x2[vk, uk] = id2[k1]
    out0 = dict()
    out0["0"] = r2
    out0["1"] = x2
    return out0


def state_init(states, init8sta):
    """
    Fixes numerical inaccuracies induced by the precision of the ECU when retrieving the initial state in the
    possible states.
    
    Parameters
    ----------
    states : numpy.ndarray
        1D numpy array of the all possible state from the discretization
    init8sta : float
        initial state of the decision variable

    Returns
    -------
    numpy.ndarray
        Correct states of discretization
    """
    g7 = numpy.abs(states - init8sta)
    u2 = numpy.where(g7 == numpy.min(g7))
    states[u2] = init8sta
    return states
