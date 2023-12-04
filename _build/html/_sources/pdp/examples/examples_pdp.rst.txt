Manage storage with PV from probabilistic input
-----------------------------------------------
Import function
***************
.. code-block:: python
    :linenos:

    import numpy as np
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from pdp import dmx0g, inv_eff, path0x, inv_in, state_init, iut_load, iut_pv

Define storage characteristics
******************************
.. code-block:: python
    :linenos:

    ex1 = 13.5  # max usable energy kWh
    mx11 = 0.9  # max charge rate
    mx22 = 0.9  # max discharge rate
    ef1 = np.sqrt(0.95) # storage efficiency
    z1 = 0.8312517e-4 # ageing coefficient
    soh1 = 0.8 # minimal storage health state
    invr7 = 15  # ESS inverter kW

Define PV array characteristics
*******************************
.. code-block:: python
    :linenos:

    pmpp = 6 # 6 kW
    mod1 = 0.3 # module 300Wc
    smod1 = 1.7522 # surface module1
    effmod1 = 0.1546 # module PV efficiency
    invr8 = 10 # PV inverter kW


Define simulation constraints and variables
*******************************************
.. code-block:: python
    :linenos:

    dx02 = 24  # number of hours of simulation
    dtx1 = 1 # timestep of integration
    pmax1 = 1  # max power from the grid
    soc0 = 0.50  # initial SOC
    gx02 = np.arange(0.1, 1.01, 0.01)  # discretization of the storage
    gx02 = state_init(gx02, soc0) # initialization of the storage


Define cost
***********
.. code-block:: python
    :linenos:

    ess1inv = 5988.87 # ESS investment cost
    c1grid = np.ones((dx02, 1), dtype=float) * 25  # grid cost import
    c2grid = np.ones((dx02, 1), dtype=float) * 25  # grid cost export
    c3grid = np.ones((dx02, 1), dtype=float) * 35  # penality cost

Define cost function
********************
.. code-block:: python
    :linenos:

    def cost3dp(dsoc0, up, dtx0, load2, pv2, sx2, pmax0, ex0, mx0, mx1, ef0, z0, ess0inv,
    soh0, inv0, c1grid, c2grid, c3grid):
        w1 = np.greater(dsoc0, mx0, out=np.ones_like(dsoc0))  # charge
        w2 = np.greater(-mx1, dsoc0, out=np.ones_like(dsoc0))  # discharge
        w3 = (w1 + w2) * 1e10
        m1 = np.greater(dsoc0, 0, out=np.ones_like(dsoc0))  # charging
        m2 = np.abs(1 - m1)  # discharge
        m3 = (m2 + m1) / ef0
        p1bess = -dsoc0 * ex0 / dtx0 * m3
        netha1 = inv_eff(np.abs(p1bess), inv0)
        netha2 = inv_in(np.abs(p1bess), inv0)
        netha2 = inv_eff(netha2, inv0)
        n6 = np.equal(netha2, 0, out=np.ones_like(netha2))
        netha2 = np.add(n6, netha2)
        m4 = np.multiply(m2, netha1) + np.divide(m1, netha2)
        p0bess = np.multiply(p1bess, m4)  # BESS power
        n2 = len(p0bess)
        doh1 = np.multiply(dsoc0, m2) * z0
        c0ess = -ess0inv * doh1 / (1 - soh0) + w3
        # <!-- compute_grid_cost -->
        pv0 = np.tile(pv2[up, :], [n2, 1])
        l2 = len(pv2[up, :])
        # print("------")
        balan4 = -pv0[up, :] + load2[up] # energy balance
        p1bess = np.tile(np.reshape(p0bess, (n2, 1)), [1, l2])
        pw0 = balan4 - p1bess  # Grid power [pw0 > 0 :importation]
        c3gr = np.multiply(pw0, np.greater(0, pw0, out=np.ones_like(pw0))) * c2grid[up]  # export to the grid
        pw2 = pw0 - pmax0  # excess of energy
        pw2 = np.multiply(pw2, np.greater(pw2, 0, out=np.ones_like(pw0)))
        c2gr = pw2 * c3grid[up]  # penalty cost
        pw1 = pw0 - pw2  # import from grid
        c1gr = np.multiply(pw1, np.greater(pw1, 0, out=np.ones_like(pw1))) * c1grid[up]
        # TODO  Compute grid cost
        c0gr = c1gr + c2gr + c3gr
        n3 = (np.mean(pv2) == 0) * 1
        sx3 = np.ones((len(sx2), 1), dtype=float)
        sx2 = np.add(np.reshape(sx2, (len(sx2), 1)) * (1 - n3), sx3 * n3)
        # ------------------------- compute_math_expectation_depending_on_pv
        cpb1 = np.matmul(c0gr, np.flip(sx2))
        out1 = dict()
        out1["0"] = np.add(c0ess, np.concatenate(cpb1))  # cost (expectation)
        out1["1"] = np.concatenate(np.matmul(pw0, sx3) / len(sx3))  # energy from grid
        return out1




Simulation
**********

.. code-block:: python
    :linenos:
    :emphasize-lines: 9-10

    load2 = iut_load # Load

    pv0 = iut_pv # probabilistic forecast of GHI
    sx1 = np.linspace(0.05, 0.95, num=50)  # probability of occurrence
    pv1 = pmpp / mod1 * smod1 * effmod1 * pv0 / 1000 / 1.05 / 1.15  # PV power
    n0 = inv_eff(pv1, invr8) # PV inverter efficiency
    pv2 = np.multiply(pv1, n0) # PV inverter output

    p13 = dmx0g(gx02, dx02, cost3dp, dtx1, load2, pv2, sx1, pmax1, ex1, mx11, mx22, ef1,
    z1, ess1inv, soh1, invr7, c1grid, c2grid, c3grid) # compute solution
    s0 = path0x(p13, gx02, soc0) # get optimal path
    print(s0) # print solution


Plot result
***********

.. code-block:: python
    :linenos:

    x = range(1, dx02 + 1)
    x1 = range(dx02)
    x2 = range(dx02 + 1)

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    im1 = ax1.plot(x, load2[x1])
    ax1.set(ylabel="Load (kW)")
    ax1.grid(axis='x', color='0.95')

    im2 = ax2.plot(x, pv2[x1])
    ax2.set(ylabel="PV (kW)")
    ax2.grid(axis='x', color='0.95')

    im3 = ax3.plot(x2, s0["0"])
    ax3.set(ylim=(0, np.max(gx02)))
    ax3.set(ylabel="SOC")
    ax3.grid(axis='x', color='0.95')
    plt.show()
