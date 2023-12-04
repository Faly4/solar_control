Examples
=============

Example: Compute the efficiency of an inverter rated at 2500W connected to a PV module of 1000Wc
************************************************************************************************
.. code-block:: python
    :linenos:
    :emphasize-lines: 7

    import numpy as np
    from matplotlib import pyplot as plt
    from pdp import inv_eff

    inv4rt4pwr = 2500  # Inverter rated power
    pwr4in = np.linspace(0, 1000, 50)  # Module power output
    η = inv_eff(pwr4in, inv4rt4pwr)  # Inerter efficiency
    plt.plot(pwr4in, η)
    plt.xlabel('Input power [W]')
    plt.title('Inverter efficiency')
    plt.ylabel('η')
    plt.grid()
    plt.show()


Example: Compute the input of an inverter rated at 2500W from its output
************************************************************************
.. code-block:: python
    :linenos:
    :emphasize-lines: 7

    import numpy as np
    from matplotlib import pyplot as plt
    from pdp import inv_in

    inv4rt4pwr = 2500 # Inverter rated power
    pwr4out = np.linspace(0,1000,50)  # Inverter power output
    pwr4in = inv_in(pwr4out, inv4rt4pwr)  # Inverter power input
    print('Power input', pwr4in)
    η = np.divide(pwr4out, pwr4in, out=np.zeros_like(pwr4out), where=pwr4in!=0)
    plt.plot(pwr4in, η)
    plt.xlabel('Input power [W]')
    plt.title('Inverter efficiency')
    plt.ylabel('η')
    plt.grid()
    plt.show()

