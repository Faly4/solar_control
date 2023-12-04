Probabilistic forecast from climatology
---------------------------------------
Import modules and data
^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    :linenos:

    import matplotlib as mpl
    import matplotlib.dates as md
    import numpy as np
    from pv_solar import clim_csi, clim_csi_hour, ghi_clim, iut_data
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    iut_data['Year'] = iut_data['time'].dt.strftime('%Y').astype(int)  # Data from Reunion University (2010 to 2011)
    data_train = iut_data.loc[iut_data['Year'] == 2010]  # split data

    za0 = 86  # Zenith angle filtering
    frac0 = 0.95  # Fraction extraterrestrial
    csi0 = 1.2  # Max value of csi
    q0 = np.linspace(start=0.05, stop=0.95, num=50)  # Define 50 quantile 0.05-0.95


Compute clear sky value
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    :linenos:

    csi_iut0 = clim_csi(data_train['csi'].values, q0, 500, 1.5, frac0, data_train['zenith'].values, za0,
        data_train['clear_sky'].values)
    csi_iut1, hour0 = clim_csi_hour(data_train['time'], data_train['csi'].values, q0, 1.5, frac0,
        data_train['zenith'].values, za0, data_train['clear_sky'].values)


Define function for plot's layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    :linenos:

    def plot_ghi(time, ghi, q, ylim0, legend0):
        d1 = len(q)
        d2 = int(len(q) / 2)  # Half of length of quantile
        cmap = plt.get_cmap('Blues', d2)  # Generate color map
        col0 = [mpl.colors.rgb2hex(cmap(u)) for u in range(cmap.N)]  # Generate color
        for u in range(d2):
            y2 = ghi[:, d1 - 1 - u]
            y1 = ghi[:, u]
            plt.fill_between(time, y2, y1, color=col0[u])
        plt.plot(time, ghi, color="darkblue", linewidth=0.12)
        plt.grid()
        plt.plot(time, data_test["GHI"], label='GHI', color="firebrick", linewidth=1.5)
        q1 = np.round(q[range(d2)] * 100)
        cbar = plt.colorbar(ScalarMappable(norm=BoundaryNorm(q1, ncolors=len(q1)), cmap=ListedColormap(col0)),
                            label='quantiles', pad=0.01)
        cbar.ax.tick_params(labelsize=7)
        plt.legend(loc="upper left")
        plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M'))  # Format datetick labels as desired
        plt.title(legend0)
        plt.xlabel('hour')
        plt.ylabel('W/m²')
        plt.ylim(0, ylim0)



Define range of date and compute probabilistic output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    :linenos:

    mask = (iut_data['time'] >= '2011-09-20') & (iut_data['time'] < '2011-09-23')
    data_test = iut_data.loc[mask]

    ghi0 = ghi_clim(data_test['clear_sky'].values, csi_iut0)  # Forecast based on full CSI value
    ghi1 = ghi_clim(data_test['clear_sky'].values, csi_iut1)  # Forecast with distribution generated for each time range index.


Plot probabilistic output
^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
    :linenos:

    time0 = data_test["time"].dt.tz_localize(None)  # Remove timezone
    ylim = np.maximum(np.max(ghi0), np.max(ghi0)) * 1.05 # Unify y max value

    fig, axs = plt.subplots(2)
    fig.suptitle('Probabilistic climatology')
    plt.subplot(2, 1, 1)
    plot_ghi(time0, ghi0, q0, ylim, "CSI")
    plt.tick_params(labelbottom = False)
    plt.xlabel("")
    plt.subplot(2, 1, 2)
    plot_ghi(time0, ghi1, q0, ylim, "CSI hour by hour")
    plt.show()