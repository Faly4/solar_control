Earth Δt for 2003
------------------
.. code-block:: Python
        :linenos:
        :emphasize-lines: 9

        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        from pv_solar import earth_dlt_t

        time_step = pd.date_range(start=pd.to_datetime("10/17/2003 12:30:30").tz_localize("Etc/GMT-7"),
        end=pd.to_datetime("10/17/2003 18:30:30").tz_localize("Etc/GMT-7"), freq='H')
        print(time_step)
        delta0 = earth_dlt_t(time_step, fig=True)
        print(delta0)
        plt.show()


Solar position
---------------
.. code-block:: Python
        :linenos:
        :emphasize-lines: 33

        import numpy as np
        import pandas as pd
        import pytz
        from pv_solar import almanac
        from matplotlib import dates as md, pyplot as plt

        """
        Geographic data
        """
        long = -105.1786  # Observer geographical longitude (°)
        lat = np.deg2rad(39.742476)  # Geocentric latitude (rad)
        el = 1830.14  # Elevation (m)
        pssr = 820  # Annual average local pressure (mbar)
        temp = 11  # Annual average local temperature (°C)

        """
        PV mounting data
        """
        γ = np.deg2rad(-10)  # Surface azimuth rotation angle (°C)
        ω = np.deg2rad(30)  # Slope of the surface measured from the horizontal plane (rad)

        """
        Date list
        """
        time_step = pd.date_range(start=pd.to_datetime("10/17/2003 00:00:00").tz_localize("Etc/GMT-7"),
                          end=pd.to_datetime("10/19/2003 23:00:00").tz_localize("Etc/GMT-7"), freq='15min')
        timezone = pytz.timezone("Etc/GMT-7")
        offset = timezone.utcoffset(time_step).seconds / 3600

        """
        Compute value
        """
        position0 = almanac(el, offset, pssr, temp, time_step, γ, long, lat, ω)
        print(position0.head())

        time0 = time_step.tz_localize(None)  # Remove timezone
        position1 = position0[['Topo_z_ang', 'Topo_az_ang', 'Incid_angl']]
        position1.set_index(time0, inplace=True)

        axes = position1.plot(subplots=True, grid=True, title='Solar Position', xlabel='Time', legend=False)
        axes[0].set_ylabel('Zenith [°]')
        axes[1].set_ylabel('Azimuth [°]')
        axes[2].set_ylabel('Incidence [°]')
        plt.tight_layout()
        plt.show()
