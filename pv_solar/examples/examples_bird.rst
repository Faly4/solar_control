Bird Clear Sky irradiation
---------------------------
.. code-block:: Python
        :linenos:
        :emphasize-lines: 29


        import numpy as np
        import pandas as pd
        import pytz
        from matplotlib import dates as md, pyplot as plt
        from pv_solar import bird_clear_sky

        """
        Geographic data
        """
        long = -105.1786  # Observer geographical longitude (°)
        lat = 39.742476  # Geocentric latitude (rad)
        pssr = 820  # Annual average local pressure (mbar)
        temp = 11  # Annual average local temperature (°C)
        oz = 0.3  # Amount of ozone in a vertical column from surface (cm)
        wtr = 1.5  # Amount of precipitable water in a vertical column from surface (cm)
        aod500 = 0.1  # Aerosol optical depth from surface in a vertical path at 0.5 μm wavelength
        aod380 = 0.15  # Aerosol optical depth from surface in a vertical path at 0.38 μm wavelength
        ba = 0.85  # Ratio of the forward-scattered irradiance to the total scattered irradiance due to aerosols
        albedo = 0.2  # Ground albedo

        """
        Timeseries data
        """
        time_step = pd.date_range(start=pd.to_datetime("10/17/2003 0:00:00").tz_localize("Etc/GMT-7"),
                                  end=pd.to_datetime("10/18/2003 23:00:00").tz_localize("Etc/GMT-7"), freq='15min')
        timezone = pytz.timezone("Etc/GMT-7")
        offset = -timezone.utcoffset(time_step).seconds / 3600  # Timezone from UTC

        ghi8bird = bird_clear_sky(time_step, offset, lat, long, pssr, oz, wtr, aod500, aod380, ba, albedo)

        """
        Plot output
        """
        time2 = time_step.tz_localize(None)  # remove time zone data to get local time display
        plt.plot(time2, ghi8bird['gh'])
        plt.plot(time2, ghi8bird['idnh'])
        plt.plot(time2, ghi8bird['diff_hz'])
        plt.gca().xaxis.set_major_formatter(md.DateFormatter('%H:%M'))  # Format datetick labels as desired
        plt.xlabel('hour')
        plt.ylabel('W/m²')
        plt.title("Bird's clear sky generated at frequency of 15 min")
        plt.legend(['Global', 'Direct', 'Diffuse'], loc='upper left')
        plt.grid()
        plt.show()