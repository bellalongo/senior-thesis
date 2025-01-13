import astropy.units as u
import lightkurve as lk


class LightcurveData(object):
    def __init__(self, tic_name, cadence):
        self.tic_name = tic_name
        self.cadence = cadence

        # Get lightcurve data
        self.lightcurve, self.name = self.get_lightcurve()

        # Lightcurve data
        self.time = self.lightcurve.time.value
        self.flux = self.lightcurve.flux.value
        self.flux_err = self.lightcurve.flux_err.value

        # Get periodogram
        self.periodogram = self.get_periodogram()

        # Get period at max power
        self.period_at_max_power = self.get_period_at_max_power()


    def append_lightcurves(self, result, result_exposures):
        """
            Appends lightcurves of the wanted cadence together
            Parameters: 
                        result: Lightkurve query result
                        result_exposures: Lightkurve query result exposures
            Returns:
                        combined_lightcurve: appended lightcurves 
                        (None if none of the query results are of the desired cadence)
        """
        all_lightcurves = []

        # Get the data whose exposure is the desired cadence
        for i, exposure in enumerate(result_exposures):
            # Check to see if exposure matches cadence 
            if exposure.value == self.cadence:
                lightcurve = result[i].download().remove_nans().remove_outliers().normalize() - 1
                all_lightcurves.append(lightcurve)
        
        # Check if there are lightcurves
        if all_lightcurves:
            combined_lightcurve = all_lightcurves[0]
            # Iterate through all of the lightcurves
            for lc in all_lightcurves[1:]:
                combined_lightcurve = combined_lightcurve.append(lc)
        else:
            return None
        
        return combined_lightcurve  


    def get_lightcurve(self):
        """
            Creates a lightcurve from the current catalog row's TIC number, as well as saves the TIC number,
            i magnitude, and literature period
            Parameters: 
                        None
            Returns:
                        lightcurve: current catalog row's lightcurve
                        name: current catalog row's TIC number
                        imag: current catalog row's imag
                        literature_period: current catalog row's literature period (0 if none)
        """
        # Initialize a variable for catching errors
        error = False

        # Pull data for that star
        try:
            result = lk.search_lightcurve(self.tic_name, mission='TESS')
            result_exposures = result.exptime
        except Exception as e:
            print(f"Error for {self.tic_name}: {e} \n")
            error = True

        lightcurve = self.append_lightcurves(result, result_exposures)
        
        if not lightcurve:
            error = True  # check if there was a result with the cadence needed

        # Star data
        name = 'TIC ' + str(lightcurve.meta['TICID']) if lightcurve else None

        if not error:
            return lightcurve, name
        else:
            return None, None


    def get_periodogram(self):
        """
            Creates a periodogram from the lightcurve with the minimum period being 2 * cadence, and the maximum 
            period being 14 days
            Parameters: 
                        None
            Returns:
                        periodogram: lightcurve's periodogram
        """
        # Convert lightcurve to periodogram
        periodogram = self.lightcurve.to_periodogram(oversample_factor=10, 
                                                     minimum_period=(2 * self.cadence * u.second).to(u.day).value, 
                                                     maximum_period=14)
        return periodogram


    def get_period_at_max_power(self):
        """
            Finds the period at max power from the lightcurve's periodogram
            Parameters: 
                        None
            Returns:
                        period_at_max_power: lightcurve's period at max power
        """
        period_at_max_power = self.periodogram.period_at_max_power.value

        return period_at_max_power