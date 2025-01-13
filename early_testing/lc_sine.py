import astropy.units as u
from astropy.modeling.models import Gaussian1D
from astropy.modeling import fitting
from lightkurve import LightCurve
import lmfit 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
import seaborn as sns


class LCSine(object):
    def __init__(self, lightcurve_data):
        self.lightcurve_data = lightcurve_data

        # Fit a sine wave to the lightcurve
        self.sine_fit = self.fit_sine_wave(self.lightcurve_data.time, self.lightcurve_data.flux)

        # # Fold and bin lightcurve
        # self.binned_lightcurve = self.fold_lightcurve()

        # # Fold and bin sine fit
        # self.binned_sine, self.sine_period = self.fold_sine_wave(self.lightcurve_data.time, self.sine_fit.params['frequency'].value, self.sine_fit.best_fit)

        # # Calculate time points of the sine wave
        # self.time_points = np.arange(min(self.lightcurve_data.time), max(self.lightcurve_data.time), self.sine_period)

        # # Calculate plot xmin and xmax
        # self.xmin = min(self.lightcurve_data.time) + 1 + self.lightcurve_data.period_at_max_power
        # self.xmax = min(self.lightcurve_data.time) + 1 + 4 * self.lightcurve_data.period_at_max_power


    def sine_wave(self, x, amplitude, frequency, phase):
        """
            Creates a sine wave based off of given parameters
            Name:       sine_wave()
            Parameters: 
                        x: data points
                        amplitude: desired amplitude
                        frequency: desired frequency
                        phase: desired phase
            Returns:
                        a sine wave
        """
        return amplitude * np.sin((2 * np.pi * frequency * x) + phase)


    def find_bin_value(self, lightcurve, num_bins):
        """
            Calculates the best bin value based off of the duration of the lightcurve
            Name:       find_bin_value()
            Parameters:
                        num_bins: desired number of bins
            Returns:
                        bin_value: number of minutes for each bin
        """
        total_points = len(lightcurve.time.value)
        total_duration_mins = ((lightcurve.time.value[total_points - 1] - lightcurve.time.value[0]) * u.day).to(u.minute)
        bin_value = (total_duration_mins / num_bins).value

        return bin_value
    
    def fit_sine_wave(self, time, flux):
        """
            Fits a sine wave to a lightcurve using the lmfit package
            Name:       fit_sine_wave()
            Parameters:
                        time: time data for the lightcurve
                        flux: flux data for the lightcurve
            Returns:
                        result: fitted sine wave
        """
        # Make an lmfit object and fit it
        model = lmfit.Model(self.sine_wave)
        params = model.make_params(amplitude = self.lightcurve_data.periodogram.max_power, 
                                   frequency = 1 / self.lightcurve_data.period_at_max_power, 
                                   phase = 0.0)
        result = model.fit(flux, params, x = time)

        return result

    # def fit_sine_wave(self, time, flux):
    #     """
    #         Fits a sine wave to a lightcurve using the lmfit package
    #         Name:       fit_sine_wave()
    #         Parameters:
    #                     time: time data for the lightcurve
    #                     flux: flux data for the lightcurve
    #         Returns:
    #                     result: fitted sine wave
    #     """
    #     # Fit the first period
    #     phase1_mask = (time > min(time)) & (time < min(time) + self.lightcurve_data.period_at_max_power)
    #     model = lmfit.Model(self.sine_wave)
    #     params = model.make_params(amplitude = self.lightcurve_data.periodogram.max_power, 
    #                                frequency = 1 / self.lightcurve_data.period_at_max_power, 
    #                                phase = 0.0)
    #     phase1_result = model.fit(flux[phase1_mask], params, x = time[phase1_mask])

    #     # Fit the second period
    #     phase2_mask = (time > min(time) + self.lightcurve_data.period_at_max_power) & (time < min(self.lightcurve_data.time) + 2*self.lightcurve_data.period_at_max_power)
    #     model = lmfit.Model(self.sine_wave)
    #     params = model.make_params(amplitude = self.lightcurve_data.periodogram.max_power, 
    #                                frequency = 1 / self.lightcurve_data.period_at_max_power, 
    #                                phase = 0.0)
    #     phase2_result = model.fit(flux[phase2_mask], params, x = time[phase2_mask])

    #     result = np.concatenate((phase1_result.best_fit, phase2_result.best_fit))

    #     return result
    

    def double_sine_wave(self, x, amp1, freq1, phase1, amp2, freq2, phase2):
        return (amp1 * np.sin((2 * np.pi * freq1 * x) + phase1) + 
                amp2 * np.sin((2 * np.pi * freq2 * x) + phase2))
    

    def fit_double_sine_wave(self, time, flux):
        """
        Fits two sine waves to a lightcurve using the lmfit package
        """
        # Create the double sine wave model
        model = lmfit.Model(self.double_sine_wave)

        # Make initial guesses for the parameters
        params = model.make_params(
            amp1 = self.lightcurve_data.periodogram.max_power,
            freq1 = 1 / self.lightcurve_data.period_at_max_power,
            phase1 = 0.0,
            amp2 = self.lightcurve_data.periodogram.max_power / 2,  # Guess for second amplitude
            freq2 = 2 / self.lightcurve_data.period_at_max_power,  # Guess for second frequency
            phase2 = 0.0
        )

        # Set some bounds to help the fit
        params['amp1'].set(min=0)
        params['amp2'].set(min=0)
        params['freq1'].set(min=0)
        params['freq2'].set(min=0)

        # Perform the fit
        result = model.fit(flux, params, x=time)

        return result



    # def fold_lightcurve(self, num_folds=1):
    #     """
    #         Folds the lightcurve on the period at max power, and bins it into 50 bins
    #         Name:       fold_lightcurve()
    #         Parameters:
    #                     num_folds: number of folds wanted to do on the period (default = 1, just folding on the period at max power)
    #         Returns:
    #                     binned_lightcurve: folded and binned lightcurve
    #     """
    #     # Fold lightcurve
    #     folded_lightcurve = self.lightcurve_data.lightcurve.fold(period=num_folds * self.lightcurve_data.period_at_max_power)

    #     # Calculate bin value
    #     bin_value = self.find_bin_value(folded_lightcurve, num_folds * 100)

    #     # Bin the folded lightcurve
    #     binned_lightcurve = folded_lightcurve.bin(bin_value * u.min)

    #     return binned_lightcurve


    # def fold_sine_wave(self, x, frequency, sine_wave, num_folds=1):
    #     """
    #         Folds the fitted sine wave on its period and bins it into 50 bins
    #         Name:       find_bin_value()
    #         Parameters:
    #                     x: time data of the lightcurve
    #                     frequency: frequency of the fitted sine wave
    #                     sine_wave: fitted sine wave (best_fit)
    #         Returns:
    #                     binned_sine: folded and binned sine wave
    #                     sine_period: period of the fitted sine wave
    #     """
    #     # Calculate the time points for the period lines
    #     sine_period = 1 / frequency

    #     # Make the sine wave into a lightcurve 
    #     sine_lightcurve = LightCurve(time=x, flux=sine_wave)

    #     # Fold sine wave
    #     folded_sine = sine_lightcurve.fold(period=num_folds * self.lightcurve_data.period_at_max_power)

    #     # Calculate bin value
    #     bin_value = self.find_bin_value(folded_sine, num_folds * 50)

    #     binned_sine = folded_sine.bin(bin_value * u.min)

    #     return binned_sine, sine_period


    def plot_periodogram(self, axis):
        """
            Plots the lightcurve's periodogram on a given axis, as well as the period at max power, and the literature
            period, if any
            Name:       plot_periodogram()
            Parameters:
                        axis: axis to be plotted on 
            Returns:
                        None
        """
        # Plot title
        axis.set_title('Periodogram', fontsize=12)
        axis.set_xlabel(r'$P_{\text{orb}}$ (days)', fontsize=10)
        axis.set_ylabel('Power', fontsize=10)
        axis.plot(self.lightcurve_data.periodogram.period, self.lightcurve_data.periodogram.power, color='#9AADD0')
        axis.axvline(x=self.lightcurve_data.period_at_max_power, color="#101935", ls=(0, (4, 5)), lw=2, 
                     label=fr'$P_{{\text{{orb, max power}}}}={np.round(self.lightcurve_data.period_at_max_power, 3)}$ days') 

        # Change scale to be log
        axis.set_xscale('log')

        # Add legend
        axis.legend(loc='upper left')


    # def plot_binned_lightcurve(self, axis):
    #     """
    #         Plots the binned lightcurve and the binned sine wave on a given axis 
    #         Name:       plot_binned_lightcurve()
    #         Parameters:
    #                     axis: axis to be plotted on 
    #                     num_folds: number of folds wanted to fold the period on (default = 1)
    #         Returns:
    #                     None
    #     """
    #     # Plot title
    #     axis.set_title(r'Lightcurve Folded on $P_{\text{orb, max power}}$', fontsize=12)
    #     axis.set_xlabel('Phase', fontsize=10)
    #     axis.set_ylabel('Normalized Flux', fontsize=10)

    #     # Plot the binned lightcurve
    #     axis.vlines(self.binned_lightcurve.phase.value, 
    #                 self.binned_lightcurve.flux - self.binned_lightcurve.flux_err, 
    #                 self.binned_lightcurve.flux + self.binned_lightcurve.flux_err, color='#9AADD0', lw=2)
        
    #     # Plot the binned sine fit 
    #     axis.plot(self.binned_sine.phase.value, self.binned_sine.flux.value, color='#101935', label='Folded Sine Wave')

    #     # Add legend 
    #     axis.legend(loc='upper right')


    def plot_lightcurve_and_sine(self, axis):
        """
            Plots the lightcurve and the sine wave, as well as the period of the sine wave
            Name:       plot_lightcurve_and_sine()
            Parameters:
                        axis: axis to be plotted on 
            Returns:
                        None
        """
        # Plot title 
        axis.set_title('Lightcurve', fontsize=12)
        axis.set_xlabel('Time (days)', fontsize=10)
        axis.set_ylabel('Normalized Flux', fontsize=10)

        # Plot lightcurve
        axis.vlines(self.lightcurve_data.lightcurve.time.value, 
                    self.lightcurve_data.lightcurve.flux - self.lightcurve_data.lightcurve.flux_err, 
                    self.lightcurve_data.lightcurve.flux + self.lightcurve_data.lightcurve.flux_err, color='#9AADD0')
        
        # Add vertical lines at each period interval of the sine wave
        for tp in self.time_points:
            axis.axvline(x = tp, color = '#4A5D96', ls = (0, (4, 5)), lw = 2, 
                         label = fr'$P_{{\text{{orb, sine}}}} = {np.round(self.sine_period, 3)}$ days' if tp == self.time_points[0] else "")
        
        # Plot sine wave
        axis.plot(self.lightcurve_data.time, self.sine_fit.best_fit, color='#101935', label='Fitted Sine Wave')

        # Set xlim and plot legend 
        axis.set_xlim(self.xmin, self.xmax)
        axis.legend(loc='upper right')


    def plot_lightcurve_with_fit(self, axis):
        """
        Plots the lightcurve along with the combined fitted sine waves.
        Name:       plot_lightcurve_with_fit()
        Parameters:
                    axis: axis to be plotted on 
        Returns:
                    None
        """
        # Plot title 
        axis.set_title('Lightcurve with Fitted Sine Waves', fontsize=12)
        axis.set_xlabel('Time (days)', fontsize=10)
        axis.set_ylabel('Normalized Flux', fontsize=10)

        # Plot lightcurve with error bars
        axis.vlines(self.lightcurve_data.time.value, 
                    self.lightcurve_data.flux - self.lightcurve_data.flux_err, 
                    self.lightcurve_data.flux + self.lightcurve_data.flux_err, color='#9AADD0', label='Light Curve')

        # Fit the sine waves
        combined_fit = self.fit_sine_wave(self.lightcurve_data.time, self.lightcurve_data.flux)

        # Create a time array for the entire lightcurve for plotting the fitted sine wave
        time_full = np.linspace(min(self.lightcurve_data.time), max(self.lightcurve_data.time), num=len(combined_fit))

        # Plot the combined fitted sine wave
        axis.plot(time_full, combined_fit, color='#101935', label='Fitted Sine Waves', linewidth=2)

        # Set xlim and plot legend 
        axis.set_xlim(self.xmin, self.xmax)
        axis.legend(loc='upper right')



    def plot_residuals(self, axis):
        """
            Plots the residuals of the lightcurve, which is the flux subtracted by the sine fit
            Name:       plot_residuals()
            Parameters:
                        axis: axis to be plotted on 
            Returns:
                        None
        """
        # Calculate residuals (lightcurve flux - sine wave flux)
        residuals = self.lightcurve_data.flux - self.sine_fit.best_fit

        # Plot title
        axis.set_title('Flux - Fitted Sine Wave', fontsize=12)
        axis.set_xlabel('Time (days)', fontsize=10)
        axis.set_ylabel('Normalized Flux', fontsize=10)

        # Plot the residuals
        axis.plot(self.lightcurve_data.time, residuals, color='#9AADD0')

        # Set xlim (no legend needed)
        axis.set_xlim(self.xmin, self.xmax)


    # def is_real_period_plot(self):
    #     """
    #         Present a plot of the periodogram, binned lightcurve, lightcurve, and residuals, which are then used to
    #         determine if the period at max power is real or not
    #         Name:       is_real_period_plot()
    #         Parameters:
    #                     None
    #         Returns:
    #                     None       
    #     """
    #     # Plot basics
    #     sns.set_style("whitegrid")
    #     # sns.set_theme(rc={'axes.facecolor': '#F8F5F2'})
    #     fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    #     plt.subplots_adjust(hspace=0.35)
    #     plt.suptitle(fr"Press 'y' if the period is real, 'n' if not.", fontweight='bold')
    #     fig.text(0.5, 0.928, r'Note: $P_{\text{orb, max power}}$ is under 5 sigma, so might NOT be real', ha='center', fontsize=12, style='italic')
    #     fig.text(0.5, 0.05, f'{self.lightcurve_data.name}', ha='center', fontsize=16, fontweight='bold')
    #     cid = fig.canvas.mpl_connect('key_press_event', lambda event: self.on_key(event))

    #     # Plot the periodogram
    #     self.plot_periodogram(axs[0, 0])  # see if can do this

    #     # Plot the binned lightcurve
    #     self.plot_binned_lightcurve(axs[1, 0])

    #     # Plot the lightcurve with the sine fit
    #     self.plot_lightcurve_and_sine(axs[0, 1])
        
    #     # Plot residuals
    #     self.plot_residuals(axs[1, 1])


    def on_key(self, event):
        """
            Event function that determines if a key was clicked
            Name:       on_key()
            Parameters: 
                        event: key press event
            Returns:
                        None
        """
        y_n_keys = {'y', 'n'}

        if event.key not in y_n_keys:
            print("Invalid key input, select 'y' or 'n'")
        else:
            if event.key == 'n':
                print('Period is not real, loading next plot ... \n')
            else:
                self.is_real_period = True

            plt.close()