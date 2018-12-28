#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from processing import Processing
from scipy import optimize


class LifeTime():
    def __init__(self, file_arr, window_length, n_average):
        self.file_arr = file_arr
        self.time_min = ""
        self.time_max = ""
        self.freq_min = ""
        self.freq_max = ""
        self.window_length = window_length
        self.n_average = n_average
        #self.plot_data(only_waterfall=True)

    def plot_data(self, padding_ratio=2, estimator='p', only_waterfall=False, **kwargs):
        '''
        the input parameters are from the function of time_average_1d from the class Processing
        window_length:  length of the tapering window
        padding_ratio:  >=1, ratio of the full frame length after zero padding to note that the final frame length will be rounded up to the next power of base 2 any illegal values disable zero padding
        n_average:      number of FFTs for one average
        estimator:      base estimator on which the time-averaging is applied, to be chosen from ['p', 'm', 'a'], which stands for periodogram, multitaper, and adaptive multitaper, respectively
        **kwargs:       some additional keyword arguments to be passed to the selected base estimator. (more details can be seen from time_average_2d)
        '''
        bud_3 = Processing(self.file_arr)
        cen_freq = bud_3.center_frequency*10**(-6) # MHz
        n_offset = int(60 * bud_3.sampling_rate)
        frequencies, times, spectrogram, n_dof = bud_3.time_average_2d(window_length=self.window_length, n_offset=n_offset, n_frame=300, padding_ratio=padding_ratio, n_average=self.n_average, estimator=estimator, **kwargs)

        plt.close("all")
        fig, (ax0, ax1) = plt.subplots(ncols=2)

        color_style = "viridis"
        pcm = ax0.pcolormesh(frequencies, times, spectrogram, cmap=color_style)
        #pcm = ax[0].pcolormesh(frequencies, times, spectrogram, cmap=color_style)

        #cax = fig.colorbar(pcm, ax=ax[0])
        #cax.set_label("power spectral density [arb. unit]")
        #ax[0].set_xlabel("frequency - {:g} MHz [kHz]".format(cen_freq))
        #ax[0].set_ylabel("times [s]")
        #ax[0].set_title(self.file_arr)
        #ax[0].set_ylim([times[0],times[-1]])
        cax = fig.colorbar(pcm, ax=ax0)
        cax.set_label("power spectral density [arb. unit]")
        ax0.set_xlabel("frequency - {:g} MHz [kHz]".format(cen_freq))
        ax0.set_ylabel("times [s]")
        ax0.set_title(self.file_arr)
        ax0.set_ylim([times[0],times[-1]])
        
        times_average = (times[:-1] + times[1:])/2

        if only_waterfall:
            if self.freq_min == "":
                ax0.set_xlim([frequencies[0],frequencies[-1]])
                plt.show()
                return
            else:
                arg_min = np.searchsorted(frequencies, self.freq_min, side="left")
                arg_max = np.searchsorted(frequencies, self.freq_max, side="right")
                ax0.set_xlim([frequencies[arg_min], frequencies[arg_max]])
                list_sum = np.sum(spectrogram[:,arg_min:arg_max], axis=1)
                ax1.plot(times_average, list_sum)
                ax1.set_xlim([times_average[0],times_average[-1]])
                ax1.set_title("decay curve")
                ax1.set_ylabel("area [arb. unit]")
                ax1.set_xlabel("times [s]")
                plt.show()
                return
        else:
            arg_min = np.searchsorted(frequencies, self.freq_min, side="left")
            arg_max = np.searchsorted(frequencies, self.freq_max, side="right")
            time_min = np.searchsorted(times_average, self.time_min, side="left")
            time_max = np.searchsorted(times_average, self.time_max, side="right")
        
            self.times_average = times_average[time_min:time_max]
            self.list_sum = np.sum(spectrogram[time_min:time_max,arg_min:arg_max], axis=1)

            p0 = [np.max(self.list_sum) - np.min(self.list_sum), - 1/times_average[int(len(times_average)/2)] * np.log(2), np.min(self.list_sum)]
        #print(p0)

            halfLife, curve_fit = self.calc_life(p0)

            ax0.set_xlim([frequencies[arg_min], frequencies[arg_max]])

            ax1.plot(self.times_average, self.list_sum)
            ax1.plot(self.times_average, curve_fit, color='r', lw=2)
            ax1.set_title("decay curve")
            ax1.set_ylabel("area [arb. unit]")
            ax1.set_xlabel("times [s]")
            print("half-life: {:} s".format(halfLife))
            plt.show()

    def set_freqRange(self, freq_min, freq_max):
        self.freq_min = freq_min
        self.freq_max = freq_max
        #if self.time_min == "":
        #    self.plot_data(only_waterfall=True)
        #else:
        #    self.plot_data(only_waterfall=False)

    def set_timeRange(self, time_min, time_max):
        if self.freq_min == "":
            print("please set the frequency range first!")
            return
        self.time_min = time_min
        self.time_max = time_max
        self.plot_data(only_waterfall=False)

    def calc_life(self, p0):
        '''
        fitting with the exponent function to get the life-time of the ion
        p0:             the initial guess for the parameters. [a, b, c]
                        using the function 
                            a * np.exp(b * x) + c
        '''

        def test_func(x, a, b, c):
            return a * np.exp(b * x) + c

        params, params_covariance = optimize.curve_fit(test_func, self.times_average, self.list_sum, p0=p0)
        halfLife = -np.log(2)/params[1]
        return halfLife, test_func(self.times_average, *params)

if __name__ == "__main__":
    #if len(sys.argv) != 2:
    #    print("Usage: {} path/to/file".format(__file__))
    #    sys.exit()
    lifetimeCalc = LifeTime("20181226_215255.wvd")
#    file_folder = "./"
#    file_name = "20181226_215255.wvd"
#    lifetimeCalc.plot_data(file_folder+file_name, window_length=500, n_average=40)
#    half_life = lifetimeCalc.calc_life(p0=[0.04,-0.07,0.024])

#time_min, time_max = 4, 19 # s
#freq_min, freq_max = 10, 50#-span/2, span/2

#window_length = 500 # number of data points in one frame
#n_frame = -1

#bud_3 = Processing(file_folder + file_name)
#cen_freq = bud_3.center_frequency*10**(-6) # MHz

#frequencies, times, spectrogram, n_dof = bud_3.time_average_2d(window_length=window_length, n_frame=n_frame, n_average=40)

#color_style = "viridis"

#plt.close("all")
#fig, ax = plt.subplots()

#pcm = ax.pcolormesh(frequencies, times, spectrogram, cmap=color_style)

#times_average = (times[:-1] + times[1:])/2
#arg_min = np.searchsorted(frequencies, freq_min, side="left")
#arg_max = np.searchsorted(frequencies, freq_max, side="right")
#time_min = np.searchsorted(times_average, time_min, side="left")
#time_max = np.searchsorted(times_average, time_max, side="right")
#ax.set_xlim([frequencies[arg_min], frequencies[arg_max]])

#cax = fig.colorbar(pcm, ax=ax)
#cax.set_label("power spectral density [arb. unit]")
#ax.set_xlabel("frequency - {:g} MHz [kHz]".format(cen_freq))
#ax.set_ylabel("times [s]")
#ax.set_title(file_name)
#plt.tight_layout(.5)

#list_sum = np.sum(spectrogram[time_min:time_max,arg_min:arg_max], axis=1)

#def test_func(x, a, b, c):
#    return a * np.exp(b * x) + c

#params, params_covariance = optimize.curve_fit(test_func, times_average[time_min:time_max], list_sum, p0=[0.04,-0.07,0.024])

#ax.plot(times_average[time_min:time_max], list_sum) 
#ax.plot(times_average[time_min:time_max], test_func(times_average[time_min:time_max], *params), color='g')

#print("half-life: {:} s".format(-np.log(2)/params[1]))


#plt.show()
