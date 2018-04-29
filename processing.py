#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
from scipy import linalg
from scipy.stats import chi2
import matplotlib.pyplot as plt
import sys, pyfftw, multiprocessing
from preprocessing import Preprocessing
from dpss import DPSS

class Processing(Preprocessing):
    '''
    The working horse for heavy-lifting jobs, mainly include windowing, averaging, and FFT etc.
    '''

    n_thread = multiprocessing.cpu_count() # number of CPU cores

    def __init__(self, file_str):
        '''
        the .wvh file should reside in the same directory where the .wvd file is found
        '''
        try:
            super().__init__(file_str)
        except FileNotFoundError:
            print("Error: cannot find the files!\nPlease check the input file name.")
            sys.exit()

    def spectrum(self, window_length=1000, n_offset=0, padding_ratio=2, window=None, beta=None):
        '''
        analyze the 1D frequency spectrum of the provided signal
        window_length:  length of the tapering window
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
        '''
        # handling various windows
        if window is None:
            window_sequence = np.ones(window_length)
        elif window == "kaiser":
            if beta is None:
                raise ValueError("additional argument beta is empty!")
            else:
                window_sequence = np.kaiser(window_length, beta)
        else:
            window_function = getattr(np, window)
            window_sequence = window_function(window_length)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1)[:-1] # Hz
        if n_point % 2 == 1: frequencies += self.sampling_rate / (2*n_point)
        # load the data, apply the window function
        signal = super().load(window_length, n_offset)[1] * window_sequence
        # create an FFT plan
        dummy = pyfftw.empty_aligned(window_length)
        fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=self.n_thread)
        spectrum = np.absolute(np.fft.fftshift(fft(signal))) / (self.sampling_rate*1e-3) # **voltage density** in V/kHz, not power density
                                                                        # This kind of normalization preserves the integral form, rather than the
                                                                        # summation form, of Parseval's theorem, which is convenient when one needs
                                                                        # to calculate the energy in a spectrum.
        return frequencies*1e-3, spectrum # kHz, V/kHz

    def fft_1d(self, window_length=1000, n_offset=0, padding_ratio=0, window=None, beta=None):
        '''
        simply calculate the 1D fast Fourier transform of the provided signal in a least intervened way
        window_length:  length of the tapering window
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
        '''
        # handling various windows
        if window is None:
            window_sequence = np.ones(window_length)
        elif window == "kaiser":
            if beta is None:
                raise ValueError("additional argument beta is empty!")
            else:
                window_sequence = np.kaiser(window_length, beta)
        else:
            window_function = getattr(np, window)
            window_sequence = window_function(window_length)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1)[:-1] # Hz
        if n_point % 2 == 1: frequencies += self.sampling_rate / (2*n_point)
        # load the data, apply the window function
        signal = super().load(window_length, n_offset)[1] * window_sequence
        # create an FFT plan
        dummy = pyfftw.empty_aligned(window_length)
        fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=self.n_thread)
        spectrum = np.fft.fftshift(fft(signal)) # V, complex-valued, orth-ordered Fourier transform
        return frequencies, spectrum # Hz, V

    def periodogram_1d(self, window_length=1000, n_offset=0, padding_ratio=2, window=None, beta=None):
        '''
        periodogram estimator for the spectral density estimation in 1D of the provided signal
        window_length:  length of the tapering window
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
        '''
        # handling various windows
        if window is None:
            window_sequence = np.ones(window_length)
        elif window == "kaiser":
            if beta is None:
                raise ValueError("additional argument beta is empty!")
            else:
                window_sequence = np.kaiser(window_length, beta)
        else:
            window_function = getattr(np, window)
            window_sequence = window_function(window_length)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1)[:-1] # Hz
        if n_point % 2 == 1: frequencies += self.sampling_rate / (2*n_point)
        # number of degrees of freedom
        n_dof = 2
        # load the data, apply the window function
        signal = super().load(window_length, n_offset)[1] * window_sequence
        # create an FFT plan
        dummy = pyfftw.empty_aligned(window_length)
        fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=self.n_thread)
        spectrum = np.absolute(np.fft.fftshift(fft(signal)))**2 / (self.sampling_rate*1e-3) / np.sum(window_sequence**2) # power density in V^2/kHz
                                                                                                        # This kind of normalization offers the peak area
                                                                                                        # as the power
        return frequencies*1e-3, spectrum, n_dof # kHz, V^2/kHz, 1

    def multitaper_1d(self, window_length=1000, n_offset=0, padding_ratio=2, half_bandwidth=3, n_taper=4, f_test=False):
        '''
        multitaper estimator for the spectral density estimation in 1D of the provided signal
        window_length:  length of the tapering window, a.k.a. N
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        half_bandwidth: half bandwidth in unit of fundamental frequency, a.k.a. NW preferably with integers
        n_taper:        number of tapering windows, a.k.a. K preferably with K < 2NW
        f_test:         whether to perform statistical test on the obtained spectrum for peak significance
        '''
        # prepare DPSSs
        dpss = DPSS(window_length, half_bandwidth, n_taper)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1)[:-1] # Hz
        if n_point % 2 == 1: frequencies += self.sampling_rate / (2*n_point)
        # number of degrees of freedom
        n_dof = 2 * n_taper
        # load the data, apply the window function
        signal = super().load(window_length, n_offset)[1] * dpss.vecs
        # create an FFT plan
        dummy = pyfftw.empty_aligned((n_taper, window_length))
        fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=self.n_thread)
        eigenspectrum = np.fft.fftshift(fft(signal))
        spectrum = np.mean(np.absolute(eigenspectrum)**2, axis=0) / (self.sampling_rate*1e-3) # power density in V^2/kHz
        if not f_test:
            return frequencies*1e-3, spectrum, n_dof # kHz, V^2/kHz, 1
        else:
            eigencoefficient = dpss.gen_spectra(n_point)
            a0 = np.sum(eigenspectrum*eigencoefficient[:,0:1], axis=0) / np.sum(eigencoefficient[:,0]**2)
            sigma2e = np.mean(np.absolute(eigenspectrum-a0*eigencoefficient[:,0:1])**2, axis=0)
            critical = n_taper * (np.power(window_length, 1/(n_taper-1))-1) / np.sum(eigencoefficient[:,0]**2)
            return frequencies*1e-3, spectrum, n_dof, np.absolute(a0)**2/sigma2e, critical.real # kHz, V^2/kHz, 1, 1, 1

    def adaptive_multitaper_1d(self, window_length=1000, n_offset=0, padding_ratio=2, half_bandwidth=3, n_taper=4, f_test=False):
        '''
        adaptive multitaper estimator for the spectral density estimation in 1D of the provided signal
        window_length:  length of the tapering window, a.k.a. N
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        half_bandwidth: half bandwidth in unit of fundamental frequency, a.k.a. NW preferably with integers
        n_taper:        number of tapering windows, a.k.a. K preferably with K < 2NW
        f_test:         whether to perform statistical test on the obtained spectrum for peak significance
        '''
        # prepare DPSSs
        dpss = DPSS(window_length, half_bandwidth, n_taper, True)
        vals = dpss.vals.reshape(n_taper, 1)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1)[:-1] # Hz
        if n_point % 2 == 1: frequencies += self.sampling_rate / (2*n_point)
        # load the data, apply the window function
        signal = super().load(window_length, n_offset)[1] * dpss.vecs
        # create an FFT plan
        dummy = pyfftw.empty_aligned((n_taper, window_length))
        fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=self.n_thread)
        eigenspectrum = np.fft.fftshift(fft(signal))
        # iteration begins
        spectrum = np.mean(np.absolute(eigenspectrum[:2])**2, axis=0) / (self.sampling_rate*1e-3) # power density in V^2/kHz
        while True:
            variance = np.sum(spectrum) / n_point # power density in V^2/kHz
            weight = (spectrum / (vals*spectrum + (1-vals)*variance))**2 * vals
            spectrum_test = np.average(np.absolute(eigenspectrum)**2, axis=0, weights=weight) / (self.sampling_rate*1e-3) # power density in V^2/kHz
            if np.allclose(spectrum_test, spectrum, rtol=1e-5, atol=1e-5): break
            spectrum = spectrum_test
        # number of degrees of freedom
        n_dof = 2 * np.sum(weight, axis=0)**2 / np.sum(weight**2, axis=0)
        if not f_test:
            return frequencies*1e-3, spectrum, n_dof # kHz, V^2/kHz, 1
        else:
            eigencoefficient = dpss.gen_spectra(n_point)
            a0 = np.sum(eigenspectrum*eigencoefficient[:,0:1]*weight, axis=0) / np.sum(eigencoefficient[:,0:1]**2*weight, axis=0)
            sigma2e = np.average(np.absolute(eigenspectrum-a0*eigencoefficient[:,0:1])**2, axis=0, weights=weight)
            critical = (np.power(window_length, 2/(n_dof-2))-1)\
                    * np.average(eigencoefficient[:,0:1]**2*np.ones((n_taper,n_point)), axis=0, weights=weight**2)\
                    / np.average(eigencoefficient[:,0:1]**2*np.ones((n_taper,n_point)), axis=0, weights=weight)**2
            return frequencies*1e-3, spectrum, n_dof, np.absolute(a0)**2/sigma2e, critical.real # kHz, V^2/kHz, 1, 1, 1

    def time_average_1d(self, window_length=1000, n_offset=0, padding_ratio=2, n_average=10, estimator='p', **kwargs):
        '''
        time-averaged estimator for the spectral density estimation in 1D of the provided signal
        window_length:  length of the tapering window
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        n_average:      number of FFTs for one average
        estimator:      base estimator on which the time-averaging is applied, to be chosen from ['p', 'm', 'a'],
                        which stands for periodogram, multitaper, and adaptive multitaper, respectively
        **kwargs:       some additional keyword arguments to be passed to the selected base estimator,
                        which includes
                        window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                                        if None, a rectangular window is implied
                                        if "kaiser" is given, an additional argument of beta is expected
                        half_bandwidth: half bandwidth in unit of fundamental frequency, a.k.a. NW preferably with integers
                        n_taper:        number of tapering windows, a.k.a. K preferably with K < 2NW
        '''
        if estimator == 'p': # periodogram
            frequencies, _, spectrogram, n_dof = self.periodogram_2d(window_length, n_average, n_offset, padding_ratio,
                    kwargs["window"], kwargs["beta"])
            # Bartlett's method for the boxcar window, Welch's method for all the others
            if kwargs["window"] is not None:
                spectrogram_suppl = self.periodogram_2d(window_length, n_average-1, n_offset+window_length//2, padding_ratio,
                        kwargs["window"], kwargs["beta"])[2]
                spectrogram = np.vstack( (spectrogram, spectrogram_suppl[:spectrogram.shape[0]-1]) )
            return (frequencies[:-1]+frequencies[1:])/2, np.mean(spectrogram, axis=0), n_dof*n_average # kHz, V^2/kHz, 1
        elif estimator == 'm': # multitaper
            frequencies, _, spectrogram, n_dof = self.multitaper_2d(window_length, n_average, n_offset, padding_ratio,
                    kwargs["half_bandwidth"], kwargs["n_taper"])
            return (frequencies[:-1]+frequencies[1:])/2, np.mean(spectrogram, axis=0), n_dof*n_average # kHz, V^2/kHz, 1
        elif estimator == 'a': # adaptive multitaper
            frequencies, _, spectrogram, n_dof = self.adaptive_multitaper_2d(window_length, n_average, n_offset, padding_ratio,
                    kwargs["half_bandwidth"], kwargs["n_taper"])
            return (frequencies[:-1]+frequencies[1:])/2, np.average(spectrogram, axis=0, weights=n_dof), np.sum(n_dof, axis=0) # kHz, V^2/kHz, 1
        else:
            raise ValueError("unrecognized identifier {:s} for the base estimator!".format(estimator))

    def plot_1d(self, frequencies, spectrum):
        '''
        plot the 1D frequency spectrum
        '''
        plt.close("all")
        fig, ax = plt.subplots()
        ax.plot(frequencies, spectrum)
        ax.set_xlim([frequencies[0], frequencies[-1]]) # kHz
        ax.set_xlabel("frequency − {:g} MHz [kHz]".format(self.center_frequency/1e6))
        ax.set_ylabel("power spectral density [arb. unit]")
        ax.set_title(self.fname)
        plt.tight_layout(.5)
        plt.show()

    def spectrogram(self, window_length=1000, n_frame=100, n_offset=0, padding_ratio=2, window=None, beta=None):
        '''
        analyze the 2D frequency spectrogram of the provided signal
        window_length:  length of the tapering window
        n_frame:        number of frames spanning along the time axis, negative means all the available frames
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
        '''
        # handling various windows
        if window is None:
            window_sequence = np.ones(window_length)
        elif window == "kaiser":
            if beta is None:
                raise ValueError("additional argument beta is empty!")
            else:
                window_sequence = np.kaiser(window_length, beta)
        else:
            window_function = getattr(np, window)
            window_sequence = window_function(window_length)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # crop the excessive frames
        if window_length*n_frame > self.n_sample-n_offset or n_frame < 0: n_frame = (self.n_sample-n_offset) // window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1) # Hz
        if n_point % 2 == 0: frequencies -= self.sampling_rate / (2*n_point)
        # build the time sequence
        times = (np.arange(n_frame+1)*window_length + n_offset) / self.sampling_rate # s
        # load the data in the block-wise
        n_block = super().n_buffer // window_length
        # placeholders for the transformed spectrogram
        spectrogram = np.full((n_frame, n_point), np.nan)
        # create a full FFT plan
        dummy_full = pyfftw.empty_aligned((n_block, window_length))
        fft_full = pyfftw.builders.fft(dummy_full, n=n_point, overwrite_input=True, threads=self.n_thread)
        while n_frame >= n_block:
            signal = super().load(window_length*n_block, n_offset)[1].reshape(n_block, window_length) * window_sequence
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            spectrogram[index:index+n_block] = np.absolute(np.fft.fftshift(fft_full(signal), axes=-1))\
                    / (self.sampling_rate*1e-3) # **voltage density** in V/kHz, not power density
            n_frame -= n_block
            if n_frame == 0: break
            n_offset += window_length*n_block
        else:
            # create a partial FFT plan
            dummy_part = pyfftw.empty_aligned((n_frame, window_length))
            fft_part = pyfftw.builders.fft(dummy_part, n=n_point, overwrite_input=True, threads=self.n_thread)
            signal = super().load(window_length*n_frame, n_offset)[1].reshape(n_frame, window_length) * window_sequence
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            spectrogram[index:] = np.absolute(np.fft.fftshift(fft_part(signal), axes=-1))\
                    / (self.sampling_rate*1e-3) # **voltage density** in V/kHz, not power density
        return frequencies*1e-3, times, spectrogram # kHz, s, V/kHz

    def fft_2d(self, window_length=1000, n_frame=100, n_offset=0, padding_ratio=0, window=None, beta=None):
        '''
        simply calculate the 2D fast Fourier transform of the provided signal in a least intervened way
        window_length:  length of the tapering window
        n_frame:        number of frames spanning along the time axis, negative means all the available frames
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
        '''
        # handling various windows
        if window is None:
            window_sequence = np.ones(window_length)
        elif window == "kaiser":
            if beta is None:
                raise ValueError("additional argument beta is empty!")
            else:
                window_sequence = np.kaiser(window_length, beta)
        else:
            window_function = getattr(np, window)
            window_sequence = window_function(window_length)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # crop the excessive frames
        if window_length*n_frame > self.n_sample-n_offset or n_frame < 0: n_frame = (self.n_sample-n_offset) // window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1) # Hz
        if n_point % 2 == 0: frequencies -= self.sampling_rate / (2*n_point)
        # build the time sequence
        times = (np.arange(n_frame+1)*window_length + n_offset) / self.sampling_rate # s
        # load the data in the block-wise
        n_block = super().n_buffer // window_length
        # placeholders for the transformed spectrogram
        spectrogram = np.full((n_frame, n_point), np.nan, dtype=complex)
        # create a full FFT plan
        dummy_full = pyfftw.empty_aligned((n_block, window_length))
        fft_full = pyfftw.builders.fft(dummy_full, n=n_point, overwrite_input=True, threads=self.n_thread)
        while n_frame >= n_block:
            signal = super().load(window_length*n_block, n_offset)[1].reshape(n_block, window_length) * window_sequence
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            spectrogram[index:index+n_block] = np.fft.fftshift(fft_full(signal), axes=-1) # V, complex-valued, orth-ordered Fourier transform
            n_frame -= n_block
            if n_frame == 0: break
            n_offset += window_length*n_block
        else:
            # create a partial FFT plan
            dummy_part = pyfftw.empty_aligned((n_frame, window_length))
            fft_part = pyfftw.builders.fft(dummy_part, n=n_point, overwrite_input=True, threads=self.n_thread)
            signal = super().load(window_length*n_frame, n_offset)[1].reshape(n_frame, window_length) * window_sequence
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            spectrogram[index:] = np.fft.fftshift(fft_part(signal), axes=-1) # V, complex-valued, orth-ordered Fourier transform
        return frequencies, times, spectrogram # Hz, s, V

    def periodogram_2d(self, window_length=1000, n_frame=100, n_offset=0, padding_ratio=2, window=None, beta=None, **kwargs):
        '''
        periodogram estimator for the spectral density estimation in 2D of the provided signal
        window_length:  length of the tapering window
        n_frame:        number of frames spanning along the time axis, negative means all the available frames
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
        '''
        # handling various windows
        if window is None:
            window_sequence = np.ones(window_length)
        elif window == "kaiser":
            if beta is None:
                raise ValueError("additional argument beta is empty!")
            else:
                window_sequence = np.kaiser(window_length, beta)
        else:
            window_function = getattr(np, window)
            window_sequence = window_function(window_length)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # crop the excessive frames
        if window_length*n_frame > self.n_sample-n_offset or n_frame < 0: n_frame = (self.n_sample-n_offset) // window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1) # Hz
        if n_point % 2 == 0: frequencies -= self.sampling_rate / (2*n_point)
        # build the time sequence
        times = (np.arange(n_frame+1)*window_length + n_offset) / self.sampling_rate # s
        # number of degrees of freedom
        n_dof = 2
        # load the data in the block-wise
        n_block = super().n_buffer // window_length
        # placeholders for the transformed spectrogram
        spectrogram = np.full((n_frame, n_point), np.nan)
        # create a full FFT plan
        dummy_full = pyfftw.empty_aligned((n_block, window_length))
        fft_full = pyfftw.builders.fft(dummy_full, n=n_point, overwrite_input=True, threads=self.n_thread)
        while n_frame >= n_block:
            signal = super().load(window_length*n_block, n_offset)[1].reshape(n_block, window_length) * window_sequence
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            spectrogram[index:index+n_block] = np.absolute(np.fft.fftshift(fft_full(signal), axes=-1))**2\
                    / (self.sampling_rate*1e-3) / np.sum(window_sequence**2) # power density in V^2/kHz
            n_frame -= n_block
            if n_frame == 0: break
            n_offset += window_length*n_block
        else:
            # create a partial FFT plan
            dummy_part = pyfftw.empty_aligned((n_frame, window_length))
            fft_part = pyfftw.builders.fft(dummy_part, n=n_point, overwrite_input=True, threads=self.n_thread)
            signal = super().load(window_length*n_frame, n_offset)[1].reshape(n_frame, window_length) * window_sequence
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            spectrogram[index:] = np.absolute(np.fft.fftshift(fft_part(signal), axes=-1))**2\
                    / (self.sampling_rate*1e-3) / np.sum(window_sequence**2) # power density in V^2/kHz
        return frequencies*1e-3, times, spectrogram, n_dof # kHz, s, V^2/kHz, 1

    def multitaper_2d(self, window_length=1000, n_frame=100, n_offset=0, padding_ratio=2, half_bandwidth=3, n_taper=4):
        '''
        multitaper estimator for the spectral density estimation in 2D of the provided signal
        window_length:  length of the tapering window, a.k.a. N
        n_frame:        number of frames spanning along the time axis, negative means all the available frames
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        half_bandwidth: half bandwidth in unit of fundamental frequency, a.k.a. NW preferably with integers
        n_taper:        number of tapering windows, a.k.a. K preferably with K < 2NW
        '''
        # prepare DPSSs
        dpss = DPSS(window_length, half_bandwidth, n_taper)
        vecs = dpss.vecs.reshape(n_taper, 1, window_length)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # crop the excessive frames
        if window_length*n_frame > self.n_sample-n_offset or n_frame < 0: n_frame = (self.n_sample-n_offset) // window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1) # Hz
        if n_point % 2 == 0: frequencies -= self.sampling_rate / (2*n_point)
        # build the time sequence
        times = (np.arange(n_frame+1)*window_length + n_offset) / self.sampling_rate # s
        # number of degrees of freedom
        n_dof = 2 * n_taper
        # load the data in the block-wise
        n_block = super().n_buffer // window_length
        # placeholders for the transformed spectrogram
        spectrogram = np.full((n_frame, n_point), np.nan)
        # create a full FFT plan
        dummy_full = pyfftw.empty_aligned((n_taper, n_block, window_length))
        fft_full = pyfftw.builders.fft(dummy_full, n=n_point, overwrite_input=True, threads=self.n_thread)
        while n_frame >= n_block:
            signal = super().load(window_length*n_block, n_offset)[1].reshape(1, n_block, window_length) * vecs
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            spectrogram[index:index+n_block] = np.mean(np.absolute(np.fft.fftshift(fft_full(signal), axes=-1))**2, axis=0)\
                    / (self.sampling_rate*1e-3) # power density in V^2/kHz
            n_frame -= n_block
            if n_frame == 0: break
            n_offset += window_length*n_block
        else:
            # create a partial FFT plan
            dummy_part = pyfftw.empty_aligned((n_taper, n_frame, window_length))
            fft_part = pyfftw.builders.fft(dummy_part, n=n_point, overwrite_input=True, threads=self.n_thread)
            signal = super().load(window_length*n_frame, n_offset)[1].reshape(1, n_frame, window_length) * vecs
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            spectrogram[index:] = np.mean(np.absolute(np.fft.fftshift(fft_part(signal), axes=-1))**2, axis=0)\
                    / (self.sampling_rate*1e-3) # power density in V^2/kHz
        return frequencies*1e-3, times, spectrogram, n_dof # kHz, s, V^2/kHz, 1

    def adaptive_multitaper_2d(self, window_length=1000, n_frame=100, n_offset=0, padding_ratio=2, half_bandwidth=3, n_taper=4):
        '''
        adaptive multitaper estimator for the spectral density estimation in 2D of the provided signal
        window_length:  length of the tapering window, a.k.a. N
        n_frame:        number of frames spanning along the time axis, negative means all the available frames
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        half_bandwidth: half bandwidth in unit of fundamental frequency, a.k.a. NW preferably with integers
        n_taper:        number of tapering windows, a.k.a. K preferably with K < 2NW
        '''
        # prepare DPSSs
        dpss = DPSS(window_length, half_bandwidth, n_taper, True)
        vecs = dpss.vecs.reshape(n_taper, 1, window_length)
        vals = dpss.vals.reshape(n_taper, 1, 1)
        # round the padded frame length up to the next radix-2 power
        n_point = int( np.power(2, np.ceil(np.log2(window_length*padding_ratio))) ) if padding_ratio >= 1 else window_length
        # crop the excessive frames
        if window_length*n_frame > self.n_sample-n_offset or n_frame < 0: n_frame = (self.n_sample-n_offset) // window_length
        # build the frequency sequence
        frequencies = np.linspace(-self.sampling_rate/2, self.sampling_rate/2, n_point+1) # Hz
        if n_point % 2 == 0: frequencies -= self.sampling_rate / (2*n_point)
        # build the time sequence
        times = (np.arange(n_frame+1)*window_length + n_offset) / self.sampling_rate # s
        # load the data in the block-wise
        n_block = super().n_buffer // window_length
        # placeholders for the transformed spectrogram and number of degrees of freedom
        spectrogram = np.full((n_frame, n_point), np.nan)
        n_dof = np.empty((n_frame, n_point))
        # create a full FFT plan
        dummy_full = pyfftw.empty_aligned((n_taper, n_block, window_length))
        fft_full = pyfftw.builders.fft(dummy_full, n=n_point, overwrite_input=True, threads=self.n_thread)
        while n_frame >= n_block:
            signal = super().load(window_length*n_block, n_offset)[1].reshape(1, n_block, window_length) * vecs
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            eigenspectrogram = np.fft.fftshift(fft_full(signal), axes=-1)
            # iteration begins
            spectrogram[index:index+n_block] = np.mean(np.absolute(eigenspectrogram[:2])**2, axis=0) / (self.sampling_rate*1e-3) # power density in V^2/kHz
            while True:
                variances = np.sum(spectrogram[index:index+n_block], axis=1, keepdims=True) / n_point # power density in V^2/kHz
                weights = (spectrogram[index:index+n_block] / (vals*spectrogram[index:index+n_block] + (1-vals)*variances))**2 * vals
                spectrogram_test = np.average(np.absolute(eigenspectrogram)**2, axis=0, weights=weights) / (self.sampling_rate*1e-3) # power density in V^2/kHz
                if np.allclose(spectrogram_test, spectrogram[index:index+n_block], rtol=1e-5, atol=1e-5): break
                spectrogram[index:index+n_block] = spectrogram_test
            n_dof[index:index+n_block] = 2 * np.sum(weights, axis=0)**2 / np.sum(weights**2, axis=0)
            n_frame -= n_block
            if n_frame == 0: break
            n_offset += window_length*n_block
        else:
            # create a partial FFT plan
            dummy_part = pyfftw.empty_aligned((n_taper, n_frame, window_length))
            fft_part = pyfftw.builders.fft(dummy_part, n=n_point, overwrite_input=True, threads=self.n_thread)
            signal = super().load(window_length*n_frame, n_offset)[1].reshape(1, n_frame, window_length) * vecs
            index = spectrogram[~np.isnan(spectrogram[:,0]), 0].size
            eigenspectrogram = np.fft.fftshift(fft_part(signal), axes=-1)
            # iteration begins
            spectrogram[index:] = np.mean(np.absolute(eigenspectrogram[:2])**2, axis=0) / (self.sampling_rate*1e-3) # power density in V^2/kHz
            while True:
                variances = np.sum(spectrogram[index:], axis=1, keepdims=True) / n_point # power density in V^2/kHz
                weights = (spectrogram[index:] / (vals*spectrogram[index:] + (1-vals)*variances))**2 * vals
                spectrogram_test = np.average(np.absolute(eigenspectrogram)**2, axis=0, weights=weights) / (self.sampling_rate*1e-3) # power density in V^2/kHz
                if np.allclose(spectrogram_test, spectrogram[index:], rtol=1e-5, atol=1e-5): break
                spectrogram[index:] = spectrogram_test
            n_dof[index:] = 2 * np.sum(weights, axis=0)**2 / np.sum(weights**2, axis=0)
        return frequencies*1e-3, times, spectrogram, n_dof # kHz, s, V^2/kHz, 1

    def time_average_2d(self, window_length=1000, n_frame=100, n_offset=0, padding_ratio=2, n_average=10, estimator='p', **kwargs):
        '''
        time-averaged estimator for the spectral density estimation in 2D of the provided signal
        window_length:  length of the tapering window
        n_frame:        number of frames spanning along the time axis, negative means all the available frames
        n_offset:       number of IQ pairs to be skipped over
        padding_ratio:  >= 1, ratio of the full frame length after zero padding to the window length
                        note that the final frame length will be rounded up to the next power of base 2
                        any illegal values disable zero padding
        n_average:      number of FFTs for one average
        estimator:      base estimator on which the time-averaging is applied, to be chosen from ['p', 'm', 'a'],
                        which stands for periodogram, multitaper, and adaptive multitaper, respectively
        window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
        '''
        if estimator == 'p': # periodogram
            frequencies, times, spectrogram, n_dof = self.periodogram_2d(window_length, n_frame*n_average, n_offset, padding_ratio,
                    kwargs["window"], kwargs["beta"])
            n_frame = spectrogram.shape[0] // n_average
            spectrogram = spectrogram[:n_frame*n_average].reshape(n_frame, n_average, -1)
            # Bartlett's method for the boxcar window, Welch's method for all the others
            if kwargs["window"] is not None:
                spectrogram_suppl = self.periodogram_2d(window_length, n_frame*n_average-1, n_offset+window_length//2, padding_ratio,
                        kwargs["window"], kwargs["beta"])[2]
                spectrogram_suppl = np.vstack( (spectrogram_suppl, np.empty(spectrogram.shape[-1])) ).reshape(n_frame, n_average, -1)[:,:-1,:]
                spectrogram = np.hstack( (spectrogram, spectrogram_suppl) )
            return frequencies, times[::n_average], np.mean(spectrogram, axis=1), n_dof*n_average # kHz, s, V^2/kHz, 1
        elif estimator == 'm': # multitaper
            frequencies, times, spectrogram, n_dof = self.multitaper_2d(window_length, n_frame*n_average, n_offset, padding_ratio,
                    kwargs["half_bandwidth"], kwargs["n_taper"])
            n_frame = spectrogram.shape[0] // n_average
            spectrogram = spectrogram[:n_frame*n_average].reshape(n_frame, n_average, -1)
            return frequencies, times[::n_average], np.mean(spectrogram, axis=1), n_dof*n_average # kHz, s, V^2/kHz, 1
        elif estimator == 'a': # adaptive multitaper
            frequencies, times, spectrogram, n_dof = self.adaptive_multitaper_2d(window_length, n_frame*n_average, n_offset, padding_ratio,
                    kwargs["half_bandwidth"], kwargs["n_taper"])
            n_frame = spectrogram.shape[0] // n_average
            spectrogram = spectrogram[:n_frame*n_average].reshape(n_frame, n_average, -1)
            n_dof = n_dof[:n_frame*n_average].reshape(n_frame, n_average, -1)
            return frequencies, times[::n_average], np.average(spectrogram, axis=1, weights=n_dof), np.sum(n_dof, axis=1) # kHz, s, V^2/kHz, 1
        else:
            raise ValueError("unrecognized identifier {:s} for the base estimator!".format(estimator))

    def plot_2d(self, frequencies, times, spectrogram):
        '''
        plot the 2D time-frequency spectrogram
        '''
        plt.close("all")
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(frequencies, times, spectrogram)
        ax.set_xlim([frequencies[0], frequencies[-1]]) # kHz
        ax.set_ylim([times[0], times[-1]]) # s
        ax.set_xlabel("frequency − {:g} MHz [kHz]".format(self.center_frequency/1e6))
        ax.set_ylabel("time [s]")
        cax = fig.colorbar(pcm, ax=ax)
        cax.set_label("power spectral density [arb. unit]")
        ax.set_title(self.fname)
        plt.tight_layout(.5)
        plt.show()

    def confidence_band(self, sde, level, n_dof):
        '''
        calculate the confidence band of the spectral density estimate at a given confidence level
        sde:    spectral density estimate
        level:  confidence level
        n_dof:  number of degrees of freedom
        '''
        upper_quantile, lower_quantile = chi2.ppf((1+level)/2, n_dof), chi2.ppf((1-level)/2, n_dof)
        upper_bound, lower_bound = n_dof/lower_quantile*sde, n_dof/upper_quantile*sde
        return upper_bound, lower_bound


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} path/to/file".format(__file__))
        sys.exit()
    processing = Processing(sys.argv[-1])
    processing.plot_2d(*processing.adaptive_multitaper_2d(window_length=500, n_frame=1000, n_offset=0, padding_ratio=1, half_bandwidth=3, n_taper=4))
