#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
import matplotlib.pyplot as plt
import json
from provider import Provide as Random

class Synthetic(Random):
    '''
    A class for generations of synthetic signals, whether deterministic or stochastic
    The synthesized signals are stored in data files in compliance with R&S .wv format
    '''

    def __init__(self, sampling_rate=1e6, duration=1, source="2019-10-24.bin"):
        '''
        sampling rate in unit of Hz
        duration in unit of s
        source is the name of a random-bit file, can be retrieved from RANDOM.ORG
        '''
        super().__init__(source)
        self.metadata = {
                "center frequency": 200e6, # Hz
                "endian": "little", 
                "format": "int16", 
                "reference level": -30., # dBm
                "resolution": 16,
                "timestamp": "2000-01-01T00:00:00+0000",
                }
        self.metadata["duration"] = duration
        self.metadata["span"] = sampling_rate / 1.25
        self.metadata["sampling rate"] = sampling_rate
        self.metadata["number of samples"] = int(sampling_rate*duration)
        # self.metadata["number of samples"] = (int(sampling_rate*duration/2621440) + 1) * 2621440
        self.comb = np.arange(self.metadata["number of samples"]) / sampling_rate # sampling comb

    def file_gen(self, comp, fname, draw):
        '''
        write the metadata to a .wvh file, digitize the synthetic signal, and write it to a .wvd file
        the synthetic signal will first be clipped once it exceeds the full dynamic range
        linear pulse-code modulation is used for digitization, with one half rounded downwards
        '''
        with open(fname+".wvh", 'w') as header:
            json.dump(self.metadata, header, indent=4, sort_keys=True)
        print("maximum value is {:g}".format(np.max(np.abs(comp))))
        comp_clip = np.clip(comp, -1, 1)
        comp_clip *= 2**(self.metadata["resolution"]-1) - .5
        comp_dig, quant_err = np.divmod(comp_clip, 1)
        IQ = np.where(quant_err <= .5, comp_dig, comp_dig+1)
        with open(fname+".wvd", "wb") as data:
            IQ.astype(self.metadata["format"]).tofile(data)
        print("file `" + fname + "' saved")
        if draw:
            self.draw(IQ)

    def draw(self, IQ):
        in_phase, quadrature = IQ[::2], IQ[1::2]
        plt.close("all")
        fig, (axi, axq) = plt.subplots(2, 1, sharex=True, sharey=True)
        axi.plot(self.comb, in_phase)
        axi.set_ylabel("in phase")
        axq.plot(self.comb, quadrature)
        axq.set_xlim([self.comb.min(), self.comb.max()])
        axq.set_xlabel("time [s]")
        axq.set_ylabel("quadrature")
        plt.show()

    def rectangle(self, freq, amp=1, duty=.5, fname="rectangle", draw=False):
        '''
        rectangular pulse train
        freq in unit of Hz
        normalized amp, with 1 corresponding to the full dynamic range
        duty cycle between 0 and 1
        '''
        comp = np.zeros(2*self.metadata["number of samples"])
        comp[::2] = amp * (np.modf(freq*self.comb)[0] <= duty).astype(float)
        self.file_gen(comp, fname, draw)

    def triangle(self, freq, amp=1, duty=1, fname="triangle", draw=False):
        '''
        triangular pulse train
        freq in unit of Hz
        normalized amp, with 1 corresponding to the full dynamic range
        duty cycle between 0 and 1
        '''
        comp = np.zeros(2*self.metadata["number of samples"])
        comp[::2] = amp * (1 - np.abs(np.modf(freq*self.comb)[0]*2/duty - 1)) * (np.modf(freq*self.comb)[0] <= duty).astype(float)
        self.file_gen(comp, fname, draw)

    def exponent(self, freq, amp=1, tau=1/3, fname="exponent", draw=False):
        '''
        exponential pulse train
        freq in unit of Hz
        normalized amp, with 1 corresponding to the full dynamic range
        tau characterizes decay rate
        '''
        comp = np.zeros(2*self.metadata["number of samples"])
        comp[::2] = amp * np.exp(-np.modf(freq*self.comb)[0]/tau)
        self.file_gen(comp, fname, draw)

    def sinusoid(self, freq, amp=1, phi=0, fname="sinusoid", draw=False):
        '''
        amp * e^[ i * (2*pi*freq*t + phi) ]
        freq in unit of Hz
        normalized amp, with 1 corresponding to the full dynamic range
        '''
        comp = amp * np.exp(1j*(2*np.pi*freq*self.comb+phi)).view(float)
        self.file_gen(comp, fname, draw)

    def sinusoids(self, *args, fname="sinusoids", draw=False):
        '''
        { amp * e^[ i * (2*pi*freq*t + phi) ] }
        each sinusoid is determined by a 3-tuple (freq, amp, phi), which is passed by *args
        freq in unit of Hz
        '''
        comp = np.zeros(self.metadata["number of samples"], dtype=complex)
        for freq, amp, phi in args:
            comp += amp * np.exp(1j*(2*np.pi*freq*self.comb+phi))
        self.file_gen(comp.view(float), fname, draw)

    def chirp(self, f0, k, amp=1, phi=0, fname="chirp", draw=False):
        '''
        amp * e^[ i * (2*pi*cycle(t) + phi) ]
        cycle(t) = f0*t + k/2*t**2
        instantaneous frequency linearly depends on t, in unit of Hz
        normalized amp, with 1 corresponding to the full dynamic range
        '''
        cycle = f0*self.comb + k/2*self.comb**2
        comp = amp * np.exp(1j*(2*np.pi*cycle+phi)).view(float)
        self.file_gen(comp, fname, draw)

    def noisy_sinusoid(self, seed, freq, sigma=1/10, snr=10, phi=0, fname="noisy_sinusoid", draw=False):
        '''
        err + amp * e^[ i * (2*pi*freq*t + phi) ]
        seed is a non-negative integer, to alter normal random numbers
        sigma is the standard deviation of the noise
        freq in unit of Hz
        snr = 20 * lg(amp/sigma) in unit of dB
        '''
        noise = sigma * super().gaussian(self.metadata["number of samples"], seed)
        amp = 10**(snr/20) * sigma
        sinusoid = amp * np.exp(1j*(2*np.pi*freq*self.comb+phi))
        comp = (noise+sinusoid).view(float)
        self.file_gen(comp, fname, draw)

    def noisy_sinusoids(self, seed, *args, sigma=1/10, fname="noisy_sinusoids", draw=False):
        '''
        an extension to noisy_sinusoid by allowing for multiple periodicities: err + { amp * e^[ i * (2*pi*freq*t + phi) ] }
        seed is a non-negative integer, to alter normal random numbers
        sigma is the standard deviation of the noise
        each sinusoid is determined by a 3-tuple (freq, snr, phi), which is passed by *args
        freq in unit of Hz
        snr = 20 * lg(amp/sigma) in unit of dB
        '''
        noise = sigma * super().gaussian(self.metadata["number of samples"], seed)
        sinusoids = np.zeros_like(noise)
        for freq, snr, phi in args:
            amp = 10**(snr/20) * sigma
            sinusoids += amp * np.exp(1j*(2*np.pi*freq*self.comb+phi))
        comp = (noise+sinusoids).view(float)
        self.file_gen(comp, fname, draw)

    def ar0(self, seed, sigma=1/5, fname="ar0", draw=False):
        '''
        stationary Gaussian white noise: x[n] = e[n], with sigma being the standard deviation
        the real and imaginary parts are two independent stochastic processes
        seed is a non-negative integer, to alter normal random numbers
        '''
        comp = np.sqrt(2)*sigma * super().gaussian(self.metadata["number of samples"], seed)
        self.file_gen(comp.view(float), fname, draw)

    def ar1(self, seed, sigma=1/6, fname="ar1", draw=False):
        '''
        1st order autoregressive process: x[n] = .3x[n-1] + e[n], with sigma being the standard deviation
        the real and imaginary parts are two independent stochastic processes
        seed is a non-negative integer, to alter normal random numbers
        '''
        noise = np.sqrt(2)*sigma * super().gaussian(self.metadata["number of samples"], seed)
        comp = np.zeros_like(noise)
        for i in range(self.metadata["number of samples"]):
            comp[i] = .3 * comp[i-1] + noise[i]
        self.file_gen(comp.view(float), fname, draw)

    def car1(self, seed, sigma=1/5, fname="car1", draw=False):
        '''
        1st order complex autoregressive process: x[n] = (.3+.4i)x[n-1] + (e1[n]+e2[n]i), with sigma being the standard deviation
        seed is a non-negative integer, to alter normal random numbers
        '''
        noise = sigma * super().gaussian(self.metadata["number of samples"], seed)
        comp = np.zeros_like(noise)
        for i in range(self.metadata["number of samples"]):
            comp[i] = (.3+.4j) * comp[i-1] + noise[i]
        self.file_gen(comp.view(float), fname, draw)

    def ar2(self, seed, sigma=1/7, fname="ar2", draw=False):
        '''
        2nd order autoregressive process: x[n] = .75x[n-1] - .5x[n-2] + e[n], with sigma being the standard deviation
        the real and imaginary parts are two independent stochastic processes
        seed is a non-negative integer, to alter normal random numbers
        '''
        noise = np.sqrt(2)*sigma * super().gaussian(self.metadata["number of samples"], seed)
        comp = np.zeros_like(noise)
        for i in range(self.metadata["number of samples"]):
            comp[i] = .75 * comp[i-1] - .5 * comp[i-2] + noise[i]
        self.file_gen(comp.view(float), fname, draw)

    def ar4(self, seed, sigma=1/125, fname="ar4", draw=False):
        '''
        4th order autoregressive process: x[n] = 2.7607x[n-1] - 3.8106x[n-2] + 2.6535x[n-3] - .9238x[n-4] + e[n], with sigma being the standard deviation
        the real and imaginary parts are two independent stochastic processes
        seed is a non-negative integer, to alter normal random numbers
        '''
        noise = np.sqrt(2)*sigma * super().gaussian(self.metadata["number of samples"], seed)
        comp = np.zeros_like(noise)
        for i in range(self.metadata["number of samples"]):
            comp[i] = 2.7607 * comp[i-1] - 3.8106 * comp[i-2] + 2.6535 * comp[i-3] -.9238 * comp[i-4] + noise[i]
        self.file_gen(comp.view(float), fname, draw)

    def ar6(self, seed, sigma=1/230, fname="ar6", draw=False):
        '''
        6th order autoregressive process: x[n] = 3.9515x[n-1] - 7.8885x[n-2] + 9.734x[n-3] - 7.7435x[n-4] + 3.8078x[n-5] - .9472x[n-6] + e[n], with sigma being the standard deviation
        the real and imaginary parts are two independent stochastic processes
        seed is a non-negative integer, to alter normal random numbers
        '''
        noise = np.sqrt(2)*sigma * super().gaussian(self.metadata["number of samples"], seed)
        comp = np.zeros_like(noise)
        for i in range(self.metadata["number of samples"]):
            comp[i] = 3.9515 * comp[i-1] - 7.8885 * comp[i-2] + 9.734 * comp[i-3] - 7.7435 * comp[i-4] + 3.8078 * comp[i-5] - .9472 * comp[i-6] + noise[i]
        self.file_gen(comp.view(float), fname, draw)


if __name__=="__main__":
    synthetic = Synthetic()
    synthetic.sinusoid(1e3, amp=1, phi=np.pi/4, draw=True)
