#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import norm

class Synthetic(object):
    '''
    A class for generations of synthetic signals, whether deterministic or stochastic
    The synthesized signals are stored in data files in compliance with R&S .wv format
    '''

    def __init__(self, span=1e6, duration=1):
        '''
        duration in unit of s
        span in unit of Hz
        '''
        self.metadata = {
                "center frequency": 200e6, # Hz
                "endian": "little", 
                "format": "int16", 
                "reference level": -30., # dBm
                "resolution": 16,
                "timestamp": "2000-01-01T00:00:00+0000",
                }
        self.metadata["duration"] = duration
        self.metadata["span"] = span
        self.metadata["sampling rate"] = span * 1.25
        self.metadata["number of samples"] = (int(self.metadata["sampling rate"]*self.metadata["duration"]/2621440) + 1) * 2621440

    def file_gen(self, in_phase, quadrature, fname):
        '''
        write the metadata to a .wvh file and write the synthetic signal to a .wvd file
        '''
        with open(fname+".wvh", 'w') as header:
            json.dump(self.metadata, header, indent=4, sort_keys=True)
        IQ = np.zeros(2*in_phase.size, dtype=in_phase.dtype)
        IQ[0::2] = in_phase
        if quadrature is not None:
            IQ[1::2] = quadrature
        with open(fname+".wvd", "wb") as data:
            IQ.tofile(data)
        print("file `" + fname + "' saved")

    def draw(self, t, in_phase, quadrature):
        if quadrature is None:
            quadrature = np.zeros_like(in_phase)
        plt.close("all")
        fig, (axi, axq) = plt.subplots(2, 1, sharex=True, sharey=True)
        axi.plot(t, in_phase)
        axi.set_ylabel("in phase")
        axq.plot(t, quadrature)
        axq.set_xlim([t.min(), t.max()])
        axq.set_xlabel("time [s]")
        axq.set_ylabel("quadrature")
        plt.tight_layout(.5)
        plt.show()

    def sinusoid(self, freq, amp=1, phi=0, fname="sinusoid", draw=False):
        '''
        amp * e^[ i * (2*pi*freq*t + phi) ]
        freq in unit of Hz
        normalized amp, not exceed 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        real = amp * np.cos(2*np.pi*freq*comb+phi) * (2**(self.metadata["resolution"]-1) - 1)
        imag = amp * np.sin(2*np.pi*freq*comb+phi) * (2**(self.metadata["resolution"]-1) - 1)
        in_phase = real.astype(self.metadata["format"])
        quadrature = imag.astype(self.metadata["format"])
        self.file_gen(in_phase, quadrature, fname)
        if draw:
            self.draw(comb, in_phase, quadrature)

    def rectangle(self, freq, amp=1, duty=.5, fname="rectangle", draw=False):
        '''
        rectangular pulse train
        freq in unit of Hz
        normalized amp, not exceed 1
        duty cycle between 0 and 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        real = np.zeros(comb.size)
        real[np.where( np.modf(freq*comb)[0] <= duty )] = amp * (2**(self.metadata["resolution"]-1) - 1)
        in_phase = real.astype(self.metadata["format"])
        self.file_gen(in_phase, None, fname)
        if draw:
            self.draw(comb, in_phase, None)

    def triangle(self, freq, amp=1, duty=1, fname="triangle", draw=False):
        '''
        triangular pulse train
        freq in unit of Hz
        normalized amp, not exceed 1
        duty cycle between 0 and 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        whole = amp / (duty/2) * np.modf(freq*comb)[0]
        subtractee = np.where(whole-amp<0, 0, whole-amp) * 2
        real = np.where(whole-subtractee<0, 0, whole-subtractee) * (2**(self.metadata["resolution"]-1) - 1)
        in_phase = real.astype(self.metadata["format"])
        self.file_gen(in_phase, None, fname)
        if draw:
            self.draw(comb, in_phase, None)

    def exponent(self, freq, amp=1, tau=1/3, fname="exponent", draw=False):
        '''
        exponential pulse train
        freq in unit of Hz
        normalized amp, not exceed 1
        tau characterizes decay rate
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        real = amp * np.exp(-np.modf(freq*comb)[0] / tau) * (2**(self.metadata["resolution"]-1) - 1)
        in_phase = real.astype(self.metadata["format"])
        self.file_gen(in_phase, None, fname)
        if draw:
            self.draw(comb, in_phase, None)

    def chirp(self, f0, k, amp=1, phi=0, fname="chirp", draw=False):
        '''
        amp * e^[ i * (2*pi*cycle(t) + phi) ]
        cycle(t) = f0*t + k/2*t**2
        instantaneous frequency linearly depends on t, in unit of Hz
        normalized amp, not exceed 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        cycle = f0*comb + k/2*comb**2
        real = amp * np.cos(2*np.pi*cycle+phi) * (2**(self.metadata["resolution"]-1) - 1)
        imag = amp * np.sin(2*np.pi*cycle+phi) * (2**(self.metadata["resolution"]-1) - 1)
        in_phase = real.astype(self.metadata["format"])
        quadrature = imag.astype(self.metadata["format"])
        self.file_gen(in_phase, quadrature, fname)
        if draw:
            self.draw(comb, in_phase, quadrature)

    def ar0(self, seed, amp=1, fname="ar0", draw=False):
        '''
        stationary Gaussian white noise: x[n] = e[n]
        seed for generation of normal random numbers
        normalized amp, not exceed 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        np.random.seed(seed)
        rv = norm(loc=0, scale=amp/5.5)
        real = rv.rvs(size=comb.size) * (2**(self.metadata["resolution"]-1) - 1)
        in_phase = real.astype(self.metadata["format"])
        self.file_gen(in_phase, None, fname)
        if draw:
            self.draw(comb, in_phase, None)

    def ar1(self, seed, amp=1, fname="ar1", draw=False):
        '''
        1st order autoregressive process: x[n] = .3x[n-1] + e[n]
        seed for generation of normal random numbers
        normalized amp, not exceed 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        np.random.seed(seed)
        rv = norm(loc=0, scale=amp/6)
        noise = rv.rvs(size=comb.size) * (2**(self.metadata["resolution"]-1) - 1)
        real = np.zeros(comb.size)
        for i in range(comb.size):
            real[i] = .3 * real[i-1] + noise[i]
        in_phase = real.astype(self.metadata["format"])
        self.file_gen(in_phase, None, fname)
        if draw:
            self.draw(comb, in_phase, None)

    def car1(self, seed, amp=1, fname="car1", draw=False):
        '''
        1st order complex autoregressive process: x[n] = (.3+.4i)x[n-1] + (e1[n]+e2[n]i)
        seed for generation of normal random numbers
        normalized amp, not exceed 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        np.random.seed(seed)
        rv = norm(loc=0, scale=amp/5)
        noise = rv.rvs(size=2*comb.size) * (2**(self.metadata["resolution"]-1) - 1)
        real, imag = np.zeros(comb.size), np.zeros(comb.size)
        for i in range(comb.size):
            real[i] = .3 * real[i-1] - .4 * imag[i-1] + noise[2*i]
            imag[i] = .4 * real[i-1] + .3 * imag[i-1] + noise[2*i+1]
        in_phase = real.astype(self.metadata["format"])
        quadrature = imag.astype(self.metadata["format"])
        self.file_gen(in_phase, quadrature, fname)
        if draw:
            self.draw(comb, in_phase, quadrature)

    def ar2(self, seed, amp=1, fname="ar2", draw=False):
        '''
        2nd order autoregressive process: x[n] = .75x[n-1] - .5x[n-2] + e[n]
        seed for generation of normal random numbers
        normalized amp, not exceed 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        np.random.seed(seed)
        rv = norm(loc=0, scale=amp/7)
        noise = rv.rvs(size=comb.size) * (2**(self.metadata["resolution"]-1) - 1)
        real = np.zeros(comb.size)
        for i in range(comb.size):
            real[i] = .75 * real[i-1] - .5 * real[i-2] + noise[i]
        in_phase = real.astype(self.metadata["format"])
        self.file_gen(in_phase, None, fname)
        if draw:
            self.draw(comb, in_phase, None)

    def ar4(self, seed, amp=1, fname="ar4", draw=False):
        '''
        4th order autoregressive process: x[n] = 2.7607x[n-1] - 3.8106x[n-2] + 2.6535x[n-3] - .9238x[n-4] + e[n]
        seed for generation of normal random numbers
        normalized amp, not exceed 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        np.random.seed(seed)
        rv = norm(loc=0, scale=amp/140)
        noise = rv.rvs(size=comb.size) * (2**(self.metadata["resolution"]-1) - 1)
        real = np.zeros(comb.size)
        for i in range(comb.size):
            real[i] = 2.7607 * real[i-1] - 3.8106 * real[i-2] + 2.6535 * real[i-3] -.9238 * real[i-4] + noise[i]
        in_phase = real.astype(self.metadata["format"])
        self.file_gen(in_phase, None, fname)
        if draw:
            self.draw(comb, in_phase, None)

    def ar6(self, seed, amp=1, fname="ar6", draw=False):
        '''
        6th order autoregressive process: x[n] = 3.9515x[n-1] - 7.8885x[n-2] + 9.734x[n-3] - 7.7435x[n-4] + 3.8078x[n-5] - .9472x[n-6] + e[n]
        seed for generation of normal random numbers
        normalized amp, not exceed 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        np.random.seed(seed)
        rv = norm(loc=0, scale=amp/245)
        noise = rv.rvs(size=comb.size) * (2**(self.metadata["resolution"]-1) - 1)
        real = np.zeros(comb.size)
        for i in range(comb.size):
            real[i] = 3.9515 * real[i-1] - 7.8885 * real[i-2] + 9.734 * real[i-3] - 7.7435 * real[i-4] + 3.8078 * real[i-5] - .9472 * real[i-6] + noise[i]
        in_phase = real.astype(self.metadata["format"])
        self.file_gen(in_phase, None, fname)
        if draw:
            self.draw(comb, in_phase, None)

    def noisy_sinusoid(self, seed, freq, amp_total=1, amp_ratio=1, phi=0, fname="noisy_sinusoid", draw=False):
        '''
        amp_s * e^[ i * (2*pi*freq*t + phi) ] + err
        seed for generation of normal random numbers
        freq in unit of Hz
        amp_total = amp_s + amp_e, not exceed 1
        amp_ratio = amp_s / amp_e
        '''
        np.random.seed(seed)
        amp_s = amp_total * amp_ratio / (1+amp_ratio)
        amp_e = amp_total / (1+amp_ratio)
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        rv = norm(loc=0, scale=amp_e/5.5)
        real = ( amp_s * np.cos(2*np.pi*freq*comb+phi) + rv.rvs(size=comb.size) ) * (2**(self.metadata["resolution"]-1) - 1)
        imag = ( amp_s * np.sin(2*np.pi*freq*comb+phi) + rv.rvs(size=comb.size) ) * (2**(self.metadata["resolution"]-1) - 1)
        in_phase = real.astype(self.metadata["format"])
        quadrature = imag.astype(self.metadata["format"])
        self.file_gen(in_phase, quadrature, fname)
        if draw:
            self.draw(comb, in_phase, quadrature)

    def noisy_sinusoids(self, seed, *args, amp_total=1, fname="noisy_sinusoids", draw=False):
        '''
        an extension to noisy_sinusoid by allowing for multiple periodicities
        err + { amp_s * e^[ i * (2*pi*freq*t + phi) ] }
        seed for generation of normal random numbers
        amp_total = amp_e + { amp_s }, not exceed 1
        each sinusoid is determined by a 3-tuple (freq, amp_ratio, phi), which is passed by *args
        freq in unit of Hz
        amp_ratio = amp_s / amp_e
        SNR: window_length / band_width * (5.5 * amp_ratio)^2 / 2
        '''
        np.random.seed(seed)
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        amp_ratio_total = np.sum([arg[1] for arg in args])
        amp_e = amp_total / (1+amp_ratio_total)
        rv = norm(loc=0, scale=amp_e/5.5)
        real, imag = rv.rvs(size=comb.size), rv.rvs(size=comb.size)
        for freq, amp_ratio, phi in args:
            amp_s = amp_total * amp_ratio / (1+amp_ratio_total)
            real += amp_s * np.cos(2*np.pi*freq*comb + phi)
            imag += amp_s * np.sin(2*np.pi*freq*comb + phi)
        in_phase = ( real * (2**(self.metadata["resolution"]-1) - 1) ).astype(self.metadata["format"])
        quadrature = ( imag * (2**(self.metadata["resolution"]-1) - 1) ).astype(self.metadata["format"])
        self.file_gen(in_phase, quadrature, fname)
        if draw:
            self.draw(comb, in_phase, quadrature)

    def clipped_sinusoid(self, freq, amp=3, phi=0, fname="clipped_sinusoid", draw=False):
        '''
        amp * e^[ i * (2*pi*freq*t + phi) ], of which any values exceeding +/-1 will be clipped to +/-1
        freq in unit of Hz
        normalized amp, no less than 1
        '''
        comb = np.arange(self.metadata["number of samples"]) / self.metadata["sampling rate"]
        real = np.clip(amp*np.cos(2*np.pi*freq*comb+phi), -1, 1) * (2**(self.metadata["resolution"]-1) - 1)
        imag = np.clip(amp*np.sin(2*np.pi*freq*comb+phi), -1, 1) * (2**(self.metadata["resolution"]-1) - 1)
        in_phase = real.astype(self.metadata["format"])
        quadrature = imag.astype(self.metadata["format"])
        self.file_gen(in_phase, quadrature, fname)
        if draw:
            self.draw(comb, in_phase, quadrature)


if __name__=="__main__":
    synthetic = Synthetic()
    synthetic.sinusoid(32e3, amp=.3, phi=np.pi/4, draw=True)
