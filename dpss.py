#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import pyfftw, multiprocessing

class DPSS(object):
    '''
    A class for generating discrete prolate spheroidal sequences, plotting them and their Fourier transforms
    '''

    n_thread = multiprocessing.cpu_count() # number of CPU cores

    def __init__(self, N=1000, NW=3, K=4, eigenvalue=False):
        '''
        N: sequence length
        W: half resolution bandwidth
        NW: N * W, preferable for integers
        K: number of tapers ordered descendently by their eigenvalues, should be no more than N
        '''
        self.N = N
        self.W = NW / N
        self.K = K
        self.gen_sequences(eigenvalue)

    def gen_sequences(self, eigenvalue):
        '''
        generate the discrete prolate spheroidal sequences in the time domain
        '''
        diag_main = ((self.N-1)/2-np.arange(self.N))**2 * np.cos(2*np.pi*self.W)
        diag_off = np.arange(1, self.N) * np.arange(self.N-1, 0, -1) / 2
        B = sparse.diags([diag_main, diag_off, diag_off], [0, 1, -1])
        dummy = 0
        while True: # calculate a few more eigenvectors to prevent from instability of numerical calculation when sequence length is small
            vals, vecs = linalg.eigsh(B, self.K+dummy)
            if vals[dummy] < 0:
                dummy += 1
            else:
                break
        vecs = vecs[:,dummy:]
        self.vecs = (vecs * np.where(vecs[0,:]>0, 1, -1)).T[::-1] # normalized energy, polarity follows Slepian convention
        if eigenvalue:
            A = toeplitz(np.insert( np.sin(2*np.pi*self.W*np.arange(1,self.N))/(np.pi*np.arange(1,self.N)), 0, 2*self.W ))
            self.vals = np.diag(self.vecs @ A @ self.vecs.T) # @ is matrix multiplication

    def plot_sequences(self):
        plt.close("all")
        fig, ax = plt.subplots()
        index = np.arange(self.N)
        for val, vec in zip(self.vals, self.vecs):
            ax.plot(index, vec, label="{:.7f}".format(val))
        ax.axhline(color='k')
        ax.legend()
        ax.set_xlim([index.min(), index.max()])
        ax.set_xlabel("index")
        ax.set_ylabel("amplitude")
        ax.set_title(r"$NW$ = {:d}".format(int(self.N*self.W)))
        plt.tight_layout(.5)
        plt.show()

    def gen_spectra(self, n_point=None):
        '''
        calculate the corresponding Fourier transforms in the frequency domain
        '''
        if n_point is None:
            n_point = int( np.power(2, np.ceil(np.log2(self.N))) )
        dummy = pyfftw.empty_aligned((self.K, self.N))
        fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=self.n_thread)
        return fft(self.vecs)

    def plot_spectra(self, spectra):
        n_point = spectra.shape[1]
        spectra = np.absolute(spectra[:,:(n_point+1)//2])**2 # positive half, converted to power
        plt.close("all")
        fig, ax = plt.subplots()
        frequency = np.arange((n_point+1)//2) / n_point
        for val, spectrum in zip(self.vals, spectra):
            ax.fill_between(frequency, spectrum, alpha=.5, label="{:.7f}".format(val))
        ax.fill_between(frequency, np.mean(spectra, axis=0), alpha=.5, label="average")
        ax.axvline(self.W, color='k')
        ax.legend()
        ax.set_yscale("log")
        ax.set_xlim([0, .5])
        ax.set_ylim(ymin=1e-6)
        ax.set_xlabel("frequency")
        ax.set_ylabel("power")
        ax.set_title(r"$NW$ = {:d}".format(int(self.N*self.W)))
        plt.tight_layout(.5)
        plt.show()


if __name__ == "__main__":
    dpss = DPSS(32, 4, 8, True)
    dpss.plot_sequences()
    dpss.plot_spectra(dpss.gen_spectra(1024))
