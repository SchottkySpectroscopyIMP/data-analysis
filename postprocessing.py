#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import sys
import numpy as np
from scipy.linalg import circulant, svd
from processing import Processing

class Postprocessing(Processing):
    '''
    A class for smoothing of a generic ordered sequence of data set.
    '''


    def __init__(self, file_str):
        super().__init__(file_str)
        frequencies, spectrum = super().periodogram_1d(padding_ratio=1)[:-1]
        index_l = np.argmin(np.abs(frequencies+self.span/2e3))
        index_r = np.argmin(np.abs(frequencies-self.span/2e3)) + 1
        spectrum = spectrum[index_l:index_r]
        self.spectrum = np.log10(spectrum) + 9

    def denoising(self, sequence, alpha=.05):
        '''
        a function for smoothing a noisy (real) data sequence by removing the noise components
        arguments:
            sequence:   the noisy data sequence
            alpha:      the thresholding parameter to separate the signal subspace from the noise subspace
        returns:
            denoised:   the denoised sequence
            stdev:      the standard deviation of the denoised sequence
            s:          the singular values ordered decreasingly
            s1:         the first derivative of s
            threshold:  the absolute threshold according to the given alpha
            index:      the index in s delimiting two subspaces
        '''
        # center the sequence before raising the dimensionality
        shift = np.mean(sequence)
        X = circulant(sequence-shift)
        # singular value decomposition
        U, s, Vh = svd(X)
        # subspace separation based on the first derivative thresholding
        s1 = s[:-1] - s[1:]
        threshold = alpha * np.amax(s1)
        index = sequence.size//2 - np.argmax(s1[sequence.size//2::-1]>threshold)
        # noise singular values estimation
        s_noise = np.sqrt(np.mean(s[index+1:]**2))
        stdev = s_noise / np.sqrt(sequence.size)
        s_corrected = s[:index+1] - s_noise
        # signal reconstruction
        A = U[:,:index+1] @ np.diag(s_corrected) @ Vh[:index+1,:]
        A_ext = np.vstack((A, A))
        denoised = np.array([ np.mean(np.diag(A_ext, k)) for k in range(0, -sequence.size, -1) ]) + shift
        return denoised, stdev, s, index, threshold


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} path/to/file".format(__file__))
        sys.exit()
    postprocessing = Postprocessing(sys.argv[-1])
