#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import json, sys, os.path

class Preprocessing(object):
    '''
    A class for either reading metadata from a .wvh file and importing segments of data from the corresponding .wvd file
    or reading metadata from a .tiq file and importing segments of data from the same file
    An auxiliary function is implemented to inspect the read data in the time domain mainly for signal's amplitude overflow-check
    '''

    n_buffer = 10**5 # maximum number of IQ pairs to be loaded at one time

    def __init__(self, file_str, verbose=True):
        '''
        file_str is a string describing the location of the .wvh file or .wvd file
        '''
        file_abs = os.path.abspath(file_str)
        self.fname = os.path.basename(file_abs)
        self.fpath = os.path.dirname(file_abs)
        if self.fname[-3:].lower() in ["wvh", "wvd"]:
            self.file_format = "wv"
            self.extract_wv()
        elif self.fname[-3:].lower() == "tiq":
            self.file_format = "tiq"
            self.extract_tiq()
        else:
            print("Error: unrecognized file format!\nOnly 'wv' or 'tiq' file can be accepted.")
            sys.exit()
        if verbose:
            self.display()

    def extract_wv(self):
        '''
        extract the metadata from the .wvh file
        '''
        with open('/'.join((self.fpath, self.fname[:-3]+"wvh"))) as wvh:
            metadata = json.load(wvh)
        self.date_time = np.datetime64(metadata["timestamp"])
        self.data_format = np.dtype(metadata["format"]).newbyteorder(metadata["endian"])
        self.ref_level = metadata["reference level"] # dBm
        self.center_frequency = metadata["center frequency"] # Hz
        self.span = metadata["span"] # Hz
        self.sampling_rate = metadata["sampling rate"] # Hz
        self.n_sample = metadata["number of samples"]
        self.digitizing_depth = metadata["resolution"] # bit
        self.duration = metadata["duration"] # s

    def extract_tiq(self):
        '''
        extract the metadata from the .tiq file
        '''
        with open('/'.join((self.fpath, self.fname))) as tiq:
            self.n_offset = int(tiq.readline().split('"')[1])
            prefix = '{' + tiq.readline().split('"')[1] + '}'
        with open('/'.join((self.fpath, self.fname)), "rb") as tiq:
            root = ET.fromstring(tiq.read(self.n_offset))
        def get_value(key):
            return next(root[0][0].iter(prefix + key)).text
        self.date_time = np.datetime64(get_value("DateTime"))
        self.data_format = np.dtype(get_value("NumberFormat")).newbyteorder(get_value("Endian"))
        self.ref_level = float(get_value("ReferenceLevel")) # dBm
        self.center_frequency = float(get_value("Frequency")) # Hz
        self.span = float(get_value("AcquisitionBandwidth")) # Hz
        self.sampling_rate = float(get_value("SamplingFrequency")) # Hz
        self.n_sample = int(get_value("NumberSamples"))
        self.scaling = float(get_value("Scaling"))
        self.trig_pos = float(get_value("TriggerPosition")) # s

    def display(self):
        '''
        display all the parameters as a list
        '''
        print("list of information:\n--------------------")
        print("file format\t\t\t" + self.file_format)
        print("name of file\t\t\t" + self.fname)
        print("path to file\t\t\t" + self.fpath)
        print("timestamp in UTC \t\t" + str(self.date_time))
        print("data format\t\t\t" + repr(self.data_format))
        print("reference level\t\t\t{:g} dBm".format(self.ref_level))
        print("center frequency\t\t{:g} MHz".format(self.center_frequency*1e-6))
        print("span\t\t\t\t{:g} kHz".format(self.span*1e-3))
        print("sampling rate\t\t\t{:g} kHz".format(self.sampling_rate*1e-3))
        print("number of samples\t\t{:d} IQ pairs".format(self.n_sample))
        print("recording duration (actual)\t{:g} s".format(self.n_sample/self.sampling_rate))
        print("--------------------")
        try: # for wv
            print("digitizing depth\t\t{:d} bits".format(self.digitizing_depth))
            print("recording duration (set)\t{:g} s".format(self.duration))
        except AttributeError: # for tiq
            print("scaling\t\t\t\t{:g}".format(self.scaling))
            print("trigger position\t\t{:g} s".format(self.trig_pos))
        print("--------------------")

    def load(self, size, offset, decimating_factor=1, draw=False):
        '''
        size:               amount of IQ pairs to be imported
        offset:             amount of IQ pairs to be skipped over
        decimating_factor:  an positive integer by which the data are decimated, the default value 1 means no downsampling
        '''
        size = (self.n_sample-offset)//decimating_factor if size*decimating_factor+offset > self.n_sample else size # crop the excessive request
        times = (np.arange(size)*decimating_factor + offset) / self.sampling_rate # s
        if self.file_format == "wv":
            wvd = np.memmap('/'.join((self.fpath, self.fname[:-3]+"wvd")), dtype=self.data_format, offset=offset*2*self.data_format.itemsize, mode='r')
            data = wvd[:2*size*decimating_factor].reshape(size,2*decimating_factor)[:,:2].flatten().astype("float64").view("complex128") / (2**(self.digitizing_depth-1) - 1) # V
        else: # for tiq
            tiq = np.memmap('/'.join((self.fpath, self.fname)), dtype=self.data_format, offset=self.n_offset+offset*2*self.data_format.itemsize, mode='r')
            data = tiq[:2*size*decimating_factor].reshape(size,2*decimating_factor)[:,:2].flatten().astype("float64").view("complex128") * self.scaling # V
        if draw:
            self.draw(times, data)
        else:
            return times, data # s, V

    def draw(self, t, signal):
        plt.close("all")
        fig, (axr, axi) = plt.subplots(2, 1, sharex=True, sharey=True)
        axr.plot(t, np.real(signal))
        axr.set_ylabel("in phase")
        axr.set_title(self.fname)
        axi.plot(t, np.imag(signal))
        axi.set_xlim([t.min(), t.max()])
        axi.set_xlabel("time [s]")
        axi.set_ylabel("quadrature")
        plt.show()

    def diagnosis(self, n_point=None, draw=True):
        '''
        plot all the data, after downsampling if necessary, in the time domain
        n_point:    maximum data points in the plot, if omitted, n_buffer is replaced in
                    negative means all samples without downsampling
        '''
        if n_point is None:
            n_point = self.n_buffer
        if n_point < 0:
            offset = 0
            while self.n_sample > offset+self.n_buffer:
                self.load(self.n_buffer, offset, 1, draw)
                offset += self.n_buffer
            self.load(self.n_buffer, offset, 1, draw)
        else:
            decimating_factor = self.n_sample // n_point + 1
            return self.load(n_point, 0, decimating_factor, draw)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} path/to/file".format(__file__))
        sys.exit()
    preprocessing = Preprocessing(sys.argv[-1])
    preprocessing.diagnosis()
