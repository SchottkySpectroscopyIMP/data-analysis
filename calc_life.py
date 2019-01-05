#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import sys, os, json, time

from processing import Processing

class LifeTime():
    def __init__(self, file_str, window_length, n_average):

        self.file_folder = "/media/qwang/Seagate Backup Plus Drive/2018-12/"
        self.file_str = file_str
            
        self.window_length = window_length
        self.n_average = n_average
        self.bud = Processing(self.file_folder + self.file_str, verbose=True)
        self.n_buffer = 26214400

    def diagnosis(self, n_point=None, offset=0):
        if n_point is None or n_point < self.window_length * self.n_average:
            frame = self.n_buffer // (self.window_length * self.n_average)
        elif n_point < -1:
            frame = -1
        else:
            frame = n_point // (self.window_length * self.n_average)
        if offset > self.bud.n_sample:
            offset = 0
        
        frequencies, times, spectrogram, n_dof = self.bud.time_average_2d(window_length = self.window_length, n_frame = frame, n_offset = offset, padding_ratio = 2, n_average = self.n_average, estimator = 'p')
        
        plt.close("all")
        fig, ax = plt.subplots()
        color_style = "viridis"
        pcm = ax.pcolormesh(frequencies, times, spectrogram, cmap=color_style)
        cax = fig.colorbar(pcm, ax=ax)
        cax.set_label("power spectral density [arb. unit]")
        ax.set_title(self.file_str)
        ax.set_xlabel("frequency - {:g} MHz [kHz]".format(self.bud.center_frequency*1e-6))
        ax.set_ylabel("times [s]")
        ax.set_ylim([times[0],times[-1]])
        plt.show()

    def load_data(self, freq_min="", freq_max=""):
        sample_num = self.bud.n_sample
        frame = self.n_buffer // (self.window_length * self.n_average) 
        self.freq_min = freq_min if freq_min != "" else - self.bud.span / 2 * 1e-3
        self.freq_max = freq_max if freq_max != "" else self.bud.span / 2 * 1e-3

        header_meta = {
            'window length':    self.window_length,
            'average':          self.n_average,
            'center frequency': self.bud.center_frequency,
            'frequency min':    self.freq_min,
            'frequency max':    self.freq_max
        }

        def processing_func(setting):
            n_frame = setting[0]
            n_offset = setting[1]
            frequencies, times, spectrogram, n_dof = self.bud.time_average_2d(window_length = self.window_length, n_frame = setting[0], n_offset = setting[1], padding_ratio = 2, n_average = self.n_average, estimator = 'p')
            arg_min = np.searchsorted(frequencies, self.freq_min, side="left")
            arg_max = np.searchsorted(frequencies, self.freq_max, side="right")
            time = (times[:-1] + times[1:]) / 2
            peakArea = np.sum(spectrogram[:,arg_min:arg_max], axis=1) * self.bud.sampling_rate * 1e-3
            life_data = pd.DataFrame(np.transpose([time,peakArea]), columns=['time', 'peakArea'])
            life_data.to_csv("life_" + self.file_str.split(".")[0] + ".csv", mode='a', header=False, columns=['time', 'peakArea'], index=False)


        if (os.path.exists("life_" + self.file_str.split(".")[0] + ".csv")):
            os.remove("life_" + self.file_str.split(".")[0] + ".csv")

        with open("life_" + self.file_str.split(".")[0] + ".csv", 'w') as header:
            json.dump(header_meta, header, indent=4, sort_keys=True)
            header.write("\n") 
        
        life_data = pd.DataFrame({'time': [], 'peakArea': []})
        life_data.to_csv("life_" + self.file_str.split(".")[0] + ".csv", mode='a', header=True, columns=['time', 'peakArea'], index=0)

        init_time = time.time()
        k = 0
        while True:
            offset = self.window_length * self.n_average * frame * k
            sample_num -= self.window_length * self.n_average * frame 
            if sample_num >= 0:
                processing_func([frame, offset])
            else:
                processing_func([-1, offset])
                break
            k += 1
        
        print("information of data load:\n--------------------")
        print("data load time\t\t\t{:} s".format(time.time()-init_time))
        print("window length\t\t\t{:d}".format(self.window_length))
        print("average\t\t\t\t{:d}".format(self.n_average))
        print("center frequency\t\t{:g} MHz".format(self.bud.center_frequency * 1e-6))
        print("frequency start\t\t\t{:} kHz".format(self.freq_min))
        print("frequency end\t\t\t{:} kHz".format(self.freq_max))
        print("--------------------")        

    def analyze_data(self, time_min="", time_max="", fit=False, Method="exp"):
        if (os.path.exists("life_" + self.file_str.split(".")[0] + ".csv") == False):
            print("No availble .csv file. Please load the file first!")
            return

        with open("life_" + self.file_str.split(".")[0] + ".csv", "r") as header:
            lines = ''
            for i in range(7):
                line = header.readline()
                lines += line
            header_meta = json.loads(lines)
            life_data = pd.read_csv(header)
        window_length = header_meta['window length']
        n_average = header_meta['average']
        center_frequency = header_meta['center frequency']
        freq_min = header_meta['frequency min']
        freq_max = header_meta['frequency max']

        print("information of data:\n--------------------")
        print("window length\t\t\t{:d}".format(window_length))
        print("average\t\t\t\t{:d}".format(n_average))
        print("center frequency\t\t{:g} MHz".format(center_frequency * 1e-6))
        print("frequency start\t\t\t{:} kHz".format(freq_min))
        print("frequency end\t\t\t{:} kHz".format(freq_max))
        print("--------------------")        

        time = life_data['time'].values
        peakArea = life_data['peakArea'].values

        index_min = np.searchsorted(time, time_min, side="left") if time_min != "" else 0
        index_max = np.searchsorted(time, time_max, side="right") if time_max != "" else len(time)
        
        time = time[index_min:index_max]
        peakArea = peakArea[index_min:index_max]
        
        
        plt.close("all")
        plt.figure()
        plt.plot(time, peakArea, 'bo')
        if fit:
            if Method == "lin":
                p0 = [-1/time[int(len(time)/2)] * np.log(2), np.log(np.max(time) - np.min(time))]
                half_life, curve_fit = self.fitting(p0, time, np.log(peakArea), "lin")
                plt.plot(time, np.exp(curve_fit), 'r', lw=2)
            else:
                p0 = [np.max(time) - np.min(time), -1/time[int(len(time)/2)] * np.log(2), np.min(time)]
                half_life, curve_fit = self.fitting(p0, time, peakArea, "exp")
                plt.plot(time, curve_fit, 'r', lw=2)
            print("half-life: {:} s".format(half_life))
        plt.xlabel("times [s]")
        plt.ylabel("area [arb. unit]")
        plt.title("decay curve")
        plt.show()

    def fitting(self, p0, x, y, Method="exp"):
        def test_func_exp(x, a, b, c):
            return a * np.exp(b * x) + c

        def test_func_lin(x, b, d):
            return b * x + d

        if Method == "exp":
            params, params_convariance = optimize.curve_fit(test_func_exp, x, y, p0=p0)
            half_life = -np.log(2) / params[1]
            return half_life, test_func_exp(x, *params)
        else:
            params, params_convariance = optimize.curve_fit(test_func_lin, x, y, p0=p0)
            half_life = -np.log(2) / params[0]
            return half_life, test_func_lin(x, *params)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: {} path/to/file".format(__file__))
        sys.exit()
    lifetime = LifeTime(sys.argv[-1], window_length=1000, n_average=100) 
    lifetime.diagnosis()
    #lifetime.diagnosis(n_point=None,offset=0)
    #lifetime.load_data(freq_min="", freq_max="")       
    #lifetime.load_data(freq_min=-10, freq_max=10)       
    #lifetime.analyze_data(time_min="", time_max="", fit=False, Method="exp")       
    #lifetime.analyze_data(time_min=25, time_max=600, fit=True, Method="exp")       
