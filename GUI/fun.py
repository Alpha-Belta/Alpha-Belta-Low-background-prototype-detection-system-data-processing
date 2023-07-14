# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:19:58 2023

@author: Yuanfei Cheng
"""
import numpy as np
from scipy.integrate import quad, cumulative_trapezoid

import matplotlib.pyplot as plt

def first_order_filter(data, alpha):
    filtered_data = [data[0]]
    for i in range(1, len(data)):
        filtered_data.append(alpha * data[i] + (1 - alpha) * filtered_data[-1])
    return filtered_data



def energy_plotting_integrate(energy): #计算事例能损
    e_all=[]
    channel_count=[]
    for i in range(len(energy)):
        temp=0
        channel_count.append(len(energy[i]))
        for j in range(len(energy[i])):
            x=[m for m in range(len(energy[i][j]))]
            cumulative_integral = cumulative_trapezoid(energy[i][j], x)
            temp+=cumulative_integral[-1]
        e_all.append(temp)
    plt.figure(figsize=(10,4),dpi=150)
    #plt.title("能谱")
    n, bins, patches=plt.hist(e_all, bins = 60,label='integrate',alpha=0.5)
    plt.title("能谱")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    a1=FWHM(n,bins)
    plt.text(0, 200, 'FWHM='+str(a1), fontsize=12, color='red')
    plt.legend()
    plt.show() 
    return(n,e_all,channel_count)
    #plt.show() 


def single_energy__integrate(energy): #计算单个通道能损
    x=[m for m in range(len(energy))]
    cumulative_integral = cumulative_trapezoid(energy, x)
    temp=cumulative_integral[-1]
    return temp


def FFT(signal):
    t = [m for m in range(len(signal))]
    freqs = np.fft.fftfreq(len(t))
    fft = np.fft.fft(signal)
    #return(np.abs(fft))
    # 绘制频谱图
    plt.figure()
    plt.plot(freqs, np.abs(fft))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()
    print('finish!')

def FWHM(n,bins):
    max_height = np.max(n)
    max_pos = np.argmax(n)

     # 计算直方图的半高位置和对应的半高值
    half_max = max_height / 2
    for i in range(max_pos, len(n)):
        if n[i] < half_max:
            left_pos = bins[i-1]
            right_pos = bins[i]
            left_val = n[i-1]
            right_val = n[i]
            break
    # 计算半高宽
    #half_width = (right_pos - left_pos) / (right_val - left_val) * (half_max - left_val)
    half_width = (right_pos - left_pos) /( bins[max_pos])
    print(right_pos,left_pos,bins[max_pos])
    print(half_width)
    return(round(half_width,2))

            