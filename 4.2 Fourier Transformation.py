# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 22:24:26 2019

@author: Dell
"""
import numpy as np
import math
import matplotlib.pyplot as plt

## The DTFT in loop-form

# create the signal
srate  = 1000 # hz
time   = np.arange(0.,2.,1/srate) # time vector in seconds
pnts   = len(time) # number of time points
signal = 2.5 * np.sin( 2*np.pi*4*time ) + 1.5 * np.sin( 2*np.pi*6.5*time )


# prepare the Fourier transform
fourTime = np.array(range(0,pnts))/pnts
fCoefs   = np.zeros((len(signal)),dtype=complex)

for fi in range(0,pnts):
    
    # create complex sine wave
    csw = np.exp( -1j*2*np.pi*fi*fourTime )
    
    # compute dot product between sine wave and signal
    # these are called the Fourier coefficients
    fCoefs[fi] = np.sum( np.multiply(signal,csw) ) / pnts


# extract amplitudes
ampls = 2*np.abs(fCoefs)

# compute frequencies vector
hz = np.linspace(0,srate/2,num=math.floor(pnts/2.)+1)

plt.stem(hz,ampls[range(0,len(hz))])
plt.xlabel('Frequency (Hz)'), plt.ylabel('Amplitude (a.u.)')
plt.xlim(0,10)
plt.show()


#####################DFT##############################################3
import mysignals as sigs
from matplotlib import pyplot as plt
from matplotlib import style
import math


import mysignals as sigs
from matplotlib import pyplot as plt
from matplotlib import style
import math


def calc_dft(sig_src_arr):
    sig_dest_imx_arr = [None]*int((len(sig_src_arr)/2))
    sig_dest_rex_arr = [None]*int((len(sig_src_arr)/2))
    sig_dest_mag_arr = [None]*int((len(sig_src_arr)/2))
    
    for j in range(int(len(sig_src_arr)/2)):
        sig_dest_rex_arr[j] =0
        sig_dest_imx_arr[j] =0

    for k in range(int(len(sig_src_arr)/2)):
        for i in range(len(sig_src_arr)):
            sig_dest_rex_arr[k] = sig_dest_rex_arr[k] + sig_src_arr[i]*math.cos(2*math.pi*k*i/len(sig_src_arr))
            sig_dest_imx_arr[k] = sig_dest_imx_arr[k] - sig_src_arr[i]*math.sin(2*math.pi*k*i/len(sig_src_arr))

    for x in range(int(len(sig_src_arr)/2)):
        sig_dest_mag_arr[x] = math.sqrt(math.pow(sig_dest_rex_arr[x],2)+math.pow(sig_dest_imx_arr[x],2))
        
    

    style.use('ggplot')
    f,plt_arr = plt.subplots(4, sharex=True)
    f.suptitle("Discrete Fourier Transform (DFT)")

    plt_arr[0].plot(sig_src_arr, color='red')
    plt_arr[0].set_title("Input Signal",color='red')
    
    plt_arr[1].plot(sig_dest_rex_arr, color='green')
    plt_arr[1].set_title("Frequency Domain(Real part)",color='green')

    plt_arr[2].plot(sig_dest_imx_arr, color='green')
    plt_arr[2].set_title("Frequency Domain(Imaginary part)",color='green')

    plt_arr[3].plot(sig_dest_mag_arr, color='magenta')
    plt_arr[3].set_title("Frequency Domain (Magnitude))",color='magenta')

    plt.show()


####################################FFT#################################################################
  import mysignals as sigs
from matplotlib import pyplot as plt
from scipy.fftpack import fft,ifft
import numpy as np
from matplotlib import style

freq_domain_signal = fft(sigs.ecg_signal)
time_domain_signal = ifft(freq_domain_signal)
magnitude = np.abs(freq_domain_signal)

style.use('dark_background')

f,plt_arr= plt.subplots(4,sharex=True)
f.suptitle("Fast Fourier Transform (FFT)")

plt_arr[0].plot(sigs.ecg_signal,color='red')
plt_arr[0].set_title("Time Domain (Input Signal)", color ='red')

plt_arr[1].plot(freq_domain_signal,color='cyan')
plt_arr[1].set_title("Frequency Domain (FFT)", color ='cyan')

plt_arr[2].plot(magnitude,color='cyan')
plt_arr[2].set_title("Magnitude", color ='cyan')

plt_arr[3].plot(time_domain_signal,color='green')
plt_arr[3].set_title("Time Domain (IFFT)", color ='green')

plt.show()
