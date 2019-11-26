# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:24:59 2019

@author: Dell
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import pylab as pl
from IPython import display
import time as ttime
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr as pr
# writing the complex number as real + imaginary
z = 4 + 1j*3
print(z)

# using the function complex
z = complex(4,3)    # this way
z = complex('4+3j') # or this way
print(z)


plt.plot(np.real(z),np.imag(z),'ro')

# some plotting touch-ups
plt.axis('square')
plt.axis([-5, 5, -5, 5])
plt.grid(True)
plt.xlabel('Real axis'), plt.ylabel('Imaginary axis')
plt.show()

# compute the magnitude of the complex number using Pythagorean theorem
mag = np.sqrt( np.real(z)**2 + np.imag(z)**2 )

# or using abs
mag = np.abs(z)

print( 'The magnitude is',mag )

# compute the angle of the complex number using Pythagorean theorem
mag = math.atan( np.imag(z) / np.real(z) )

# or using abs
phs = np.angle(z)

print( 'The angle is',phs )

#Euler formula
x = np.linspace(-3,3,num=50)

plt.plot(x,np.exp(x),label='y=e^x')

# some plotting touch-ups
plt.axis([min(x),max(x),0,np.exp(x[-1])])
plt.grid(True)
plt.legend()
plt.xlabel('x')
plt.show()


# [cos(k),sin(k)] is on the unit circle for any real k

# define k (any real number)
k = 2/np.pi

# Euler's notation
euler = np.exp(1j*k)

# plot dot
plt.plot(np.cos(k),np.sin(k),'ro')

# draw unit circle for reference
x = np.linspace(-np.pi,np.pi,num=100)
plt.plot(np.cos(x),np.sin(x))

# some plotting touch-ups
plt.axis('square')
plt.grid(True)
plt.xlabel('Real axis'), plt.ylabel('Imaginary axis')
plt.show()

#Complex sine wave
# now show in 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time,np.real(csw),np.imag(csw))
ax.set_xlabel('Time (s)'), ax.set_ylabel('Real part'), ax.set_zlabel('Imag part')
ax.set_title('Complex sine wave in all its 3D glory')
plt.show()



# two vectors
v1 = [ 1, 7, 5 , 1,4,0,6,8,1,8]
v2 = [ 10, 85,35,15,55,5,72,81,13,92 ]

# compute the dot product
dp = sum( np.multiply(v1,v2) )/1000

print('The dot product is',dp)

#pearson correlation
corr, dp1=pr(v2,v2)
corr

# dot products of sine waves

# general simulation parameters
srate = 500; # sampling rate in Hz
time  = np.arange(0.,2.,1./srate) # time in seconds

# sine wave parameters
freq1 = 5;    # frequency in Hz
freq2 = 5;    # frequency in Hz

ampl1 = 2;    # amplitude in a.u.
ampl2 = 2;    # amplitude in a.u.

phas1 = np.pi/2; # phase in radians
phas2 = np.pi/2; # phase in radians

# generate the sine wave
sinewave1 = ampl1 * np.sin( 2*np.pi * freq1 * time + phas1 );
sinewave2 = ampl2 * np.sin( 2*np.pi * freq2 * time + phas2 );

# compute dot product
dp = np.dot(sinewave1,sinewave2);

# print result
print('dp =',dp)

# with a signal

# phase of signal
theta = 0*np.pi/4;


# simulation parameters
srate = 1000;
time  = np.arange(-1.,1.,1./srate)

# signal
sinew  = np.sin(2*np.pi*5*time + theta)
gauss  = np.exp( (-time**2) / .1);
signal = np.multiply(sinew,gauss)

# sine wave frequencies
sinefrex = np.arange(2.,10.,.5);

# plot signal
plt.plot(time,signal)
plt.xlabel('Time (sec.)'), plt.ylabel('Amplitude (a.u.)')
plt.title('Signal')
plt.show()



# initialize dot products vector
dps = np.zeros(len(sinefrex));

# loop over sine waves
for fi in range(1,len(dps)):
    
    # create sine wave
    sinew = np.sin( 2*np.pi*sinefrex[fi]*time)
    
    # compute dot product
    dps[fi] = np.dot( sinew,signal ) / len(time)


# and plot
plt.stem(sinefrex,dps)
plt.xlabel('Sine wave frequency (Hz)'), plt.ylabel('Dot product')
plt.title('Dot products with sine waves')
plt.show()