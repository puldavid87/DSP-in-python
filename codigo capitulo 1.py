# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:12:50 2019

@author: Dell
"""
"""
Capítulo: Introudcción de DSP
Programa: Generación de señales
Descripción: Introducción de procesamiento digital de señales orientado a la generación de señales continuas y discretas
Librerías a utilizar: numpy y pylab
"""
#importar librerías
#as es un forma reducción del llamado de librería en el resto de funciones
import numpy as np 
import pylab as plt

#Pasos para generar una señal 
# 𝑥_1 (𝑡)=𝐴𝑐𝑜𝑠(𝜔𝑡+𝜃)

srate = 500# muestras totales sobre 2
time_1  = np.arange(0.,2.,1/srate) # muestras  (incio, fin, cuantos valores se encesita)
time_1
# elegir parámetros
freq = 3;    # frequencia en Hertz
ampl = 2;    # amplitud
phas = np.pi/freq; # fase in radianes

# generacionde señal
sinewave_1 = ampl * np.sin( 2*np.pi * freq * time_1 + phas )

#########################otra forma de generar la señal###########################################################

time_2=np.linspace(0,2.0,1000) #tiempo en segundos (incio, fin, frecuencia)
time_2

# elegir parámetros
freq = 5;    # frequencia en Hertz
ampl = 2;    # amplitud
sinewave_2=ampl*np.sin(2*np.pi*time_2*freq)

#generar la gráfica
plt.plot(sinewave_1,label='sinewave to 3Hz with phase', color='red') 
plt.plot(sinewave_2,label='sinewave to 5Hz', color='blue')
plt.legend(loc='best')
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude (a.u.)')
plt.show()