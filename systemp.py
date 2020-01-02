# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:20:55 2019

@author: zahra
"""

import matplotlib.pyplot as plt
import hirax_tools as ht
from hirax_tools.utils import pointing
import numpy as np
import scipy as sp
from astropy import units
from astropy import coordinates as coords
from matplotlib import dates



rd = ht.RawData('C:/Users/zahra/Downloads/01891349_0000.h5', transpose_on_init=True) #uncomment this when opening a new console
'''
unix_times=rd.times
from astropy.time import Time
times = Time(unix_times, format='unix')

hartrao = coords.EarthLocation.from_geodetic(
    lon=27.6853931*units.deg,
    lat=-25.8897515*units.deg,
    height=1415.710*units.m,
    ellipsoid='WGS84')

alt, az = (90-17.1)*units.deg, 180*units.deg 
altazs = coords.SkyCoord(frame='altaz', alt=alt*np.ones(len(times)),az=az*np.ones(len(times)),
                         location=hartrao, obstime=times)
j2000_pointing = altazs.transform_to(coords.ICRS)

fornax = coords.SkyCoord(ra='03h22m41.7s', dec='-37d12m30s')
cen=coords.SkyCoord(ra='13h25m27.6s', dec='-43d01m11s')
pointing_separation = j2000_pointing.separation(cen)

num_dates = dates.date2num(times.to_datetime())

plt.plot(pointing_separation)
plt.show()


T_source=10**7 #This is a fake value at the moment
'''
r=[100,200,300,400,500,590,600,700,800]

x=pointing(rd.times, 79.8, 180, frame='Galactic').b
x=x.deg

for i in r:
    band_ind=i
    print (i)
    filtered = rd.filtered_on_time((3, 3), filter_type='low', char_freqs=1/0.5)
    y=np.abs(filtered[:, band_ind])
    plt.plot(x,y)
    plt.show()



band_ind=500
freq_MHz=rd.bands[band_ind]
print (freq_MHz)


def signal(band_ind):
    filtered = rd.filtered_on_time((3, 3), filter_type='low', char_freqs=1/0.5)
    y=np.abs(filtered[:, band_ind])
    return y

plt.plot(x,signal(band_ind))


y=signal(band_ind)
mean = sum(x * y) / sum(y)
sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
const=1.8e7

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))+const

popt,pcov = sp.optimize.curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])
print (np.sqrt(np.diagonal(pcov)))
Y=max(Gauss(x,*popt))-const
#print (Y)
#T_sys=T_source/(Y-1)
#print (T_sys,'Tsys')


    

plt.title('Signal amplitude as a function of galactic latitude for 527 MHz')
plt.plot(x, Gauss(x, *popt), 'r-', label='fit')
plt.xlabel('Galactic latitude (degrees)')
plt.ylabel('Signal Amplitude')
plt.show()

#print(rd.bands[band_ind])
'''
x=np.array([527,566,605,644,682,722,761])
y=np.array([2.75,2.79,2.25,2.36,3.35,3.44,3.75])
plt.scatter(x,y)
plt.xlabel('Frequency (MHz)')
plt.ylabel('Tsys (do not trust the values)')
'''