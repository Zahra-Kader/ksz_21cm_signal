import numpy as np
import matplotlib.pyplot as plt
import useful_functions as uf
import sys
sys.path.insert(0,'/home/zahra/hirax_tools/hirax_tools')
from array_config import HIRAXArrayConfig
import scipy as sp
from scipy.interpolate import interp1d,spline,UnivariateSpline
from mpl_toolkits.mplot3d import Axes3D
import pylab
import constants as cc
import density as den




chi=uf.chi
r=uf.r

D=uf.D_1
H=uf.H
f=uf.f
c=cc.c_light_Mpc_s
nu_21=1420. #MHz
z=1.26
n=1000
delta_z=0.05

def HiraxNoise(l,Ddish,Dsep,zi,delta_z): #the Dsep refers to the dish seperation including the dish diameter of 6. So we're assuming dishes are 1m away
        z_min=zi-delta_z
        z_max=zi+delta_z
        nu_min=nu_21/(1+z_max) #743. #585. # #freq in MHz, these min and max values are based on HIRAX
        nu_max=nu_21/(1+z_min)#784. #665. #
        cp= HIRAXArrayConfig.from_n_elem_compact(1024,Dsep)
        nubar=(nu_min+nu_max)/2.

        Fov_deg=(cp.fov(frequency=nubar)) * ((180./np.pi)**2)
        Fov_str=(cp.fov(frequency=nubar))
        Tsys= 50. + 60.*((nubar/300.)**-2.5)
        lam=3.e8/(nubar*1.e6)
        Nbeam=1.
        npol=2.
        nu21=1420.e6
        Aeff=np.pi*((Ddish/2.)**2)*0.67
        Ttot=100.8e6 #36e6   4*365*24*3600.
        #Ttot=2*365*24*3600
        Sarea=15000.
        pconv=(chi(zi)**2)*r(zi)
        n_u=cp.baseline_density_spline(frequency=nubar)
        n=np.array([])
        for i in l:
            if n_u(i/(2.*np.pi))==0:
                n_u1=1/1e100
            else:
                n_u1=n_u(i/(2.*np.pi))
            n=np.append(n,n_u1)
        Nbs= 1024.*(1024.-1.)
        #norm=Nbs/sp.integrate.trapz(n*2*np.pi*(l/(2.*np.pi)), l/(2.*np.pi))
        C=n#*noprint (nu_min)
        A_bull= ((Tsys**2)*(lam**4)*Sarea) / (nu21*npol*Ttot*(Aeff**2)*Fov_deg*Nbeam)
        Warren=Tsys**2*lam**2/Aeff*4.*np.pi/(nu21*npol*Nbeam*Ttot)
        #return (Warren/C)*1e12
        return (A_bull/C)*1e12

ell_large=np.linspace(1,10000,100000)
y_large=np.linspace(1,1000,100000)

ell=np.linspace(1,5000,n)
y=np.linspace(200,5000,n)


#np.savetxt('Hirax_noise_z_1_Ddish_6_Dsep_7_geom.out',(ell_large,HiraxNoise(ell_large,6.,7.,1.)))
#ell_new,Hirax_noise_z_1=np.genfromtxt('/home/zahra/python_scripts/kSZ_21cm_signal/Hirax_noise_z_1_Ddish_6_Dsep_7.out')
Hirax_noise_z_1pt26_deltaz_pt2 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.26,0.2), bounds_error=False)
Hirax_noise_z_1pt26_deltaz_pt05 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.26,0.05), bounds_error=False)
Hirax_noise_z_1pt26_deltaz_pt05_dist_18 = interp1d(ell_large, HiraxNoise(ell_large,6.,18.,1.26,0.05), bounds_error=False)
Hirax_noise_z_1pt26_deltaz_pt05_dist_11 = interp1d(ell_large, HiraxNoise(ell_large,6.,11.,1.26,0.05), bounds_error=False)


Hirax_noise_z_1pt11_deltaz_pt05 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.11,0.05), bounds_error=False)
Hirax_noise_z_1pt21_deltaz_pt05 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.21,0.05), bounds_error=False)
Hirax_noise_z_1pt31_deltaz_pt05 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.31,0.05), bounds_error=False)
Hirax_noise_z_1pt41_deltaz_pt05 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.41,0.05), bounds_error=False)

SKA_noise_dish_2d=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_26.dat',
                   dtype=float,unpack=True)
SKA_noise_interferom_2d=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_26.dat',
                dtype=float,unpack=True)

ell_dish=SKA_noise_dish_2d[:,0]
SKA_noise_dish=SKA_noise_dish_2d[:,1]
ell_interferom=SKA_noise_interferom_2d[:,0]
SKA_noise_interferom=SKA_noise_interferom_2d[:,1]

SKA_noise_dish_interp = interp1d(ell_dish, SKA_noise_dish) #original values are in mK^2 so we convert to microK^2
#SKA_noise_dish_spline=UnivariateSpline(ell_dish, SKA_noise_dish)
#SKA_noise_dish_spline.set_smoothing_factor(0.5)
SKA_noise_interferom_interp=interp1d(ell_interferom, SKA_noise_interferom)
#SKA_noise_interferom_spline=UnivariateSpline(ell_interferom, SKA_noise_interferom)

SKA_noise_total_interp=interp1d(ell_interferom,1./(1./(1.e6*SKA_noise_dish_interp(ell_interferom))+1./(1.e6*SKA_noise_interferom_interp(ell_interferom))))
#SKA_noise_total_spline=UnivariateSpline(ell_interferom,1./(1./(1.e6*SKA_noise_dish_interp(ell_interferom))+1./(1.e6*SKA_noise_interferom_interp(ell_interferom))))


kperp_arr=np.linspace(1.e-3,1.,n)
ell=kperp_arr*chi(z)

plt.loglog(kperp_arr,HiraxNoise(ell,6.,18.,1.26,0.05))
plt.ylim(1e-5,100)

plt.show()
