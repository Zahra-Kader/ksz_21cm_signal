# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 12:53:47 2018

@author: zahra
"""

import distance as cd
from scipy.interpolate import interp1d
import numpy as np
import perturbation as cp
import density as den
import constants as cc
import matplotlib.pyplot as plt
import scipy as sp
import pylab
from matplotlib.colors import LogNorm

#import perturbation as cp

b_HI=1.0
omega_HI=0.8e-3
n_points=100
nu_21=1420.

cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.67, 'omega_b_0' : 0.049, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False,'X_H':.75}
H0=cc.H100_s*cosmo['h']

#z=np.logspace(-10,np.log(2000),2000)
#z=np.linspace(1e-4,10,n_points)
z=np.geomspace(1e-4,10,n_points)

kabs,P= np.genfromtxt('/home/zahra/python_scripts/kSZ_21cm_signal/camb_63347152_matterpower_z0_16000_kmax.dat', dtype=float,
                      unpack=True)
#interpolate the matter power spec
Mps_interpf = interp1d(kabs, P, bounds_error=False,fill_value="extrapolate")

k=np.linspace(1.e-4,10.,10)





Mps_interpf_div_ksq=interp1d(kabs, P/kabs**2, bounds_error=False,fill_value=0.)


def zed(chi_in):
    chi_full = cd.comoving_distance(z, **cosmo)
    f=interp1d(chi_full,z,bounds_error=False,fill_value=0.)
    return f(chi_in)

def chi(z):
    chi_full = cd.comoving_distance(z, **cosmo)
    return chi_full

def H(z):
    H=cd.hubble_z(z,**cosmo)
    return H

def D_1(z):
    D_1=cp.fgrowth(z,cosmo['omega_M_0'],0)
    return D_1

#plt.plot(z,D_1(z))
#plt.show()
chi_m=chi(1100)
chi_array=np.linspace(0,chi_m,2000)
#plt.plot(chi_array,D_1(zed(chi_array)))
#plt.show()

def f(z):
    f=(den.omega_M_z(z,**cosmo))**(cc.gamma)
    return f

#plt.plot(den.omega_M_z(z,**cosmo),f(z))
#plt.show()

def r(z):
    r=cc.c_light_Mpc_s*(1+z)**2/H(z)
    return r

def kpar(y,z):
    kpar=y/r(z)
    return kpar

def T_mean(z):
    T_mean=566.*cosmo['h']*H0*omega_HI*(1+z)**2/(H(z)*0.003) #\mu K, microkelvin
    return T_mean


def kpar_min(z,delta_z):
    z_max=z+delta_z
    z_min=z-delta_z
    nu_min=nu_21/(1+z_max)
    nu_max=nu_21/(1+z_min)
    delta_nu_dimless=(nu_max-nu_min)/nu_21
    return 2.*np.pi/r(z)/delta_nu_dimless

def ell_lims(z,Dmin,Dmax): #D=Dmin for kperp_min and D=Dmax for kperp_max
    nu=nu_21/(1+z)*1.e6
    c_metres=cc.c_light_cm_s/100.
    lam=c_metres/nu
    u_min=Dmin/lam
    u_max=Dmax/lam
    return 2.*np.pi*u_min, 2.*np.pi*u_max


def P_delta_delta(kperp,kpar):
    Kperp,Kpar=np.meshgrid(kperp,kpar)
    k=np.sqrt(Kpar**2+Kperp**2)
    return k**3*Mps_interpf(k)

def P_vv(kperp,kpar,z):
    Kperp,Kpar=np.meshgrid(kperp,kpar)
    k=np.sqrt(Kpar**2+Kperp**2)
    mu_k=Kpar/k
    Pvv=f(z)**2*H(z)**2*Mps_interpf(k)*mu_k**2/((1+z)**2*k**2)/cc.c_light_Mpc_s**2
    return k**3*Pvv
    #return k**3*Mps_interpf(k)/k**4-----------USING THIS GIVES THE SAME AMPLITUDES THAT UE LI HAD IN HIS PAPER


def P_delta_v(kperp,kpar,z):
    Kperp,Kpar=np.meshgrid(kperp,kpar)
    k=np.sqrt(Kpar**2+Kperp**2)
    mu_k=Kpar/k
    Pdeltav=f(z)*H(z)*Mps_interpf(k)*mu_k/((1+z)*k)/cc.c_light_Mpc_s
    return k**3*Pdeltav


kpar=np.geomspace(5.e-3,1.,30)
kperp=np.geomspace(5.e-3,1.,30)

#k=np.sqrt(kpar**2+kperp**2)
#k=np.linspace(1.e-2,110,100)

#P=P_delta_delta(k)*P_vv(k,1.)+P_delta_v(k,1.)**2

#plt.semilogy(k,P)
#plt.plot(k,P_vv(k,1.))
#plt.plot(k,P_delta_v(k,1.))
'''
#plt.show()
print (P_delta_delta(kperp,kpar).max())
print (P_vv(kperp,kpar,1.).max())

pylab.pcolormesh(kperp,kpar,P_delta_delta(kperp,kpar),cmap='Blues',norm=LogNorm()) ;  cbar=plt.colorbar();
plt.tick_params(axis='both', which='major');
#pylab.xlim([np.min(kperp),np.max(kperp)])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k_\perp$',fontsize=12); plt.ylabel(r'$k_\parallel$',fontsize=12)
plt.title(r'$P_{\delta \delta}$')
pylab.show()

pylab.pcolormesh(kperp,kpar,P_vv(kperp,kpar,1.),cmap='Blues',norm=LogNorm()) ;  cbar=plt.colorbar()
#pylab.xlim([np.min(kperp),.5])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k_\perp$',fontsize=12); plt.ylabel(r'$k_\parallel$',fontsize=12)
plt.title(r'$P_{vv}$')
pylab.show()

pylab.pcolormesh(kperp,kpar,P_delta_v(kperp,kpar,1.),cmap='Blues',norm=LogNorm()) ;  cbar=plt.colorbar()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$k_\perp$',fontsize=12); plt.ylabel(r'$k_\parallel$',fontsize=12)
plt.title(r'$P_{\delta v}$')
pylab.show()
'''



'''
plt.loglog(k,k**3*Mps_interpf(k),label=r'$\rm{P_{\delta \delta}}$')
plt.loglog(k,k**3*P_delta_v(k,1.),label=r'$\rm{P_{\delta v}}$')
plt.loglog(k,k**3*P_vv(k,1.),label=r'$\rm{P_{vv}}$')
plt.xlabel('k')
plt.ylabel(r'$\rm{k^3 P(k,z=1)}$')
plt.legend()
plt.show()
'''

#plt.plot(z,T_mean(z))
#plt.xlabel('z')
#plt.ylabel('T(z)')
#plt.show()
##print (z)
'''
def chi_flat():
    for i in enumerate(z):
        chi =2*(1-(1/np.sqrt(1+z)))/H0
    return chi
#chi_f=chi_flat()
    ##print ("Comoving distance to z is %.1f Mpc" % (chi))
##print (chi)
##print (z)
#return res
#result=zed()
##plt.loglog(chi,b(chi))
##plt.show()
##plt.loglog(chi_f,z)
##plt.show()
##print (b(chi))
#f=cp.fgrowth(b(chi), omega_M_0=0.27, unnormed=False)
##print (f)
##plt.loglog(b(chi),f)
'''
delta_z=2.
z_r=10.
z_ri=z_r-delta_z/2
z_rf=z_r+delta_z/2
chi_ri=chi(z_ri)
chi_rf=chi(z_rf)
delta_chi=chi_rf-chi_ri
r_H=2*cc.c_light_Mpc_s/(3*H0*np.sqrt(cosmo['omega_M_0'])*(1+z_r)**1.5)
#r_H=cd.light_travel_distance(z_r,0.0,**cosmo)
chi_r=chi(z_r)
theta=r_H/cd.angular_diameter_distance(z_r,0,**cosmo)
#print (theta)

import reionization as cr

def tau_ind(z):
    tau=cr.integrate_optical_depth(z,x_ionH=1.0, x_ionHe=1.0, **cosmo)
    return tau

def tau_inst(z):
    tau_r=cr.optical_depth_instant(z, x_ionH=1.0, x_ionHe=1.0, z_rHe = None,return_tau_star=False, verbose=0, **cosmo)
    return tau_r
#print (tau_r)
#cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
 #        'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}
#I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo)
