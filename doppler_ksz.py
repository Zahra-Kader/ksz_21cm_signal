# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:20:16 2019

@author: zahra
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import useful_functions as uf
import density as den
import constants as cc
from sympy import *

cosmo=uf.cosmo

Mps_interpf=uf.Mps_interpf
Mps_interpf_div_sq=uf.Mps_interpf_div_ksq

n_points=uf.n_points
z_r=10


G=cc.G_const_Mpc_Msun_s/cc.M_sun_g

#rho_c=3*H0**2/(8.*np.pi*G)

rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g #in units of g/Mpc^3

sigma_T=cc.sigma_T_Mpc

tau_r=0.055

T_rad=2.725 #In Kelvin

k_min=1e-4

k_max=10

#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]

#get matter power spec data

mu_e=1.14

m_p=cc.m_p_g #in grams

rho_g0=cosmo['omega_b_0']*rho_c

def ionized_elec(Yp,N_He):
    x=(1-Yp*(1-N_He/4))/(1-Yp/2)
    return x

def density(z,rho_g0):
    rho_g=rho_g0*(1+z)**3
    return rho_g

def ionized_electron_number(z):
    n=ionized_elec(0.24,1)*density(z,rho_g0)/(mu_e*m_p)
    return n

#print (tau(z))
#plt.plot(z,tau(z))
#plt.show()
x=ionized_elec(0.24,0)
#kp=np.logspace(np.log(k_min),np.log(k_max),n_points)
H=uf.H
z=uf.z
ell=np.geomspace(1,1e3,n_points)
chi=uf.chi
f=uf.f
D=uf.D_1
tau=uf.tau_inst

def Integrand_doppler(z2,z1,kpar,ell): #makes no difference to speed if you write chi=uf.chi, etc. inside or outside the function
    chi_1=chi(z1)
    chi_2=chi(z2)
    f_1=f(z1)
    f_2=f(z2)
    D_1=D(z1)
    D_2=D(z2)
    tau_1=tau(z1)
    tau_2=tau(z2)
    k=np.sqrt(kpar**2+ell**2/chi_1**2)
    mu_k=kpar/k
    #cos_func=np.cos(kpar*chi_1)*np.cos(kpar*chi_2)+np.sin(kpar*chi_1)*np.sin(kpar*chi_2)
    integrand=(1+z1)/chi_1**2*np.exp(-tau_1)*(1+z2)*np.exp(-tau_2)*D_1*D_2*f_1*f_2*mu_k**2/k**2*Mps_interpf(k)*np.cos(kpar*(chi_1-chi_2))
    return integrand

def redshift_second_int(chi2):
    f_2=f(z2)
    D_2=D(z2)
    tau_2=tau(z2)
    H_2=H(z2)
    integrand=(1+z2)*np.exp(-tau_2)*D_2*f_2*H_2
    return integrand

z2=uf.z
chi2=chi(z2)
plt.plot(chi2,redshift_second_int(chi2))

print (np.polyfit(chi2,redshift_second_int(chi2),deg=6,full=False,cov=True))
polyfit=np.poly1d(np.polyfit(chi2,redshift_second_int(chi2),6))
plt.plot(chi2,polyfit(chi2))



def box_vol(n_side,Z1,Z2,Kpar,ell):
    d_z1=(Z1[1]-Z1[0])/n_side
    d_z2=((Z2)[1]-Z2[0])/n_side
    d_kpar=(Kpar[1]-Kpar[0])/n_side
    unit_vol=d_z1*d_z2*d_kpar
    sum=0
    for i in range(1,n_side+1):
        for j in range(1,n_side+1):
            for k in range(1,n_side+1):
                sum+=Integrand_doppler(i*d_z1,j*d_z2,k*d_kpar,ell)
    sum*=unit_vol
    return sum

'''
def trapz(Z1,Z2,Kpar,ell):
    array=np.array([])
    for i in ell:
        integral=sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(Integrand_doppler(Z1,Z2,Kpar,i),Z1,axis=0),Z2,axis=0),Kpar,axis=0)
        array=np.append(array,integral)
    return array
'''
Kpar=np.geomspace(1e-6,0.1,n_points)
Z2=z
Z1=z


def func_doppler_quadrature_single_ell(ell):
    int1=lambda Z1,Kpar,ell: sp.integrate.quadrature(Integrand_doppler,1e-4,10,args=(Z1,Kpar,ell, ),vec_func=False)[0]
    int2=lambda Kpar,ell: sp.integrate.quadrature(int1,1e-4,10,args=(Kpar,ell,),vec_func=False)[0]
    int3=sp.integrate.quadrature(int2,1e-6,0.1,args=(ell,),vec_func=False)
    return int3

def func_doppler_nquad_single_ell(ell):
    integral=sp.integrate.nquad(lambda z2,z1,kpar: Integrand_doppler(z2,z1,kpar,ell),[[1e-4,10],[1e-4,10],[1e-4,0.1]],opts={'epsabs':1e-5,'epsrel':1e-5,'limit':1000},full_output=False)
    return integral



#print(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(Integrand_doppler(Z2,Z1,Kpar,100),Z2,axis=0),Z1,axis=0),Kpar,axis=0),'trapz')
#print(func_doppler_quadrature_single_ell(100),'quadrature')
#%timeit func_doppler_nquad(ell)[0]

#print(func_doppler_nquad_single_ell(1000),'nquad')

#plt.plot(ell,ell*(ell+1)*func_doppler_nquad(ell)[0]/2/np.pi)
#plt.show()
