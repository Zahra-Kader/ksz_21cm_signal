# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 21:25:18 2018

@author: zahra
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import useful_functions as uf
import distance as cd
import density as den
import constants as cc
import reionization as cr
from decimal import Decimal

cosmo=uf.cosmo

Mps_interpf=uf.Mps_interpf

n_points=uf.n_points

x_e=1.

G=cc.G_const_Mpc_Msun_s/cc.M_sun_g

#rho_c=3*H0**2/(8.*np.pi*G)

rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g #in units of g/Mpc^3

sigma_T=cc.sigma_T_Mpc

tau_r=0.055

T_rad=2.725 #In Kelvin

chi_m = uf.chi(1100)

chi_min=0.0001

k_min=1e-4

k_max=4.5e-2

chi_array=np.linspace(chi_min,chi_m,n_points)

#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]

#get matter power spec data

mu_e=1.44

m_p=cc.m_p_g #in grams

rho_g0=cosmo['omega_b_0']*rho_c

#plt.loglog(uf.kabs,Mps_interpf(uf.kabs))

#plt.show()


zed=uf.zed
#z=uf.z



def ionized_elec(Yp,N_He):

    x=(1-Yp*(1-N_He/4))/(1-Yp/2)

    return x



def density(z,rho_g0):

    rho_g=rho_g0*(1+z)**3

    return rho_g



def ionized_electron_number(z):

    n=ionized_elec(0.24,1)*density(z,rho_g0)/(mu_e*m_p)

    return n



def tau(z):

    tau_der=sigma_T*cc.c_light_Mpc_s*ionized_electron_number(z)/((1+z)*uf.H(z))

    tau=sp.integrate.trapz(tau_der,z)

    return tau

#print (tau(z))

#plt.plot(z,tau(z))

#plt.show()

x=ionized_elec(0.24,0)

'''

x=np.linspace(0,5,5)

y=np.linspace(0,5,5)

xx=x[:,np.newaxis]

yy=y[np.newaxis,:]

integrand1=xx*yy

print (integrand1)

f1=sp.integrate.trapz(integrand1,xx,axis=0)

print (f1)

integral=sp.integrate.trapz(f1,y,axis=0)

print (integral)



#k=np.linspace(k_min,k_max,n_points)

k=1.

ell=1.

k_p=np.linspace(k_min,k_max,n_points)

zz=z[:,np.newaxis]

kk=k_p[np.newaxis,:]

#const=8*np.pi**2/((cc.c_light_Mpc_s)*(2*ell+1)**3)*(sigma_T*rho_g0/(mu_e*m_p))**2

mu=k**2+kk**2-(k-kk)**2/(2*k*kk)

I=k*(k-2*kk*mu)*(1-mu**2)/(kk**2*(k**2+kk**2-2*k*kk*mu))

delta_sq=k**3/(2*np.pi**2)*uf.f(zz)**2*kk**2/(2*np.pi)**3*Mps_interpf(kk)*Mps_interpf(k-kk)*I

integrand=[(1+zz)**2*ionized_elec(0.24,1)**2*delta_sq*uf.chi(zz)*uf.H(zz)]

integrand=np.resize(integrand,(5,5))

print (integrand)

f=sp.integrate.trapz(integrand,zz,axis=0)

print (f)

C_l=sp.integrate.trapz(f,k_p,axis=0)

print (C_l)

'''



K=np.linspace(k_min,k_max,n_points)


ell=np.linspace(1e-2,1e4,n_points)

chi=uf.chi

#k=ell/chi


def chi_check(z):
    chi_der=cc.c_light_Mpc_s/uf.H(z)
    chi=sp.integrate.trapz(chi_der,z)
    return chi

#print (chi_check(z))

min_k=1e-2
def Delta_b(k,z):
    k,kp=np.meshgrid(K,K)
    #mu=kk/i

    #mu=(i**2+kk**2-(np.abs(i-kk))**2)/(2*i*kk)

    #print (mu)

    #I=i**2/(kk**2*(i**2+kk**2))
    #k_inp=np.linspace(min_k,i,n_points)
    #print (k_inp)
    k_norm=np.linalg.norm(k)
    kp_norm=np.linalg.norm(kp)

    #print (k_norm,'knorm')

    mu=np.dot(k,kp)/(k_norm*kp_norm)

    #print (mu,'mu')

    I=k*(k-2*kp*mu)*(1-mu**2)/(k**2+kp**2-2*k*kp*mu)
    #print (I,'I')
    constants=k**3/((2*np.pi**2)*(2*np.pi)**3)

    delta_sq=constants*uf.f(z)**2*uf.D_1(z)**4*Mps_interpf(kp)*Mps_interpf(np.abs(k-kp))*I/(1+z)**2

    Integral=sp.integrate.trapz(delta_sq,K)
    #print (Delta_b_sqrt)

    return Integral



k=np.linspace(min_k,10,n_points)
#k=ell/chi(z)

print ((np.sqrt(Delta_b(k,z=0))*k).min(),'min')
print ((np.sqrt(Delta_b(k,z=4))*k).min(),'min')
plt.loglog(k,np.sqrt(Delta_b(k,z=0))*k,'b')

plt.loglog(k,np.sqrt(Delta_b(k,z=4))*k,'r')



plt.ylim(1e-13,1e3)

plt.xlim(1e-2,10)

plt.ylabel(r'$\rm{\Delta_b(k,z)k/H(z)}$')

plt.xlabel('k [h/Mpc]')

plt.show()


def C_l(ell,z_min):
    z=np.linspace(z_min,10,n_points)
    C_l=np.array([])
    for i in ell:     
        const=1e12*8*np.pi**2*T_rad**2*x**2/cc.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2/(2*i+1)**3
        integrand=const*(1+z)**4*uf.H(z)*chi(z)*np.exp(-2*tau(z))*Delta_b(i/chi(z),z)
        #print (np.shape(integrand))
   
        #integrand=np.resize(integrand,(n_points+1))
   
        #print (np.shape(integrand))
        Redshift_dep=sp.integrate.trapz(integrand,z)
        C_l=np.append(C_l,Redshift_dep)  
    return C_l


'''
def C_l_single_ell(ell,z_max):
   
    z=np.linspace(1e-2,z_max,n_points)
       
    const=1e12*8*np.pi**2*T_rad**2*x**2/cc.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2/(2*np.pi)

    integrand=const*(1+z)**4*uf.H(z)*uf.chi(z)*np.exp(-2*tau(z))*Delta_b(ell/chi(z),z)
    #print (np.shape(integrand))

    #integrand=np.resize(integrand,(n_points+1))

    #print (np.shape(integrand))

    Redshift_dep=sp.integrate.cumtrapz(integrand,z,initial=0)

    return integrand,Redshift_dep



def C_l(ell):

    C_l=R_d()*Delta_b(k)**2/(2*ell+1)**3

    return C_l





def C_l(ell):

    const=8*np.pi**2*T_rad**2*x**2/cc.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2

    integrand=[const*(1+z)**2*uf.chi(z)*uf.f(z)**2*uf.H(z)*Delta_b(k)**2*np.exp(-2*tau(z))]

    #print (np.shape(integrand))

    #integrand=np.resize(integrand,(n_points,n_points))

    #print (np.shape(integrand))

    Redshift_dep=sp.integrate.trapz(integrand,z)

    C_l=Redshift_dep/(2*ell+1)**3

    return C_l
'''
plt.semilogy(ell,ell*(ell+1)*C_l(ell,1e-4)/(2*np.pi))
#plt.ylim(0,5)
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)Cl^{OV}}/(2 \pi)[\mu K^2]$')
plt.show()


'''
plt.plot(np.linspace(1e-2,1,n_points),C_l_single_ell(3000,1)[1])
plt.plot(np.linspace(1e-2,2,n_points),C_l_single_ell(3000,2)[1])
plt.plot(np.linspace(1e-2,3,n_points),C_l_single_ell(3000,3)[1])
plt.plot(np.linspace(1e-2,4,n_points),C_l_single_ell(3000,4)[1])
plt.plot(np.linspace(1e-2,5,n_points),C_l_single_ell(3000,5)[1])
plt.plot(np.linspace(1e-2,6,n_points),C_l_single_ell(3000,6)[1])


plt.xlabel('z')
plt.ylabel(r'$\rm{Cl_{3000}^{OV}}/(2 \pi)[\mu K^2]$')
#plt.loglog(ell,ell*(ell+1)*C_l(ell)*1e12/(2*np.pi))

plt.show()
'''