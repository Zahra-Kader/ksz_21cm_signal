# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 00:22:49 2018

@author: zahra
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 07:33:16 2018

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

n_points=100

z_r=10

x_e=1.

G=cc.G_const_Mpc_Msun_s/cc.M_sun_g

#rho_c=3*H0**2/(8.*np.pi*G)

rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g #in units of g/Mpc^3

sigma_T=cc.sigma_T_Mpc

tau_r=0.055

T_rad=2.725 #In Kelvin

chi_r = uf.chi(z_r)

chi_m = uf.chi(1100)

chi_min=0.0001

k_min=1e-4

k_max=2e-2

chi_array=np.linspace(chi_min,chi_m,n_points)

#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]

#get matter power spec data

mu_e=1.44

m_p=cc.m_p_g #in grams

rho_g0=cosmo['omega_b_0']*rho_c

#plt.loglog(uf.kabs,Mps_interpf(uf.kabs))

#plt.show()


zed=uf.zed
z=np.geomspace(1e-4,10,n_points)



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

##print (tau(z))

#plt.plot(z,tau(z))

#plt.show()

x=ionized_elec(0.24,0)

ell=np.linspace(100,1e4,n_points)

chi=uf.chi
H=uf.H
#k=ell/chi
kp=np.geomspace(k_min,k_max,n_points)


def chi_check(z):
    chi_der=cc.c_light_Mpc_s/uf.H(z)
    chi=sp.integrate.trapz(chi_der,z)
    return chi

##print (chi_check(z))

min_k=1e-2

def Delta_b(k,z):
    Delta_b=np.array([])
    kp_norm=np.linalg.norm(kp)
    k_norm=np.linalg.norm(k)
    D_z=uf.D_1(z)
    f_z=uf.f(z)
    for i in k:
        mu=np.dot(i,kp)/(k_norm*kp_norm)
        I=i*(i-2*kp*mu)*(1-mu**2)/(i**2+kp**2-2*i*kp*mu)
        constants=i**3/((2*np.pi**2)*(2*np.pi)**3)*2*np.pi
       
        delta_sq=constants*f_z**2*D_z**4*Mps_interpf(kp)*Mps_interpf(np.sqrt(i**2+kp**2-2*i*kp*mu))*I/(1+z)**2
        Integral=sp.integrate.trapz(delta_sq,kp)
        Delta_b=np.append(Delta_b,Integral)
    return Delta_b



def Delta_b_noredshift(k):
    Delta_b=np.array([])
    kp_norm=np.linalg.norm(kp)
    k_norm=np.linalg.norm(k)
    for i in k:
        mu=np.dot(i,kp)/(k_norm*kp_norm)
        I=i*(i-2*kp*mu)*(1-mu**2)/(i**2+kp**2-2*i*kp*mu)
        constants=i**3/((2*np.pi**2)*(2*np.pi)**3)*2*np.pi
        delta_sq=constants*Mps_interpf(kp)*Mps_interpf(np.abs(i-kp))*I
        Integral=sp.integrate.trapz(delta_sq,kp)
        Delta_b=np.append(Delta_b,Integral)
    return Delta_b    

def powerspec(k):
    Mps=np.array([])
    kp_norm=np.linalg.norm(kp)
    k_norm=np.linalg.norm(k)
    for i in k:
       mu=np.dot(i,kp)/(k_norm*kp_norm) 
       I=i*(i-2*kp*mu)*(1-mu**2)/(i**2+kp**2-2*i*kp*mu)
       powerspec=Mps_interpf(np.sqrt(i**2+kp**2-2*i*kp*mu))
       Integral=sp.integrate.trapz(powerspec,kp)
       Mps=np.append(Mps,Integral)
    return Mps
       
def powerspec_final(ell):
    z1,z2=np.meshgrid(z,z)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=tau(z1)
    tau_z2=tau(z2)
    H_z=uf.H(z)
    chi_z=chi(z)
    tau_z=uf.tau_inst(z)
    f_z=uf.f(z)
    D_z=uf.D_1(z) 
    Mps=np.array([])
    for i in ell:
        Integrand=[sp.integrate.trapz(powerspec(np.sqrt(j**2+i**2/chi_z**2)),z) for j in kpar]
        Integral=sp.integrate.trapz(Integrand,kpar)
        #Integral=sp.integrate.trapz((1+z)**2/chi_z**2*powerspec(i/chi_z)*D_z**4*H_z*f_z**2*np.exp(-2*tau_z),z)
        #Integral=[sp.integrate.trapz(sp.integrate.trapz((1+z1)*(1+z2)*np.exp(-tau_z1)
        #*np.exp(-tau_z2)*2*f_z1*f_z2*D_z1**2*D_z2**2/chi_z1**2*powerspec(i/chi_z1),z,axis=0),z,axis=0)]
        Mps=np.append(Mps,Integral)
    return Mps

#plt.plot(ell,ell*(ell+1)*powerspec_final(ell))  
#plt.show()
    
def Delta_b_noredshift_single(ell,z):
    kp_norm=np.linalg.norm(kp)
    k=ell/uf.chi(z)
    k_norm=np.linalg.norm(k)  
    mu=np.dot(k,kp)/(k_norm*kp_norm)
    I=k*(k-2*kp*mu)*(1-mu**2)/(k**2+kp**2-2*k*kp*mu)
    constants=1/((2*np.pi**2)*(2*np.pi)**3)
    delta_sq=constants*Mps_interpf(kp)*Mps_interpf(np.abs(ell-kp))*I
    Integral=sp.integrate.trapz(delta_sq,kp)
    return Integral  


k=np.geomspace(1e-2,10,n_points)
#k=ell/chi(z)

##print ((np.sqrt(Delta_b(k,z=0))*k).min(),'min')
##print ((np.sqrt(Delta_b(k,z=4))*k).min(),'min')
plt.loglog(k,np.sqrt(Delta_b(k,z=0))*k,'b')
plt.loglog(k,np.sqrt(Delta_b(k,z=4))*k,'r')
plt.ylim(1e-13,1e3)
plt.xlim(1e-2,10)
plt.ylabel(r'$\rm{\Delta_b(k,z)k/H(z)}$')
plt.xlabel('k [h/Mpc]')
plt.show()

def C_l_integrand(ell,z):
    H_z=uf.H(z)
    chi_z=chi(z)
    tau_z=uf.tau_inst(z)
    f_z=uf.f(z)
    D_z=uf.D_1(z)    
    const=1e12*8*np.pi**2*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s
    integrand=const*(1+z)**2*np.exp(-2*tau_z)*chi_z*f_z**2*H_z*D_z**4*Delta_b_noredshift(ell/chi_z)/(2*ell+1)**3
        ##print (np.shape(integrand))
        #integrand=np.resize(integrand,(n_points+1))
        ##print (np.shape(integrand))
    return integrand


def C_l_allredshift(ell,z_min):
    z=np.geomspace(z_min,z_r,n_points)
    f_z=uf.f(z)
    D_z=uf.D_1(z)
    H_z=uf.H(z)
    chi_z=chi(z)
    tau_z=tau(z)
    C_l=np.array([])
    for i in ell:     
        const=1e12*8*np.pi**2*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s
        integrand=const*(1+z)**2*chi_z*np.exp(-2*tau_z)*f_z**2*H_z*Delta_b_noredshift(i/chi_z)*D_z**4/(2*i+1)**3
        Redshift_dep=sp.integrate.trapz(integrand,z)
        C_l=np.append(C_l,Redshift_dep)
    #print (C_l)
    return C_l


def C_l_mu_Delta_b(k,z):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    mu,kp=np.meshgrid(Mu,Kp)
    #chi_Z=chi(Z)
    f=uf.f
    D=uf.D_1
    #kp_norm=np.linalg.norm(Kp)
    #k_norm=np.linalg.norm(ell/chi_Z)
    C_l=np.array([])
    for i in k: 
        #mu=np.dot(k,Kp)/(k_norm*kp_norm)
        #I=k*(k-2*kp*j)*(1-j**2)/(k**2+kp**2-2*k*kp*j)
        constants=i**3/((2*np.pi**2)*(2*np.pi)**3)*2*np.pi
        integral=[constants*sp.integrate.trapz(sp.integrate.trapz(f(z)**2
        *Mps_interpf(kp)*Mps_interpf(np.sqrt(i**2+kp**2-2*i*kp*mu))*i*(i-2*kp*mu)
        *(1-mu**2)/(i**2+kp**2-2*i*kp*mu)/(1+z)**2*D(z)**4*H(z)**2,Kp,axis=0),Mu,axis=0)] 
        C_l=np.append(C_l,integral)
    return C_l

plt.loglog(k*0.7,np.sqrt(C_l_mu_Delta_b(k*0.7,0))*k*0.7/H(0))
plt.loglog(k*0.7,np.sqrt(C_l_mu_Delta_b(k*0.7,4))*k*0.7/H(4))
plt.ylim(1e-13,1e3)
plt.xlim(1e-2,10)
plt.show()
'''
def C_l_mu_integral(ell,z_min):
    Z=np.geomspace(z_min,z_r,n_points)
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)

    mu,kp=np.meshgrid(Mu,Kp)
    H_z=uf.H
    chi_z=chi
    #chi_Z=chi(Z)
    tau_z=uf.tau_inst
    f_z=uf.f
    D_z=uf.D_1
    const=1e12*8*np.pi**2*T_rad**2*x**2/cc.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2
    #kp_norm=np.linalg.norm(Kp)
    #k_norm=np.linalg.norm(ell/chi_Z)
    C_l=np.array([])
    for i in ell: 
        constants=1/((2*np.pi**2)*(2*np.pi)**(3))*2*np.pi
        #mu=np.dot(k,Kp)/(k_norm*kp_norm)
        #I=k*(k-2*kp*j)*(1-j**2)/(k**2+kp**2-2*k*kp*j)
        integral=[constants*const*sp.integrate.trapz(sp.integrate.trapz(C_l_mu_Delta_b(i/chi(j),j)/(1+z)**2*H_z
        *np.exp(-2*tau_z)*chi_z/(2*i+1)**3,Kp,axis=0),Mu,axis=0) for j in z]
        integral_final=sp.integrate.trapz(integral,Z)
        C_l=np.append(C_l,integral_final)
    return C_l

def C_l_mu_Delta_b_single(k,z):
    Kp=np.linspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    mu,kp=np.meshgrid(Mu,Kp)
    #chi_Z=chi(Z)
    f_z=uf.f(z)
    D_z=uf.D_1(z)
    #kp_norm=np.linalg.norm(Kp)
    #k_norm=np.linalg.norm(ell/chi_Z)
        #mu=np.dot(k,Kp)/(k_norm*kp_norm)
        #I=k*(k-2*kp*j)*(1-j**2)/(k**2+kp**2-2*k*kp*j)
    constants=k**3/((2*np.pi**2)*(2*np.pi)**3)
    integral=constants*sp.integrate.trapz(sp.integrate.trapz(f_z**2
    *Mps_interpf(kp)*Mps_interpf(np.sqrt(k**2+kp**2-2*k*kp*mu))*k*(k-2*kp*mu)
    *(1-mu**2)/(k**2+kp**2-2*k*kp*mu)/(1+z)**2*D_z**4,Kp,axis=0),Mu,axis=0) 
    return integral
'''
def C_l_mu_integrand(ell,z):
    #chi_Z=chi(Z)
    tau=uf.tau_inst
    f=uf.f
    D=uf.D_1
    const=1e12*8*np.pi**2*T_rad**2*x**2/cc.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2/(2*ell+1)**3
    #kp_norm=np.linalg.norm(Kp)
    #k_norm=np.linalg.norm(ell/chi_Z)
        #mu=np.dot(k,Kp)/(k_norm*kp_norm)
        #I=k*(k-2*kp*j)*(1-j**2)/(k**2+kp**2-2*k*kp*j)
    array=np.array([])
    for j in z:
        integral=const*C_l_mu_Delta_b(ell/chi(z),j)/H(j)*(1+j)**4*np.exp(-2*tau(j))*chi(j) 
        array=np.append(array,integral)
    return integral
'''
def C_l_mu_integrand_single(k,z):
    Kp=np.linspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    mu,kp=np.meshgrid(Mu,Kp)
    H=uf.H
    chi=uf.chi
    #chi_Z=chi(Z)
    tau=uf.tau_inst
    f=uf.f
    D=uf.D_1
    const=1e12*np.pi**2*T_rad**2*x**2/cc.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2
    #kp_norm=np.linalg.norm(Kp)
    #k_norm=np.linalg.norm(ell/chi_Z)
    constants=k**3/((2*np.pi**2)*(2*np.pi)**3)
    #k=ell/chi(z)
    #mu=np.dot(k,Kp)/(k_norm*kp_norm)
    #I=k*(k-2*kp*j)*(1-j**2)/(k**2+kp**2-2*k*kp*j)
    integral=constants*sp.integrate.trapz(sp.integrate.trapz(f(z)**2*D(z)**4
    *Mps_interpf(kp)*Mps_interpf(np.sqrt(k**2+kp**2-2*k*kp*mu))*k*(k-2*kp*mu)
    *(1-mu**2)/(k**2+kp**2-2*k*kp*mu)/(1+z)**2,Kp,axis=0),Mu,axis=0) 
    return integral
'''
def C_l_castro(ell,z_min):
    z=np.geomspace(z_min,z_r,n_points)
    kp=np.geomspace(k_min,k_max,n_points)
    Y1=kp*uf.chi(z)/ell
    Mu=np.linspace(-1,1,n_points)
    mu,y1=np.meshgrid(Mu,Y1)
    H=uf.H
    chi=uf.chi
    #chi_Z=chi(Z)
    tau=uf.tau_inst
    f=uf.f
    D=uf.D_1
    const=1e12*T_rad**2*x**2/cc.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2
    #kp_norm=np.linalg.norm(Kp)
    #k_norm=np.linalg.norm(ell/chi_Z)
    constants=1/(2*np.pi)**(3/2)*2*np.pi/2
    C_l=np.array([])
    for i in ell: 
        #mu=np.dot(k,Kp)/(k_norm*kp_norm)
        #I=k*(k-2*kp*j)*(1-j**2)/(k**2+kp**2-2*k*kp*j)
        integral=[constants*sp.integrate.trapz(sp.integrate.trapz(f(j)**2
        *Mps_interpf(i*y1/chi(j))*Mps_interpf(i/chi(j)*np.sqrt(1+y1**2-2*y1*mu))*i/chi(j)
        *(1-2*y1*mu)*(1-mu**2)/(1+y1**2-2*y1*mu)*const*(1+j)**2*H(j)*D(j)**4
        *np.exp(-2*tau(j))/chi(j)**2,Y1,axis=0),Mu,axis=0) for j in z]
        integral_final=sp.integrate.trapz(integral,z)
        C_l=np.append(C_l,integral_final)
    return C_l

def C_l_castro_triplemesh(ell,z_min):
    Z=np.geomspace(z_min,z_r,n_points)
    kp=np.geomspace(k_min,k_max,n_points)
    Y1=kp*uf.chi(Z)/ell
    Mu=np.linspace(-1,1,n_points)
    mu,y1,z=np.meshgrid(Mu,Y1,Z)
    H=uf.H
    chi=uf.chi
    #chi_Z=chi(Z)
    tau=uf.tau_inst
    f=uf.f
    D=uf.D_1
    const=1e12*T_rad**2*x**2/cc.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2
    #kp_norm=np.linalg.norm(Kp)
    #k_norm=np.linalg.norm(ell/chi_Z)
    constants=1/(2*np.pi)**(3/2)*2*np.pi/2
    C_l=np.array([])
    for i in ell: 
        print (i)
        #mu=np.dot(k,Kp)/(k_norm*kp_norm)
        #I=k*(k-2*kp*j)*(1-j**2)/(k**2+kp**2-2*k*kp*j)
        integral=[constants*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(f(z)**2
        *Mps_interpf(i*y1/chi(z))*Mps_interpf(i/chi(z)*np.sqrt(1+y1**2-2*y1*mu))*i/chi(z)
        *(1-2*y1*mu)*(1-mu**2)/(1+y1**2-2*y1*mu)*const*(1+z)**2*H(z)*D(z)**4
        *np.exp(-2*tau(z))/chi(z)**2,Y1,axis=0),Mu,axis=0),Z,axis=0)]
        C_l=np.append(C_l,integral)
    return C_l

def C_l_castro_integrand(ell,z):
    Kp=np.geomspace(k_min,k_max,n_points)
    Y1=Kp*uf.chi(z)/ell
    Mu=np.linspace(-1,1,n_points)
    mu,y1=np.meshgrid(Mu,Y1)
    H=uf.H
    chi=uf.chi 
    tau=uf.tau_inst
    #chi_Z=chi(Z)
    f=uf.f
    D=uf.D_1
    const=1e12*T_rad**2*x**2/cc.c_light_Mpc_s*(sigma_T*rho_g0/(mu_e*m_p))**2
    #kp_norm=np.linalg.norm(Kp)
    #k_norm=np.linalg.norm(ell/chi_Z)
    C_l=np.array([])
    for i in z: 
        k=ell/chi(i)
        constants=1/(2*np.pi)**(3/2)*2*np.pi/2
        #mu=np.dot(k,Kp)/(k_norm*kp_norm)
        #I=k*(k-2*kp*j)*(1-j**2)/(k**2+kp**2-2*k*kp*j)
        integral=[constants*sp.integrate.trapz(sp.integrate.trapz(f(i)**2
        *Mps_interpf(k*y1)*Mps_interpf(k*np.sqrt(1+y1**2-2*y1*mu))*k*(1-2*y1*mu)**2
        *(1-mu**2)/(1+y1**2-2*y1*mu)**2*const*(1+i)**2*H(i)*D(i)**4
        *np.exp(-2*tau(i))/chi(i)**2,Y1,axis=0),Mu,axis=0)]
        C_l=np.append(C_l,integral)
    return C_l
        
'''
plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,1e-4))

#plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,0.8)/(2*np.pi))
#plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,5)/(2*np.pi))
#plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,2.5)/(2*np.pi))
plt.xlim(0,10000)

plt.ylim(0,5)
plt.show()


plt.plot(z,3000*(3000+1)*C_l_integrand(3000,z))
plt.xlim(0,10)
plt.ylim(0,1.4)
plt.show()



plt.plot(ell,ell*(ell+1)*C_l_mu_integral(ell,1e-4)/(2*np.pi))
plt.xlim(0,10000)
plt.ylim(0,0.4)
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)Cl^{OV}}/(2 \pi)[\mu K^2]$')
plt.show()
'''
plt.plot(z,3000*(3000+1)*C_l_mu_integrand(3000,z)/(2*np.pi))
#plt.ylim(0,4)
plt.show()
'''
plt.plot(ell,ell*(ell+1)*C_l_castro(ell,1e-4)/(2*np.pi))
plt.ylim(0,5)
plt.ylabel(r'$\rm{l(l+1)Cl^{OV}}/(2 \pi)[\mu K^2]$')
plt.xlabel('l')
#plt.xlim(0,10000)
#plt.ylim(0,5)
plt.show()

plt.plot(z,3000*(3000+1)*C_l_castro_integrand(3000,z)/(2*np.pi))
plt.ylabel(r'$\rm{l(l+1)Cl^{OV}}/(2 \pi)[\mu K^2]$')
plt.xlabel('l')
plt.xlim(0,10)
plt.ylim(0,1.4)
plt.show()

plt.loglog(k,np.sqrt(C_l_mu_Delta_b(k,0))*k)
plt.loglog(k,np.sqrt(C_l_mu_Delta_b(k,4))*k)
plt.ylim(1e-13,1e3)
plt.xlim(1e-2,10)
plt.show()

plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,0.4)/(2*np.pi),'b')
plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,1.2)/(2*np.pi),'g')
plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,2.05)/(2*np.pi),'r')
plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,3)/(2*np.pi),'k')
plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,4)/(2*np.pi),'m')
plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,5)/(2*np.pi),'y')
blue_patch = mpatches.Patch(color='blue', label='z=0.4')
green_patch = mpatches.Patch(color='green', label='z=1.2')
red_patch = mpatches.Patch(color='red', label='z=2.05')
black_patch = mpatches.Patch(color='black', label='z=3')
magenta_patch = mpatches.Patch(color='magenta', label='z=4')
yellow_patch = mpatches.Patch(color='yellow', label='z=5')
plt.legend(handles=[blue_patch,green_patch,red_patch,black_patch,magenta_patch,yellow_patch])
plt.ylabel(r'$\rm{l(l+1)Cl^{OV}}/(2 \pi)[\mu K^2]$')
plt.xlabel('l')
#plt.ylim(1e-6,1e-3)
plt.show()

plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,1e-4)/(2*np.pi),'b')
plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,0.8)/(2*np.pi),'g')
plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,1.6)/(2*np.pi),'r')
plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,2.5)/(2*np.pi),'k')
plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,3.5)/(2*np.pi),'m')
plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,4.5)/(2*np.pi),'y')
plt.plot(ell,ell*(ell+1)*C_l_allredshift(ell,5.5)/(2*np.pi),'c')

blue_patch = mpatches.Patch(color='blue', label='6>z>0')
green_patch = mpatches.Patch(color='green', label='6>z>0.8')
red_patch = mpatches.Patch(color='red', label='6>z>1.6')
black_patch = mpatches.Patch(color='black', label='6>z>2.5')
magenta_patch = mpatches.Patch(color='magenta', label='6>z>3.5')
yellow_patch = mpatches.Patch(color='yellow', label='6>z>4.5')
cyan_patch = mpatches.Patch(color='cyan', label='6>z>5.5')
plt.legend(handles=[blue_patch,green_patch,red_patch,black_patch,magenta_patch,yellow_patch,cyan_patch])
plt.ylabel(r'$\rm{l(l+1)Cl^{OV}}/(2 \pi)[\mu K^2]$')
plt.xlabel('l')
plt.show()


plt.semilogy(ell,ell*(ell+1)*C_l(ell,1e-4),'b')
plt.semilogy(ell,ell*(ell+1)*C_l(ell,0.8),'g')
plt.semilogy(ell,ell*(ell+1)*C_l(ell,1.6),'r')
plt.semilogy(ell,ell*(ell+1)*C_l(ell,2.5),'k')
plt.semilogy(ell,ell*(ell+1)*C_l(ell,3.5),'m')
plt.semilogy(ell,ell*(ell+1)*C_l(ell,4.5),'y')
plt.semilogy(ell,ell*(ell+1)*C_l(ell,5.5),'c')
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)Cl^{OV}}/(2 \pi)[\mu K^2]$')
'''

 