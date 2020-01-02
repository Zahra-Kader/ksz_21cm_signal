# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:32:05 2018

@author: zahra
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import useful_functions as uf
import constants as cc


#rho_0=den.cosmo_densities(**cosmo)[1]*cc.M_sun_g #in units of g/Mpc^3

z=uf.z
n_points=100
r=uf.r
chi=uf.chi
f=uf.f
D=uf.D_1
T_mean=uf.T_mean
H=uf.H
tau=uf.tau_inst
Mps_interpf=uf.Mps_interpf

def Mps_k(ell):
    #Cl=np.array([])
    Mps=np.array([])
    kpar=(2/uf.r(1))
    for i in ell:
        k=np.sqrt(kpar**2+(i/uf.chi(1))**2)
        Mps_one_ell=uf.Mps_interpf(k)
        #Mps_div_ell=Mps*kpar**4*(T_mean(1)*(kpar**2/k**2))**2/(chi(1)**2*r(1))
        #Cl=np.append(Cl,Mps_div_ell)
        Mps=np.append(Mps,Mps_one_ell)
    return Mps

kpar=(2./uf.r(1))
kperp=0.
Mps_zero_ell=uf.Mps_interpf(kpar)
#print (Mps_zero_ell)

ell=np.linspace(1.,1.e4,n_points)

def Cl_21_func_of_y(ell,z,y,delta_z):
    z1=1.
    kpar=y/r(z1)
    chi_1=uf.chi(z1)
    f_1=uf.f(z1)
    D_1=uf.D_1(z1)
    k=np.sqrt(kpar**2+(ell/chi_1)**2)
    mu_k_sq=kpar**2/k**2
    a=uf.b_HI+f_1*mu_k_sq
    Cl=(uf.T_mean(z1)*a*D_1)**2*uf.Mps_interpf(k)/chi_1**2/r(z1)
    return Cl


def Cl_21(ell,z1):
    y=np.linspace(1,3000,n_points)
    kpar=y/r(z1)
    Cl=np.array([])
    for i in ell:
        #integral=sp.integrate.quadrature(lambda kpar:Cl_21_func_of_y(z1,kpar,i),1e-4,1e-1)[0]
        chi_1=uf.chi(z1)
        f_1=uf.f(z1)
        D_1=uf.D_1(z1)
        k=np.sqrt(kpar**2+(i/chi_1)**2)
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f_1*mu_k_sq
        integral=sp.integrate.trapz((uf.T_mean(z1)*a*D_1)**2*uf.Mps_interpf(k)/chi_1**2/r(z1)/2/np.pi,y,axis=0)
        Cl=np.append(Cl,integral)
    return Cl

def Integrand_doppler_21cm(ell,z,y,delta_z):
    z1=1.
    z2=z1
    r_1=r(z1)
    chi_1=chi(z1)
    f_1=f(z1)
    D1=D(z1)
    tau_1=tau(z1)
    H_1=H(z1)
    H_2=H(z2)
    f_2=f(z2)
    D2=D(z2)
    chi_2=chi(z2)
    tau_2=tau(z2)
    kpar=y/r_1
    k=np.sqrt(kpar**2+ell**2/chi_1**2)
    const=uf.T_mean(z1)*uf.T_mean(z2)/cc.c_light_Mpc_s**2
    mu_k=kpar/k
    rsd=1+f_1*mu_k**2
    integrand=D1*D2*H_1*H_2*f_1*f_2*Mps_interpf(k)*np.cos(kpar*(chi_1-chi_2))*rsd**2/(1+z1)/(1+z2)*mu_k**2/k**2/chi_1**2/r_1
    return const*integrand


def Cl_21_doppler(ell,z1):
    y=np.linspace(1,3000,n_points)
    kpar=y/r(z1)
    z1=1.
    z2=z1
    r_1=r(z1)
    chi_1=chi(z1)
    f_1=f(z1)
    D1=D(z1)
    tau_1=tau(z1)
    H_1=H(z1)
    H_2=H(z2)
    f_2=f(z2)
    D2=D(z2)
    chi_2=chi(z2)
    tau_2=tau(z2)
    Cl=np.array([])
    for i in ell:
        #integral=sp.integrate.quadrature(lambda kpar:Cl_21_func_of_y(z1,kpar,i),1e-4,1e-1)[0]
        k=np.sqrt(kpar**2+i**2/chi_1**2)
        const=uf.T_mean(z1)*uf.T_mean(z2)/cc.c_light_Mpc_s**2/2/np.pi
        mu_k=kpar/k
        rsd=1+f_1*mu_k**2
        integral=sp.integrate.trapz(D1*D2*H_1*H_2*f_1*f_2*Mps_interpf(k)*np.cos(kpar*(chi_1-chi_2))*rsd**2/(1+z1)/(1+z2)*mu_k**2/k**2/chi_1**2/r_1,y,axis=0)
        Cl=np.append(Cl,integral)
    return const*Cl

'''
def Integrand_doppler_21_lowl(ell):
    chi_1=uf.chi(z1)
    f_1=uf.f(z1)
    D_1=uf.D_1(z1)
    tau_1=uf.tau_inst(z1)
    H_1=uf.H(z1)
    H_2=uf.H(z2)
    f_2=uf.f(z2)
    D_2=uf.D_1(z2)
    chi_2=uf.chi(z2)
    tau_2=uf.tau_inst(z2)
    k=np.sqrt(kpar**2+ell**2/chi_1**2)
    mu_k=kpar/k
    rsd=1+f_1*mu_k**2
    integrand=D_1*D_2*H_1*H_2*f_1*f_2*Mps_interpf(kpar)*np.cos(kpar*(chi_1-chi_2))*rsd**2/chi_1**2/(1+z1)/(1+z2)/kpar**2
    return integrand

def func_doppler_21_lowl(z1,z2,ell):
    array=np.array([])
    err=np.array([])
    const=uf.T_mean(z1)*uf.T_mean(z2)/(2*np.pi)/cc.c_light_Mpc_s**2
    for i in ell:
        #print (i)
        integral=sp.integrate.quad(lambda kpar:Integrand_doppler_21cm(z1,z2,kpar,i),1e-4,1e2)
        array=np.append(array,const*integral[0])
        err=np.append(err,integral[1])
    return array,err
    '''

def Cl_21_momentum(ell,z_1,y,delta_z):
    z_2=z_1
    kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n_points)
    Kp=np.geomspace(1.e-10,.1,n_points)
    mu,kp=np.meshgrid(Mu,Kp)
    Cl_y_integ=np.array([])
    for i in ell:
        k_perp=i/chi(z_1)
        k=np.sqrt(kpar**2+k_perp**2)
        zeta=(kpar/k*mu+k_perp/k*np.sqrt(1-mu**2))
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*zeta))
        theta_kp=kpar*zeta/k+k_perp*np.sqrt(1-zeta**2)/k
        theta_K=kpar/K/k*(k-kp*zeta)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f(z_1)*mu_k_sq
        const=1/(8*np.pi**2*cc.c_light_Mpc_s**2)*T_mean(z_1)**2*T_mean(z_2)**2*D(z_1)**2*D(z_2)**2*f(z_1)*f(z_2)*H(z_1)*H(z_2)/(1+z_1)/(1+z_2)/chi(z_1)**2/r(z_1)
        Cl_21_func_of_y=Mps_interpf(kp)*Mps_interpf(K)*kp**2*a**4*theta_kp*(theta_kp/kp**2+theta_K/K/kp)
        integral=const*sp.integrate.trapz(sp.integrate.trapz(Cl_21_func_of_y,Mu,axis=0),Kp,axis=0)
        Cl_y_integ=np.append(Cl_y_integ,integral)
    return Cl_y_integ


def Cl_21_momentum_single(ell,z_1,y,delta_z):
    z_2=z_1
    kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n_points)
    Kp=np.geomspace(1e-10,.1,n_points)
    mu,kp=np.meshgrid(Mu,Kp)
    Cl_y_integ=np.array([])
    k_perp=ell/chi(z_1)
    k=np.sqrt(kpar**2+k_perp**2)
    zeta=(kpar/k*mu+k_perp/k*np.sqrt(1-mu**2))
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*zeta))
    theta_kp=kpar*zeta/k+k_perp*np.sqrt(1-zeta**2)/k
    theta_K=kpar/K/k*(k-kp*zeta)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
    mu_k_sq=kpar**2/k**2
    a=uf.b_HI+f(z_1)*mu_k_sq
    const=1/(8*np.pi**2*cc.c_light_Mpc_s**2)*T_mean(z_1)**2*T_mean(z_2)**2*D(z_1)**2*D(z_2)**2*f(z_1)*f(z_2)*H(z_1)*H(z_2)/(1+z_1)/(1+z_2)/chi(z_1)**2/r(z_1)
    Cl_21_func_of_y=Mps_interpf(kp)*Mps_interpf(K)*kp**2*a**4*theta_kp*((theta_kp/kp**2)+theta_K/K/kp)
    integral=const*sp.integrate.trapz(sp.integrate.trapz(Cl_21_func_of_y,Mu,axis=0),Kp,axis=0)
    return integral


def Cl_21_momentum_integrated(ell,z_1):
    z_2=z_1
    y=np.geomspace(1.,3000.,n_points)
    Kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n_points)
    Kp=np.geomspace(1.e-10,10.,n_points)
    mu,kp,kpar=np.meshgrid(Mu,Kp,Kpar)
    Cl_y_integ=np.array([])
    for i in ell:
        k_perp=i/chi(z_1)
        k=np.sqrt(kpar**2+k_perp**2)
        zeta=(kpar/k*mu+k_perp/k*np.sqrt(1-mu**2))
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*zeta))
        theta_kp=kpar*zeta/k+k_perp*np.sqrt(1-zeta**2)/k
        theta_K=kpar/K/k*(k-kp*zeta)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f(z_1)*mu_k_sq
        const=1/(16*np.pi**3*cc.c_light_Mpc_s**2)*T_mean(z_1)**2*T_mean(z_2)**2*D(z_1)**2*D(z_2)**2*f(z_1)*f(z_2)*H(z_1)*H(z_2)/(1+z_1)/(1+z_2)/chi(z_1)**2
        Cl_21_func_of_y=Mps_interpf(kp)*Mps_interpf(K)*kp**2*a**4*theta_kp*((theta_kp/kp**2)+theta_K/K/kp)
        integral=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(Cl_21_func_of_y,Mu,axis=0),Kp,axis=0),Kpar,axis=0)
        Cl_y_integ=np.append(Cl_y_integ,integral)
    return Cl_y_integ

def Cl_21_momentum_squeezed_integrated(ell,z_1):
    z_2=z_1
    y=np.geomspace(1.,3000.,n_points)
    Kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n_points)
    Kp=np.geomspace(1.e-10,.1,n_points)
    mu,kp,kpar=np.meshgrid(Mu,Kp,Kpar)
    Cl_y_integ=np.array([])
    for i in ell:
        k_perp=i/chi(z_1)
        k=np.sqrt(kpar**2+k_perp**2)
        zeta=(kpar/k*mu+k_perp/k*np.sqrt(1-mu**2))
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*zeta))
        theta_kp=kpar*zeta/k+k_perp*np.sqrt(1-zeta**2)/k
        theta_K=kpar/K/k*(k-kp*zeta)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f(z_1)*mu_k_sq
        const=1/(16*np.pi**3*cc.c_light_Mpc_s**2)*T_mean(z_1)**2*T_mean(z_2)**2*D(z_1)**2*D(z_2)**2*f(z_1)*f(z_2)*H(z_1)*H(z_2)/(1+z_1)/(1+z_2)/chi(z_1)**2
        Cl_21_func_of_y=Mps_interpf(kp)*Mps_interpf(k)*kp**2*a**4*theta_kp*((theta_kp/kp**2))
        integral=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(Cl_21_func_of_y,Mu,axis=0),Kp,axis=0),Kpar,axis=0)
        Cl_y_integ=np.append(Cl_y_integ,integral)
    return Cl_y_integ

ell_large=np.linspace(1,1e4,10000)

#np.save('Cl_21_times_Cl_v21_z_1pt26_10000pts',Cl_21(ell,1.26)*Cl_21_doppler(ell,1.26))
#Cl_21_times_Cl_v21_z_1pt26=np.load('Cl_21_times_Cl_v21_z_1pt26_10000pts.npy')
#Cl_21_v21_z_1pt26_interp=sp.interpolate.interp1d(ell_large,Cl_21_times_Cl_v21_z_1pt26)
'''
plt.loglog(ell,ell*(ell+1)*Cl_21_momentum_integrated(ell,1.26)/2/np.pi,'r')
plt.loglog(ell,ell*(ell+1)*Cl_21_momentum_squeezed_integrated(ell,1.26)/2/np.pi,'b')
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)C_l^{p21}}(z)/(2\pi)[\mu K^4]$')
plt.legend(('Full signal','Squeezed limit'))
plt.show()
'''

'''
plt.loglog(ell,Cl_21_v21_z_1pt26_interp(ell),'r')
#plt.loglog(ell,Cl_21_doppler(ell,1.26),'b')
#plt.loglog(ell,Cl_21_momentum_integrated(ell,1.26),'g')
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)Cl^{21}}(z)$')
#plt.legend(('21 cm velocity times density','21 cm momentum'))
plt.show()




plt.loglog(ell,ell*(ell+1)*Cl_21_func_of_y(ell,1,y=277,delta_z=0.1)/(2*np.pi),'r')
plt.loglog(ell,ell*(ell+1)*Cl_21_func_of_y(ell,1,y=750,delta_z=0.1)/(2*np.pi),'b')
plt.loglog(ell,ell*(ell+1)*Cl_21_func_of_y(ell,1,y=2032,delta_z=0.1)/(2*np.pi),'g')
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)Cl^{21}}(y)/(2\pi)[\mu K^2]$')
plt.legend(('y=277','y=750','y=2032'))
plt.xlim(100,1000)
plt.ylim(0.1,50)
plt.show()


plt.loglog(ell,ell*(ell+1)*Integrand_doppler_21cm(ell,1,y=277,delta_z=0.1)/(2*np.pi),'r')
plt.loglog(ell,ell*(ell+1)*Integrand_doppler_21cm(ell,1,y=750,delta_z=0.1)/(2*np.pi),'b')
plt.loglog(ell,ell*(ell+1)*Integrand_doppler_21cm(ell,1,y=2032,delta_z=0.1)/(2*np.pi),'g')
plt.xlim(100,1000)
plt.ylim(1e-7,1e-4)
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)C_l^{21-Doppler}}(y)/2\pi[\mu K^2]$')
plt.legend(('y=277','y=750','y=2032'))
plt.show()

plt.loglog(ell,ell*(ell+1)*Cl_21_doppler(ell,1)/2/np.pi)
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)Cl^{21-Doppler}}(z)/2\pi[\mu K^2]$')
plt.show()



plt.loglog(ell,ell*(ell+1)*Cl_21_momentum(ell,1,277)/2/np.pi,'r')
plt.loglog(ell,ell*(ell+1)*Cl_21_momentum(ell,1,750)/2/np.pi,'b')
plt.loglog(ell,ell*(ell+1)*Cl_21_momentum(ell,1,1024)/2/np.pi,'g')
plt.legend(('y=277','y=750','y=2032'))
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)C_l^{{p21}}(y,z)/2\pi[\mu K^4]}$')
plt.show()

plt.loglog(ell,ell*(ell+1)*Cl_21_momentum_integrated(ell,1.)/2/np.pi)
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)Cl^{p21}}(z)/2\pi[\mu K^4]$')
plt.show()
'''
