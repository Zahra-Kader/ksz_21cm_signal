# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import useful_functions as uf
import density as den
import constants as cc
import matplotlib.patches as mpatches
import lin_kSZ_redshift_dep as lkSZ
import neutral_H_autocorr as nH
from scipy.interpolate import interp1d


cosmo=uf.cosmo
Mps_interpf=uf.Mps_interpf
Mps_interpf_div_sq=uf.Mps_interpf_div_ksq
n_points=uf.n_points
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g #in units of g/Mpc^3
sigma_T=cc.sigma_T_Mpc
tau_r=0.055
z_r=10
delta_z=0.5
T_rad=2.725 #In Kelvin
k_min=1e-4
k_max=10
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
#get matter power spec data
mu_e=1.14
m_p=cc.m_p_g #in grams
rho_g0=cosmo['omega_b_0']*rho_c
#plt.loglog(uf.kabs,Mps_interpf(uf.kabs))
#plt.show()
zed=uf.zed
z=uf.z
r=uf.r
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
#kp=np.logspace(np.log(k_min),np.log(k_max),n_points)
H=uf.H
ell=np.geomspace(1,1e4,n_points)
chi=uf.chi
Kpar=np.geomspace(1e-4,1e-2,n_points)
kpar=Kpar
#k=ell/chi


def chi_check(z):
    chi_der=cc.c_light_Mpc_s/uf.H(z)
    chi=sp.integrate.trapz(chi_der,z)
    return chi
min_k=1e-2

def y(z):
    y=(1+z)**(3/2)
    return y

def vis_function(z):
    delta_z=0.5
    delta_y=1.5*np.sqrt(1+z_r)*delta_z
    x=(1/2)*(1+np.tanh((y(z_r)-y(z))/delta_y))
    #x_H=1/(1+np.exp(-(z-z_r)/delta_z))
    vis_function=cc.c_light_Mpc_s*sigma_T*rho_g0/mu_e/m_p*(1+z)**2*np.exp(-uf.tau_inst(z))*(x)/H(z)
    return vis_function

#norm=1/sp.integrate.quad(vis_function,0,np.inf)[0]
#print (norm)
#z=np.linspace(0,1e10,100000000)

#print (sp.integrate.trapz(vis_function(z),z))

#plt.plot(np.linspace(200,400,1000),vis_function(np.linspace(200,400,1000)))
#plt.show()
'''
def Delta_b_z1_z2(k,z1,z2):
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    Delta_b=np.array([])
    for i in k:
        kp=np.linspace(k_min,k_max,n_points)
        kp_norm=np.linalg.norm(kp)
        k_norm=np.linalg.norm(k)
        mu=np.dot(i,kp)/(k_norm*kp_norm)
        I=i*(i-2*kp*mu)*(1-mu**2)/(i**2+kp**2-2*i*kp*mu)
        constants=i**3/((2*np.pi**2)*(2*np.pi)**3)
        delta_sq=constants*Mps_interpf(kp)*Mps_interpf(np.abs(i-kp))*I*f_z1*f_z2*D_z1**2*D_z2**2/(1+z1)/(1+z2)
        Integral=sp.integrate.trapz(delta_sq,kp)
        Delta_b=np.append(Delta_b,Integral)
    return Delta_b

def Delta_b(k):
    kp=np.geomspace(k_min,k_max,n_points)
    kp_norm=np.linalg.norm(kp)
    k_norm=np.linalg.norm(k)
    Delta_b=np.array([])
    constants=1/4
    for i in k:
        mu=np.dot(i,kp)/(k_norm*kp_norm)
        I=i*(i-2*kp*mu)*(1-mu**2)/(i**2+kp**2-2*i*kp*mu)
        delta_sq=constants*Mps_interpf(kp)*Mps_interpf(np.sqrt(i**2+kp**2-2*i*kp*mu))*I
        Integral=sp.integrate.trapz(delta_sq,kp)
        Delta_b=np.append(Delta_b,Integral)
    return Delta_b


def F_non_limber(k_perp,arg):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    mu,kp=np.meshgrid(Mu,Kp)
    const=1/(2*np.pi)**2*2*np.pi/2
    F=np.array([])
    for i in k_perp:
        integrand=[const*sp.integrate.trapz(sp.integrate.trapz((j*mu*kp**2/np.sqrt(i**2+j**2)
        +i*kp**2*np.sqrt(1-mu**2)/np.sqrt(i**2+j**2))
        *((j*mu+i*np.sqrt(1-mu**2))*(np.sqrt(i**2+j**2)-2*kp*mu)/kp**2/(i**2+j**2+kp**2-2*np.sqrt(i**2+j**2)*kp*mu)
        +2*j/kp/(i**2+j**2+kp**2-2*np.sqrt(i**2+j**2)*kp*mu))
        *Mps_interpf(kp)*Mps_interpf(np.abs(np.sqrt(i**2+j**2)-kp))*np.cos(j*arg)
        ,Kp,axis=0),Mu,axis=0) for j in kpar]
        integral=sp.integrate.trapz(integrand,kpar)
        F=np.append(F,integral)
    return F

def F_non_limber_nokpar(k_perp,arg):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    mu,kp=np.meshgrid(Mu,Kp)
    const=1/(2*np.pi)**2*2*np.pi/2
    F=np.array([])
    for i in k_perp:
        integrand=[const*sp.integrate.trapz(sp.integrate.trapz((j*mu*kp**2/np.sqrt(i**2+j**2)
        +i*kp**2*np.sqrt(1-mu**2)/np.sqrt(i**2+j**2))
        *((j*mu+i*np.sqrt(1-mu**2))*(np.sqrt(i**2+j**2)-2*kp*mu)/kp**2/(i**2+j**2+kp**2-2*np.sqrt(i**2+j**2)*kp*mu)
        +2*j/kp/(i**2+j**2+kp**2-2*np.sqrt(i**2+j**2)*kp*mu))
        *Mps_interpf(kp)*Mps_interpf(np.abs(np.sqrt(i**2+j**2)-kp))*np.cos(arg)
        ,Kp,axis=0),Mu,axis=0)]
        F=np.append(F,integrand)
    return F

def F_limber_I(k,arg):
    Kp=np.geomspace(k_min,k_max,n_points)
    Y1=Kp*uf.chi(z)/ell
    Mu=np.linspace(-1,1,n_points)
    mu,y1=np.meshgrid(Mu,Y1)
    Delta_b=np.array([])
    constants=1/(2*np.pi)**(3/2)*2*np.pi/2
    for i in k:
        Integral=[sp.integrate.trapz(sp.integrate.trapz(constants*Mps_interpf(y1)
        *Mps_interpf(np.abs(np.sqrt(i**2+j**2)-y1))*np.sqrt(i**2+j**2)*(np.sqrt(i**2+j**2)-2*y1*mu)
        *(1-mu**2)/(i**2+j**2+y1**2-2*np.sqrt(i**2+j**2)*y1*mu)*np.cos(j*arg),Y1,axis=0),Mu,axis=0) for j in kpar]
        Delta_b=np.append(Delta_b,sp.integrate.trapz(Integral,kpar))
    return Delta_b
#Took cos function out
def F_limber_y1(k,z_dep):
    Kp=np.geomspace(k_min,k_max,n_points)
    Y1=Kp*z_dep/ell
    Mu=np.linspace(-1,1,n_points)
    mu,y1=np.meshgrid(Mu,Y1)
    Delta_b=np.array([])
    constants=1/(2*np.pi)**(3/2)*2*np.pi/2
    for i in k:
        Integral=[sp.integrate.trapz(sp.integrate.trapz(constants*Mps_interpf(i*y1)
        *Mps_interpf(i*np.sqrt(1+y1**2-2*y1*mu))*(1-2*y1*mu)**2
        *(1-mu**2)/(1+y1**2-2*y1*mu)**2,Y1,axis=0),Mu,axis=0)]
        Delta_b=np.append(Delta_b,Integral)
    return Delta_b


def F_limber(k):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    mu,kp=np.meshgrid(Mu,Kp)
    Delta_b=np.array([])
    constants=1/(2*np.pi)**(3/2)*2*np.pi/2
    for i in k:
        Integral=[sp.integrate.trapz(sp.integrate.trapz(constants*Mps_interpf(kp)
        *Mps_interpf(np.abs(i-kp))*i*(i-2*kp*mu)
        *(1-mu**2)/(i**2+kp**2-2*i*kp*mu),Kp,axis=0),Mu,axis=0)]
        Delta_b=np.append(Delta_b,Integral)
    return Delta_b


#OV=lksz.C_l_mu_integral(ell,1e-4)

def Integrand(Mu,Kp,Z1,Z2,Kpar,ell):
    mu,kp,z1,z2,kpar=np.meshgrid(Mu,Kp,Z1,Z2,Kpar)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=uf.tau_inst(z1)
    tau_z2=uf.tau_inst(z2)
    k_perp=ell/chi_z1
    k=np.sqrt(k_perp**2+kpar**2)
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
    integrand=(kpar*mu*kp**2/k+k_perp*kp**2*np.sqrt(1-mu**2)/k)*((kpar*mu+k_perp*np.sqrt(1-mu**2))*(k-2*kp*mu)/kp**2/K**2+kpar/kp/K**2)*Mps_interpf(kp)*Mps_interpf(K)*(1+z1)*(1+z2)*np.exp(-tau_z1)*np.exp(-tau_z2)*f_z1*f_z2*D_z1**2*D_z2**2/chi_z1**2*np.cos(kpar*(chi_z1-chi_z2))
    return integrand



def Integrand_part(Mu,Kp,Z1,Z2,Kpar,ell):
    mu,kp,z1,z2,kpar=np.meshgrid(Mu,Kp,Z1,Z2,Kpar)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=uf.tau_inst(z1)
    tau_z2=uf.tau_inst(z2)
    k_perp=ell/chi_z1
    k=np.sqrt(k_perp**2+kpar**2)
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
    integrand=(kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k)**2*Mps_interpf(kp)*Mps_interpf(K)*(1+z1)*(1+z2)*np.exp(-tau_z1)*np.exp(-tau_z2)*f_z1*f_z2*D_z1**2*D_z2**2/chi_z1**2*np.cos(kpar*(chi_z1-chi_z2))
    return integrand


def func_fixedquad(mu,kp,ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/16/np.pi**3
    array=np.array([])
    for i in ell:
        int1=lambda kp,z1,z2,kpar,i: sp.integrate.fixed_quad(Integrand,-1,1,args=(kp,z1,z2,kpar,i, ),n=9)[0]
        int2=lambda z1,z2,kpar,i: sp.integrate.fixed_quad(int1,1e-5,1e-1,args=(z1,z2,kpar,i, ),n=9)[0]
        int3=lambda z2,kpar,i: sp.integrate.fixed_quad(int2,1e-4,10,args=(z2,kpar,i,),n=9)[0]
        int4=lambda kpar,i: sp.integrate.fixed_quad(int3,1e-4,10,args=(kpar,i,),n=9)[0]
        int5=sp.integrate.fixed_quad(int4,1e-4,1e-2,args=(i,),n=9)[0]
        array=np.append(array,const*int5)
    return array

def func_fixed_quad_split(mu,kp,ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/16/np.pi**3
    array=np.array([])
    for i in ell:
        #print (i)
        inside1=lambda Z2,Kpar,mu,kp,i: sp.integrate.fixed_quad(Integrand,1e-4,10,args=(Z2,Kpar,mu,kp,i,),n=100)[0]
        inside2=lambda Kpar,mu,kp,i: sp.integrate.fixed_quad(inside1,1e-4,10,args=(Kpar,mu,kp,i,),n=100)[0]
        for j in mu:
            for m in kp:
                outside=sp.integrate.fixed_quad(inside2,1e-4,10,args=(i,j,m,),n=100)[0]
                array=np.append(array,const*outside)
    return array

mu=np.linspace(-0.9999,0.9999,n_points)
kp=np.geomspace(1e-4,10,n_points)
#reshape=np.reshape(func_fixed_quad_split(mu,kp,ell),(n_points,n_points,n_points))
#res=sp.integrate.trapz(sp.integrate.trapz(reshape,mu,axis=0),kp,axis=0)
#plt.plot(ell,ell*(ell+1)*res/2/np.pi)
#plt.show()

n=100
Mu=np.linspace(-1,1,n)
Kp=np.geomspace(1e-5,1e-1,n)
Z=np.geomspace(1e-4,10,n)
ell=np.geomspace(1e-2,1e4,n)
mu,kp,z=np.meshgrid(Mu,Kp,Z)
k=ell/chi(z)
K=np.sqrt(k**2+kp**2-2*k*kp*mu)
#print (K.min())
#print (K.max())

#plt.plot(ell,ell*(ell+1)*func_fixedquad_kp_first(ell)/2/np.pi)
#plt.show()

#plt.plot(ell,ell*(ell+1)*func_fixedquad_mu_first(ell)/2/np.pi)
#plt.show()

def func_quad(ell):
    err=np.array([])
    array=np.array([])
    for i in ell:
        #print (i)
        int1=lambda kp,z1,z2,kpar,i: sp.integrate.quadrature(Integrand,-0.9999,0.9999,args=(kp,z1,z2,kpar,i, ),tol=1,vec_func=False)[0]
        int2=lambda z1,z2,kpar,i: sp.integrate.quadrature(int1,k_min,10,args=(z1,z2,kpar,i, ),tol=1,vec_func=False)[0]
        int3=lambda z2,kpar,i: sp.integrate.quadrature(int2,1e-4,10,args=(z2,kpar,i,),tol=1,vec_func=False)[0]
        int4=lambda kpar,i: sp.integrate.quadrature(int3,1e-4,10,args=(kpar,i,),tol=1,vec_func=False)[0]
        int5=sp.integrate.quadrature(int4,1e-6,0.1,args=(i,),tol=1,vec_func=False)[0]
        error=sp.integrate.quadrature(int4,1e-6,0.1,args=(i,),tol=1,vec_func=False)[1]
        array=np.append(array,int5)
        err=np.append(err,error)
    return array,err



#plt.plot(ell,ell*(ell+1)*func_quad(ell)[0]/2/np.pi)
#plt.show()

#print (func_quad(ell)[1])

def kSZ_auto_single_bin(ell,z_1,delta_z1):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/(16*np.pi**3)
    Kpar=np.geomspace(1e-10,1e-1,n_points)
    Mu=np.linspace(-0.9999,0.9999,n_points)
    Kp=np.geomspace(1e-4,10,n_points)
    kpar,mu,kp=np.meshgrid(Kpar,Mu,Kp)
    z_2=z_1
    delta_z2=delta_z1
    chi_z1=chi(z_1)
    f_z1=uf.f(z_1)
    f_z2=uf.f(z_2)
    D_z1=uf.D_1(z_1)
    D_z2=uf.D_1(z_2)
    tau_z1=uf.tau_inst(z_1)
    tau_z2=uf.tau_inst(z_2)
    redshifts=delta_z1*delta_z2*(1+z_1)*np.exp(-tau_z1)/chi_z1**2*f_z1*D_z1**2*f_z2*D_z2**2*(1+z_2)*np.exp(-tau_z2)
    Cl=np.array([])
    for i in ell:
        k_perp=i/chi_z1
        k=np.sqrt(k_perp**2+kpar**2)
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
        theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
        theta_K=kpar/K/k*(k-kp*mu)-kp*k_perp*np.sqrt(1-mu**2)/k/K
        integrand=Mps_interpf(kp)*Mps_interpf(K)*kp**2*theta_kp*(theta_kp/kp**2)#+theta_K/K/kp)
        integral=const*redshifts*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand,Kpar,axis=0),Mu,axis=0),Kp,axis=0)
        Cl=np.append(Cl,integral)
    return Cl

plt.loglog(ell,ell*(ell+1)*kSZ_auto_single_bin(ell,1,0.3)/2/np.pi,'b')
plt.loglog(ell,ell*(ell+1)*kSZ_auto_single_bin(ell,2,0.6)/2/np.pi,'r')
plt.ylim(1e-14,1e-9)
plt.xlabel('l')
plt.ylabel(r'$\rm{(l(l+1)Cl^{OV}}/(2 \pi)[\mu K^2])}$')
plt.show()

#Cross corr
def Integrand_crosscorr_full(Mu,Kp,Z2,Kpar,z1,ell):
    mu,kp,z2,kpar=np.meshgrid(Mu,Kp,Z2,Kpar)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    T_mean_z1=uf.T_mean(z1)
    tau_z2=uf.tau_inst(z2)
    r_z1=uf.r(z1)
    k_perp=ell/chi_z1
    k=np.sqrt(k_perp**2+kpar**2)
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
    mu_k_21_sq=kpar**2/(kpar**2+ell**2/chi(z1)**2)
    theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
    theta_K=kpar/K/k*(k-kp*mu)-kp*k_perp*np.sqrt(1-mu**2)/k/K
    const=1e6*T_rad/cc.c_light_Mpc_s*x*(sigma_T*rho_g0/(mu_e*m_p))*T_mean_z1*(1+f_z1*mu_k_21_sq)/chi(z1)**2/r_z1/16/np.pi**3
    integrand=const*kp**2*theta_kp*(theta_kp/kp**2+theta_K/kp/K)
    full=integrand*Mps_interpf(kp)*Mps_interpf(K)*(1+z2)*H(z1)/(1+z1)*np.exp(-tau_z2)*f_z1*f_z2*D_z1**2*D_z2**2*np.cos(kpar*(chi_z1-chi_z2))/chi(z1)**2
    return full


def Integrand_crosscorr_part(Mu,Kp,Z2,Kpar,z1,ell):
    mu,kp,z2,kpar=np.meshgrid(Mu,Kp,Z2,Kpar)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    T_mean_z1=uf.T_mean(z1)
    tau_z2=uf.tau_inst(z2)
    r_z1=uf.r(z1)
    k_perp=ell/chi_z1
    k=np.sqrt(k_perp**2+kpar**2)
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
    mu_k_21_sq=kpar**2/(k**2)
    theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
    const=1e6*T_rad/cc.c_light_Mpc_s*x*(sigma_T*rho_g0/(mu_e*m_p))*T_mean_z1*(1+f_z1*mu_k_21_sq)/chi(z1)**2/r_z1/16/np.pi**3
    integrand=const*theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)
    integrand=const*theta_kp**2*Mps_interpf(kp)*Mps_interpf(k)
    full=integrand*(1+z2)*H(z1)/(1+z1)*np.exp(-tau_z2)*f_z1*f_z2*D_z1**2*D_z2**2*np.cos(kpar*(chi_z1-chi_z2))/chi(z1)**2
    return full

def func_quad_crosscorr(ell,z1):
    err=np.array([])
    array=np.array([])
    for i in ell:
        #print (i)
        int1=lambda kp,z2,kpar,z1,i: sp.integrate.quadrature(Integrand_crosscorr_part,-0.9999,0.9999,args=(kp,z2,kpar,z1,i, ),vec_func=False)[0]
        int2=lambda z2,kpar,z1,i: sp.integrate.quadrature(int1,k_min,1e-2,args=(z2,kpar,z1,i,),vec_func=False)[0]
        int3=lambda kpar,z1,i: sp.integrate.quadrature(int2,1e-4,10,args=(kpar,z1,i,),vec_func=False)[0]
        int4=sp.integrate.quadrature(int3,1e-6,2e-4,args=(z1,i,),vec_func=False)[0]
        error=sp.integrate.quadrature(int3,1e-6,2e-4,args=(z1,i,),vec_func=False)[1]
        err=np.append(err,error)
        array=np.append(array,int4)
    return array,err

#print (func_quad_crosscorr(ell,1)[1])



#Moved cross corr bispec in squeezed limit to a bispec.py file

def func_nquad(ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s/16/np.pi**3
    array=np.array([])
    for i in ell:
        #print (i)
        integral=sp.integrate.nquad(lambda Mu,Kp,Z1,Z2,Kpar: Integrand(Mu,Kp,Z1,Z2,Kpar,i),[[-0.9999,0.9999],[1e-4,10],[1e-4,10],[1e-4,10],[1e-6,1e-4]],opts={'epsabs':1,'epsrel':1})
        array=np.append(array,const*integral[0])
    return array

OV_z_min=lkSZ.C_l_mu_integral(ell,0.8)
OV_full_signal=lkSZ.C_l_mu_integral(ell,1e-4)
#np.savetxt('OV_full_signal.out',(ell,OV_full_signal))

OV_z_max=lkSZ.C_l_mu_integral(ell,1.1)
#np.savetxt('OV_1.1.out',(ell,OV_z_max))
'''
ell_new,OV_full_signal=np.genfromtxt('/home/zahra/python_scripts/kSZ_21cm_signal/OV_full_signal.out')
interp_OV_full_signal = interp1d(ell_new, OV_full_signal, bounds_error=False)
'''
ell_new,OV_z_min=np.genfromtxt('/home/zahra/python_scripts/kSZ_21cm_signal/OV_0.8.out')
interp_OV_min = interp1d(ell_new, OV_z_min, bounds_error=False)

#plt.plot(ell,interp_OV_min(ell))
#plt.show()

ell_new,OV_z_max=np.genfromtxt('/home/zahra/python_scripts/kSZ_21cm_signal/OV_1.1.out')
interp_OV_max = interp1d(ell_new, OV_z_max, bounds_error=False)
#plt.plot(ell,interp_OV_max(ell))
#plt.show()

diff=OV_z_min-OV_z_max


#plt.plot(ell,ell*(ell+1)*diff/2/np.pi,'r')
#plt.plot(ell,ell*(ell+1)*kSZ_auto_single_bin(ell,0.9,0.2)/2/np.pi,'b')
#plt.show()

plt.plot(ell,ell*(ell+1)*crosscorr_squeezedlim(ell,0.95,0.3)/2/np.pi,'k')
plt.plot(ell,ell*(ell+1)*nH.Cl_21_momentum(0.95,ell)/2/np.pi,'m')
plt.plot(ell,ell*(ell+1)*diff/2/np.pi,'b')
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)C_l[\mu K^2]/2\pi}$')
black_patch = mpatches.Patch(color='black', label='Squeezed bispectrum')
magenta_patch = mpatches.Patch(color='magenta', label='21 cm Momentum signal')
blue_patch=mpatches.Patch(color='blue',label='OV signal')
plt.legend(handles=[black_patch,magenta_patch,blue_patch])
plt.show()

#plt.plot(ell,ell*(ell+1)*func_nquad(ell)/2/np.pi)
#plt.show()

#print (func_quad(ell)[1])
plt.plot(ell,ell*(ell+1)*auto_nonlimb/2/np.pi,'m')
plt.plot(ell,ell*(ell+1)*OV/2/np.pi,'k')
black_patch = mpatches.Patch(color='black', label='limber OV')
magenta_patch = mpatches.Patch(color='magenta', label='non-limber OV')
plt.legend(handles=[black_patch,magenta_patch])
plt.ylabel(r'$\rm{(l(l+1)Cl^{OV}}/(2 \pi)[\mu K^2])}$')
plt.xlabel('l')
plt.show()

#print (func_quad_crosscorr(ell,1,277)[1])
plt.plot(ell,ell*(ell+1)*func_quad_crosscorr(ell,1)[0]/2/np.pi)
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)Cl^{21-OV}}/(2 \pi)[\mu K^2]}$')
plt.show()


#Plot for the binning of the non limber kSZ signal

def k_perp(ell,z):
    k_perp=np.array([])
    for i in ell:
        k_perp_single=i/chi(z)
        k_perp=np.append(k_perp,k_perp_single)
    return k_perp

def kpar_min(z):
    kpar=1/chi(z)          #kpar_min=2*pi/(r*delta_nu)
    return kpar

plt.plot(ell,k_perp(ell,0.95),'r')
plt.plot(ell,ell*kpar_min(0.95)/ell,'b')
plt.xlabel('l')
plt.ylabel('Contributions from k components')
red_patch = mpatches.Patch(color='red', label=r'$\rm k_{\perp}$')
blue_patch = mpatches.Patch(color='blue', label=r'$\rm k^{min}_\parallel$')
plt.legend(handles=[blue_patch,red_patch])
plt.show()

def func_single_ell(ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/(2*np.pi)**2/16/np.pi**3
    int1=lambda kp,z1,z2,kpar,ell: sp.integrate.quadrature(Integrand,-0.9999,0.9999,args=(kp,z1,z2,kpar,ell, ),tol=1e-7,vec_func=False)[0]
    int2=lambda z1,z2,kpar,ell: sp.integrate.quadrature(int1,k_min,k_max,args=(z1,z2,kpar,ell, ),tol=1e-7,vec_func=False)[0]
    int3=lambda z2,kpar,ell: sp.integrate.quadrature(int2,1e-4,10,args=(z2,kpar,ell,),tol=1e-7,vec_func=False)[0]
    int4=lambda kpar,ell: sp.integrate.quadrature(int3,1e-4,10,args=(kpar,ell,),tol=1e-7,vec_func=False)[0]
    int5=const*sp.integrate.quadrature(int4,1e-4,1e-2,args=(ell,),tol=1e-7,vec_func=False)[0]
    err=sp.integrate.quadrature(int4,1e-4,1e-2,args=(ell,),tol=1e-7,vec_func=False)[1]
    return int5,err

#print (func_quad(ell))
#print (func_single_ell(2000)[1])
#print (2000*(2000+1)*func_single_ell(2000)[0]/2/np.pi)

def Integrand_cos_and_redshift(Z1,Z2,kpar):
    z1,z2=np.meshgrid(Z1,Z2)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=uf.tau_inst(z1)
    tau_z2=uf.tau_inst(z2)
    integrand=(1+z1)*(1+z2)*np.exp(-tau_z1)*np.exp(-tau_z2)*f_z1*f_z2*D_z1**2*D_z2**2/chi_z1**2*np.cos(kpar*(z1-z2))
    return integrand

def func_cos(kpar):
    array=np.array([])
    err=np.array([])
    for i in kpar:
        integral1=lambda z2,i:sp.integrate.quadrature(Integrand_cos_and_redshift,1e-4, 10, args=(z2,i, ),tol=1e-6,vec_func=False)[0]
        integral2=sp.integrate.quadrature(integral1, 1e-4, 10, args=(i,),tol=1e-6,vec_func=False)[0]
        error=sp.integrate.quadrature(integral1, 1e-4, 10, args=(i,),tol=1e-6,vec_func=False)[1]
        err=np.append(err,error)
        array=np.append(array,integral2)
    return array,err

#print (func_cos(kpar)[1])
#y_cos=2/kpar**2*(1-np.cos(kpar*(10-1e-4)))
#plt.loglog(kpar,y_cos)
#plt.plot(kpar,func_cos(kpar)[0])
#plt.show()

def Limber_redshift(z):
    chi_z=chi(z)
    f_z=uf.f(z)
    D_z=uf.D_1(z)
    H_z=uf.H(z)
    tau_z=uf.tau_inst(z)
    return np.exp(-2*tau_z)*f_z**2*D_z**4*H_z*(1+z)**2/chi_z**2

sp.integrate.quadrature(Limber_redshift,1e-4,10)

def Integrand_limber(Mu,Kp,Z,ell):
    mu,kp,z=np.meshgrid(Mu,Kp,Z)
    chi_z=chi(z)
    f_z=uf.f(z)
    D_z=uf.D_1(z)
    H_z=uf.H(z)
    tau_z=uf.tau_inst(z)
    k=ell/chi_z
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
    integrand=k*(k-2*kp*mu)/K**2*Mps_interpf(kp)*Mps_interpf(K)*np.exp(-2*tau_z)*f_z**2*D_z**4*H_z*(1-mu**2)*(1+z)**2/chi_z**2
    return integrand

#k*(k-2*kp*mu)/K**2*Mps_interpf(kp)*Mps_interpf(K)*np.exp(-2*tau_z)*f_z**2*D_z**4*H_z*(1-mu**2)*(1+z)**2/chi_z**2

#k*(k-2*kp*mu)/K**2*Mps_interpf(kp)*Mps_interpf(K)*np.exp(-2*tau_z)*f_z**2*D_z**4*H_z
def func_limber(ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s/8/np.pi**2
    array=np.array([])
    err=np.array([])
    for i in ell:
        #print (i)
        int1=lambda kp,z,i: sp.integrate.quadrature(Integrand_limber,-0.9999,0.9999,args=(kp,z,i, ),vec_func=False)[0]
        int2=lambda kp,i: sp.integrate.quadrature(int1,1e-4,10,args=(kp,i, ),vec_func=False)[0]
        int3=sp.integrate.quadrature(int2,1e-4,10,args=(i,),vec_func=False)[0]
        error=sp.integrate.quadrature(int2,1e-4,10,args=(i,),vec_func=False)[1]
        err=np.append(err,error)
        array=np.append(array,const*int3)
    return array,err

#print (func_limber(ell)[1])
#plt.plot(ell,ell*(ell+1)*func_limber(ell)[0]/2/np.pi)
#plt.show()


def func_limber_one_ell(ell):
    #const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s/8/np.pi**2
    int1=lambda kp,z,ell: sp.integrate.quadrature(Integrand_limber,-0.9999,0.9999,args=(kp,z,ell, ),tol=1e-4,maxiter=50,vec_func=False)[0]
    int2=lambda z,ell: sp.integrate.quadrature(int1,1e-4,10,args=(z,ell, ),tol=1e-4,maxiter=50,vec_func=False)[0]
    int3=sp.integrate.quadrature(int2,1e-4,10,args=(ell,),tol=1e-4,maxiter=50,vec_func=False)[0]
    err=sp.integrate.quadrature(int2,1e-4,10,args=(ell,),tol=1e-4,maxiter=50,vec_func=False)[1]
    return int3,err

#print (func_limber_one_ell(2000)[0])


def func_limber_nquad(ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s/8/np.pi**2
    array=np.array([])
    err=np.array([])
    add_info=np.array([])
    for i in ell:
        #print (i)
        integral=sp.integrate.nquad(lambda Mu,Kp,Z: Integrand_limber(Mu,Kp,Z,i),[[-0.9999,0.9999],[1e-10,10],[1e-4,10]],full_output=True)
        array=np.append(array,const*integral[0])
        err=np.append(err,integral[1])
        add_info=np.append(add_info,integral[2])
    return array,err,add_info

#plt.plot(ell,ell*(ell+1)*func_limber_nquad(ell)[0]/2/np.pi)
#plt.show()

def limber_nquad_func_of_kp(ell,Kp):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s/8/np.pi**2
    array=np.array([])
    for i in Kp:
        integral=const*sp.integrate.nquad(lambda Mu,Z: Integrand_limber(Mu,i,Z,ell),[[-0.9999,0.9999],[1e-4,10]],full_output=True)[0]
        array=np.append(array,integral)
    return array

Kp=np.geomspace(1e-10,100,100)
plt.loglog(Kp,1e6*limber_nquad_func_of_kp(1e3,Kp))

def Integrand_doppler(Z1,Z2,Kpar,ell):
    z1,z2,kpar=np.meshgrid(Z1,Z2,Kpar)
    chi_1=uf.chi(z1)
    f_1=uf.f(z1)
    D_1=uf.D_1(z1)
    tau_1=uf.tau_inst(z1)
    f_2=uf.f(z2)
    D_2=uf.D_1(z2)
    chi_2=uf.chi(z2)
    tau_2=uf.tau_inst(z2)
    k=np.sqrt(kpar**2+ell**2/chi_1**2)
    mu_k=kpar/k
    integrand=(1+z1)/chi_1**2*np.exp(-tau_1)*(1+z2)*np.exp(-tau_2)*D_1*D_2*f_1*f_2*mu_k**2*Mps_interpf_div_sq(k)*np.cos(kpar*(chi_1-chi_2))
    return integrand

def func_doppler_quadrature(ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/2/np.pi
    array=np.array([])
    err=np.array([])
    for i in ell:
        print (i)
        int1=lambda z2,kpar,i: sp.integrate.quadrature(Integrand_doppler,1e-4,10,args=(z2,kpar,i, ),tol=1,vec_func=False)[0]
        int2=lambda kpar,i: sp.integrate.quadrature(int1,1e-4,10,args=(kpar,i, ),tol=1,vec_func=False)[0]
        int3=sp.integrate.quadrature(int2,1e-4,1e-2,args=(i,),tol=1,vec_func=False)[0]
        error=sp.integrate.quadrature(int2,1e-4,1e-2,args=(i,),tol=1,vec_func=False)[1]
        err=np.append(err,error)
        array=np.append(array,const*int3)
    return array,err

def func_doppler_nquad(ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/np.pi/2
    array=np.array([])
    err=np.array([])
    for i in ell:
        print (i)
        integral=sp.integrate.nquad(lambda Z1,Z2,Kpar: Integrand_doppler(Z1,Z2,Kpar,i),[[1e-4,10],[1e-4,10],[1e-6,1e-3]],opts={'epsabs':1,'epsrel':1},full_output=False)
        array=np.append(array,const*integral[0])
        err=np.append(err,integral[1])
    return array,err

#plt.loglog(ell,ell*(ell+1)*func_doppler_nquad(ell)[0]/2/np.pi)
#plt.show()






#print (func_doppler_quadrature(ell)[1])
#print (func_doppler_21(0.95,0.95,ell)[1])
plt.semilogy(ell,func_doppler_21(0.95,0.95,ell)[0]/2/np.pi)
plt.xlabel('l')
plt.ylabel(r'$\rm{Cl^{Doppler}}/(2 \pi)[\mu K^2])}$')
plt.show()

def func_limber_nquad_single_ell(ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s/8/np.pi**2
    integral=sp.integrate.nquad(lambda Mu,Z,Kp: Integrand_limber(Mu,Z,Kp,ell),[[-0.9999,0.9999],[1e-4,10],[1e-4,1e-2]],full_output=True)[0]
    return const*integral

#print (func_doppler_nquad(ell)[1])
#plt.loglog(ell,ell*(ell+1)*func_doppler_nquad(ell)[0]/2/np.pi)
#plt.show()


def func_no_kpar_int_fixed_quad(ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s/8/np.pi**2
    array=np.array([])
    for i in ell:
        int1=lambda Mu,Z,i: sp.integrate.fixed_quad(Integrand_limber,1e-4,10,args=(Mu,Z,i, ),n=60)[0]
        int2=lambda Z,i: sp.integrate.fixed_quad(int1,-0.9999,0.9999,args=(Z,i, ),n=60)[0]
        int3=sp.integrate.fixed_quad(int2,1e-4,10,args=(i,),n=60)[0]
        array=np.append(array,const*int3)
    return array


#plt.plot(ell,ell*(ell+1)*func(ell)/2/np.pi)
#plt.show()
#plt.plot(ell,ell*(ell+1)*func_no_kpar_int_fixed_quad(ell)/2/np.pi)
#plt.show()

#checking integrals
#Checking k' integral with ell=3000,z=4,mu=0.5


kpar=np.linspace(1e-6,2e-4,100)
z1=10
def Doppler_kSZ_reionisation(ell):
    chi_1=uf.chi(z1)
    f_1=uf.f(z1)
    D_1=uf.D_1(z1)
    tau_1=uf.tau_inst(z1)
    H_1=uf.H(z1)
    kpar=np.geomspace(1e-2,1e4,n_points)
    array=np.array([])
    for i in ell:
        k=np.sqrt(kpar**2+i**2/chi(z1)**2)
        integral=sp.integrate.trapz(Mps_interpf(k)/k**4,kpar)
        array=np.append(array,integral)
    return array

#plt.plot(ell,ell*(ell+1)*Doppler_kSZ_reionisation(ell)/2/np.pi)
#plt.show()

def Integrand_kp_check(kp):
    mu=0.5
    z=4
    ell=3000
    chi_z=chi(z)
    f_z=uf.f(z)
    D_z=uf.D_1(z)
    H_z=uf.H(z)
    tau_z=uf.tau_inst(z)
    k=ell/chi_z
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
    integrand=k*(k-2*kp*mu)*(1-mu**2)/K**2*Mps_interpf(kp)*Mps_interpf(K)*(1+z)**2*np.exp(-2*tau_z)*f_z**2*D_z**4/chi_z**2*H_z
    return integrand


#print (sp.integrate.quadrature(Integrand_kp_check,1e-4,10,tol=1e-23,maxiter=400))

kp=np.geomspace(1e-4,10)
#print (sp.integrate.trapz(Integrand_kp_check(kp),kp))

#print (sp.integrate.quad(Integrand_kp_check,1e-4,10))

l_min=1e-2
chi_min=chi(1e-4)
kpar_min=1e-4
k_min_nl=np.sqrt(l_min**2/chi_min**2+kpar_min**2)
K_min_nl=k_min_nl**2*(1+k_min**2/k_min_nl**2-2*k_min/k_min_nl*(-1))
print (K_min_nl)

l_max=1e4
chi_max=chi(10)
kpar_max=1e-2
k_max_nl=np.sqrt(l_max**2/chi_max**2+kpar_max**2)
K_max_nl=k_max_nl**2*(1+k_max**2/k_max_nl**2-2*k_max/k_max_nl)
print (K_max_nl)

def F(Z,kpar,ell):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-0.9999,0.9999,n_points)
    mu,kp,z1,z2=np.meshgrid(Mu,Kp,Z,Z)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=tau(z1)
    tau_z2=tau(z2)
    array=np.array([])
    for i in ell:
        for j in kpar:
            k_perp=i/chi_z1
            k=np.sqrt(k_perp**2+j**2)
            K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
            integrand=sp.integrate.trapz(sp.integrate.trapz((j*mu*kp**2/k+k_perp*kp**2
            *np.sqrt(1-mu**2)/k)*((j*mu+k_perp*np.sqrt(1-mu**2))*(k-2*kp*mu)/kp**2
            /(k**2+kp**2-2*k*kp*mu)+j/kp/(K**2))*Mps_interpf(kp)
            *Mps_interpf(K)*(1+z1)*(1+z2)*np.exp(-tau_z1)*np.exp(-tau_z2)
            *f_z1*f_z2*D_z1**2*D_z2**2/chi_z1**2*np.cos(j*(chi_z1-chi_z2)),Mu,axis=0),Kp,axis=0)
            array=np.append(array,integrand)
    return array

def F_crosscorr(ell,kpar,Z,zi):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    mu,kp,z1=np.meshgrid(Mu,Kp,Z)
    f=uf.f
    D=uf.D_1
    tau=uf.tau_inst
    H=uf.H
    array=np.array([])
    for i in ell:
        for j in kpar:
            k_perp=i/chi(z1)
            k=np.sqrt(k_perp**2+j**2)
            integrand=sp.integrate.trapz(sp.integrate.trapz((j*mu*kp**2/k+k_perp*kp**2
            *np.sqrt(1-mu**2)/k)*((j*mu+k_perp*np.sqrt(1-mu**2))*(k-2*kp*mu)/kp**2
            /(k**2+kp**2-2*k*kp*mu)+j/kp/(k**2+kp**2-2*k*kp*mu))*Mps_interpf(kp)
            *Mps_interpf(np.sqrt(k**2+kp**2-2*k*kp*mu))*(1+z1)/(1+zi)*np.exp(-tau(z1))
            *f(z1)*f(zi)*D(z1)**2*D(zi)**2/chi(z1)**2*H(zi)*np.cos(j*(chi(z1)-chi(zi))),Kp,axis=0),Mu,axis=0)
            array=np.append(array,integrand)
    return array


def F_no_kpar_int(ell,j,Z):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    mu,kp,z1,z2=np.meshgrid(Mu,Kp,Z,Z)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=tau(z1)
    tau_z2=tau(z2)
    k_perp=ell/chi_z1
    k=np.sqrt(k_perp**2+j**2)
    integrand=sp.integrate.trapz(sp.integrate.trapz((j*mu*kp**2/k+k_perp*kp**2
    *np.sqrt(1-mu**2)/k)*((j*mu+k_perp*np.sqrt(1-mu**2))*(k-2*kp*mu)/kp**2
    /(k**2+kp**2-2*k*kp*mu)+j/kp/(k**2+kp**2-2*k*kp*mu))*Mps_interpf(kp)
    *Mps_interpf(np.sqrt(k**2+kp**2-2*k*kp*mu))*(1+z1)*(1+z2)*np.exp(-tau_z1)*np.exp(-tau_z2)
    *f_z1*f_z2*D_z1**2*D_z2**2/chi_z1**2,Kp,axis=0),Mu,axis=0)
    return integrand

Z=uf.z

#print (F(ell,kpar,Z),np.shape(F(ell,kpar,Z)))
F_reshaped=np.reshape(F(Z,kpar,ell),(n_points,n_points,n_points,n_points))
F_reshaped_trans=np.transpose(F_reshaped)

F_reshaped_crosscorr=np.reshape(F_crosscorr(ell,kpar,Z,1),(n_points,n_points,n_points))
F_reshaped_trans_crosscorr=np.transpose(F_reshaped_crosscorr)
#np.savez('F_non_limber_100',F=F_reshaped_trans)

#npzfile=np.load('F_nonlimber.npz')
def C_l_no_kpar_int(ell,j):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/(2*np.pi)**2*2*np.pi/2
    array=np.array([])
    for i in ell:
        integral=sp.integrate.trapz(sp.integrate.trapz(const*F_no_kpar_int(i,j,Z),Z,axis=0),Z,axis=0)
        array=np.append(array,integral)
    return array


def C_l(ell):
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/16/np.pi**3
    integral=sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(const*F_reshaped_trans,Z,axis=0),Z,axis=0),kpar,axis=0)
    return integral

#plt.plot(ell,ell*(ell+1)*C_l(ell)/2/np.pi)
#plt.show()

#print (C_l(ell))

def C_l_crosscorr(ell,zi,y):
    kpar_21=uf.kpar(y,zi)
    mu_k_21=kpar_21/(np.sqrt(kpar**2+ell**2/chi(zi)**2))
    const=1e12*T_rad/cc.c_light_Mpc_s*x*(sigma_T*rho_g0/(mu_e*m_p))*uf.T_mean(zi)*(1+uf.f(zi)*mu_k_21)/chi(zi)**2/uf.r(zi)/(2*np.pi)**2*2*np.pi/2
    integral=sp.integrate.trapz(sp.integrate.trapz(const*F_reshaped_trans_crosscorr,Z,axis=0),kpar,axis=0)
    return integral


#plt.plot(ell,-1*ell*(ell+1)*C_l_crosscorr(ell,1,277)/2/np.pi)
#plt.ylabel(r'$\rm{l(l+1)Cl^{Cross corr}}/(2 \pi)[\mu K^2]$')
#plt.xlabel('l')
#plt.show()



def C_l_full(ell):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    Z=uf.z
    mu,kp,z1,z2,kpar=np.meshgrid(Mu,Kp,Z,Z,Kpar)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=tau(z1)
    tau_z2=tau(z2)
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/(2*np.pi)**2*2*np.pi/2
    C_l=np.array([])
    for i in ell:
        #print (i)
        k_perp=i/chi_z1
        k=np.sqrt(k_perp**2+kpar**2)
        integrand=[const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz((kpar*mu*kp**2/k
        +k_perp*kp**2*np.sqrt(1-mu**2)/k)
        *((kpar*mu+k_perp*np.sqrt(1-mu**2))*(k-2*kp*mu)
        /kp**2/(k**2+kp**2-2*k*kp*mu)
        +2*kpar/kp/(k**2+kp**2-2*k*kp*mu))
        *Mps_interpf(kp)*Mps_interpf(np.abs(k-kp))*(1+z1)
        *(1+z2)*np.exp(-tau_z1)*np.exp(-tau_z2)*f_z1*f_z2*D_z1**2*D_z2**2/chi_z1**2
        *np.cos(kpar*(chi_z1-chi_z2)),Kp,axis=0),Mu,axis=0),Z,axis=0),Z,axis=0),Kpar,axis=0)]
        C_l=np.append(C_l,integrand)
    return C_l


def I_nonlimber(ell,kpar):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    Z=uf.z
    mu,kp,z1,z2=np.meshgrid(Mu,Kp,Z,Z)
    chi=uf.chi
    I=np.array([])
    for i in ell:
        #print (i)
        integral=[sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz((j*mu*
                                                                kp**2/np.sqrt(i**2/chi(z2)**2+j**2)
        +i*kp**2*np.sqrt(1-mu**2)/chi(z2)/np.sqrt(i**2/chi(z2)**2+j**2))
        *((j*mu+i*np.sqrt(1-mu**2))*(np.sqrt(i**2/chi(z2)**2+j**2)-2*kp*mu)
        /kp**2/(i**2/chi(z2)**2+j**2+kp**2-2*np.sqrt(i**2/chi(z2)**2+j**2)*kp*mu)
        +2*j/kp/(i**2/chi(z2)**2+j**2+kp**2-2*np.sqrt(i**2/chi(z2)**2+j**2)*kp*mu))
        *z1*Mps_interpf(kp)*Mps_interpf(np.abs(np.sqrt(i**2/chi(z2)**2+j**2)-kp))
        ,Kp,axis=0),Mu,axis=0),Z,axis=0),Z,axis=0) for j in kpar]
        I=np.append(I,integral)
    return I

#print (I_nonlimber(ell,kpar))

def I_nonlimber_double_forloop(ell):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    mu,kp=np.meshgrid(Mu,Kp)
    chi=uf.chi
    I=np.array([])
    for i in ell:
        #print (i)
        integrand=[sp.integrate.trapz(sp.integrate.trapz((j*mu*kp**2/np.sqrt(i**2/chi(m)**2+j**2)
        +i*kp**2*np.sqrt(1-mu**2)/chi(m)/np.sqrt(i**2/chi(m)**2+j**2))
        *((j*mu+i*np.sqrt(1-mu**2))*(np.sqrt(i**2/chi(m)**2+j**2)-2*kp*mu)/kp**2/(i**2/chi(m)**2+j**2+kp**2
           -2*np.sqrt(i**2/chi(m)**2+j**2)*kp*mu)
        +2*j/kp/(i**2/chi(m)**2+j**2+kp**2-2*np.sqrt(i**2/chi(m)**2+j**2)*kp*mu))
        *Mps_interpf(kp)*Mps_interpf(np.abs(np.sqrt(i**2/chi(m)**2+j**2)-kp))
        ,Kp,axis=0),Mu,axis=0) for j in kpar for m in z]
        integral=sp.integrate.trapz(sp.integrate.trapz(integrand,kpar,axis=0),z,axis=0)
        I=np.append(I,integral)
    return I

def I_limber(ell):
    Kp=np.geomspace(k_min,k_max,n_points)
    Mu=np.linspace(-1,1,n_points)
    Z=uf.z
    mu,kp,kpar,z=np.meshgrid(Mu,Kp,Kpar,Z)
    chi=uf.chi(z)
    I=np.array([])
    for i in ell:
        #print (i)
        k_perp=i**2/chi**2
        k=np.sqrt(k_perp**2+kpar**2)
        integral=[sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(k*(k-2*kp*mu)
        *(1-mu**2)/(k**2+kp**2-2*k*kp*mu)
        *Mps_interpf(kp)*Mps_interpf(np.abs(k-kp))
        ,Kp,axis=0),Mu,axis=0),Kpar,axis=0),Z,axis=0)]
        I=np.append(I,integral)
    return I

def C_l_old(ell,kpar):
    z1,z2=np.meshgrid(z,z)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=tau(z1)
    tau_z2=tau(z2)
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2
    C_l=np.array([])
    for i in ell:
        #print (i)
        #integrand=const*(1+z1)*(1+z2)*np.exp(-tau(z1))*np.exp(-tau(z2))*Delta_b(i/chi(z))*uf.f(z1)*uf.f(z2)*uf.D_1(z1)**2*uf.D_1(z2)**2/uf.chi(z1)**2
        #print (np.shape(integrand))

        #integrand=np.resize(integrand,(n_points+1))

        #print (np.shape(integrand))
        Redshift_dep=[const*sp.integrate.trapz(sp.integrate.trapz((1+z1)*(1+z2)*np.exp(-tau_z1)
        *np.exp(-tau_z2)*F_non_limber_nokpar(i/chi_z2,j*(chi_z1-chi_z2))*f_z1*f_z2*D_z1**2*D_z2**2/chi_z1**2,z,axis=0)
    ,z,axis=0) for j in kpar]
        Integral=sp.integrate.trapz(Redshift_dep,kpar)
        #print (Redshift_dep)
        #print (Integral)
        C_l=np.append(C_l,Integral)
    return C_l

def C_l_final(ell):
    C_l=np.array([])
    for i in ell:
        Integral=sp.integrate.trapz(C_l_old(ell,kpar),kpar)
        C_l=np.append(C_l,Integral)
    return C_l

#plt.plot(ell,ell*(ell+1)*denom(ell)/2/np.pi)
#plt.show()
#arg=np.geomspace(1e-4,14e3,n_points)
#(j*mu/(i**2+j**2)+i/(i**2+j**2)*np.sin(np.arccos(mu)))*((j*mu+i*np.sin(np.arccos(mu)))*((np.sqrt(i**2+j**2))))
#integrand=[sp.integrate.trapz(sp.integrate.trapz(1/(i**2+j**2+kp**2-2*np.sqrt(i**2+j**2)
 #       *mu*kp)*((i**2+j**2+kp**2-2*np.sqrt(i**2+j**2)*kp*mu-1)*(j*mu+i*np.sin(np.arccos(mu)))+j)
  #      *np.cos(j*arg),Kp,axis=0),Mu,axis=0) for j in kpar]

#print (Delta_b(k))
#np.savetxt('Fval20_triple_int.out',(k,F(k,arg)))
#F=F(k,arg=10)

k_new,F=np.genfromtxt('C:\\Users\\zahra\\.spyder-py3\\Fval20_triple_int.out')
interp_F = interp1d(k_new, F, bounds_error=False)
#plt.loglog(k,interp_F)
#plt.show()
plt.loglog(k,F)
plt.show()
#plt.ylim(1e-13,1e3)
#plt.xlim(1e-2,10)

def C_l(ell):
    z1,z2=np.meshgrid(z,z)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    H_1=uf.H(z1)
    H_2=uf.H(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=uf.tau_inst(z1)
    tau_z2=uf.tau_inst(z2)
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/2/np.pi
    C_l=np.array([])
    for i in ell:
        #print (i)
        for j in kpar:
            Redshift_dep=[const*sp.integrate.trapz(sp.integrate.trapz((1+z1)*(1+z2)*np.exp(-tau_z1)
            *np.exp(-tau_z2)*F_non_limber_no_kpar(np.sqrt(i**2/chi_z2**2+j**2),j)*f_z1*f_z2*D_z1**2*D_z2**2/chi_z1**2*np.cos(j*(chi_z1-chi_z2)),z,axis=0),z,axis=0)]
            C_l=np.append(C_l,sp.integrate.trapz(Redshift_dep,kpar))
    return C_l

def C_l_limb(ell):
    z1=np.geomspace(1e-4,10,n_points)
    chi_z1=chi(z1)
    f_z1=uf.f(z1)
    H_1=uf.H(z1)
    D_z1=uf.D_1(z1)
    tau_z1=uf.tau_inst(z1)
    const=1e12*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s
    C_l=np.array([])
    for i in ell:
        #print (i)
        Redshift_dep=const*sp.integrate.trapz((1+z1)**2*np.exp(-2*tau_z1)
        *F_limber_y1(i/chi(z1),chi_z1)*f_z1**2*D_z1**4/chi_z1**2*H_1,z)
        C_l=np.append(C_l,Redshift_dep)
    return C_l


def C_l_integrand(ell,kpar):
    z1,z2=np.meshgrid(z,z)
    chi_z1=chi(z1)
    chi_z2=chi(z2)
    chi_z=uf.chi(z)
    H_z1=uf.H(z1)
    H_z2=uf.H(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=tau(z1)
    tau_z2=tau(z2)
    const=1e12*8*np.pi**2*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/(2*np.pi)
    C_l=np.array([])
    for i in ell:
        #integrand=const*(1+z1)*(1+z2)*np.exp(-tau(z1))*np.exp(-tau(z2))*Delta_b(i/chi(z))*uf.f(z1)*uf.f(z2)*uf.D_1(z1)**2*uf.D_1(z2)**2/uf.chi(z1)**2
        #print (np.shape(integrand))

        #integrand=np.resize(integrand,(n_points+1))
        #print (Delta_b(np.sqrt(kpar**2+i**2/chi_z1**2)))
        #print (np.shape(integrand))
        Redshift_dep=const*sp.integrate.trapz(sp.integrate.trapz((1+z1)*(1+z2)*np.exp(-tau_z1)
        *np.exp(-tau_z2)*Delta_b(np.sqrt(kpar**2+i**2/chi_z2**2))*np.cos(kpar*(chi_z1-chi_z2))*f_z1
        *f_z2*D_z1**2*D_z2**2*chi_z1,z,axis=0),z,axis=0)
        #print (Redshift_dep)
        #print (Integral)
        C_l=np.append(C_l,Redshift_dep/(2*i+1)**3)
    return C_l
def C_l_integrand_chi_int(ell,kpar):
    chi_z=uf.chi(z)
    chi_1,chi_2=np.meshgrid(chi_z,chi_z)
    z1=uf.zed(chi_1)
    z2=uf.zed(chi_2)
    H_z1=uf.H(z1)
    H_z2=uf.H(z2)
    f_z1=uf.f(z1)
    f_z2=uf.f(z2)
    D_z1=uf.D_1(z1)
    D_z2=uf.D_1(z2)
    tau_z1=tau(z1)
    tau_z2=tau(z2)
    const=1e12*8*np.pi**2*T_rad**2*x**2*(sigma_T*rho_g0/(mu_e*m_p))**2/cc.c_light_Mpc_s**2/(2*np.pi)
    C_l=np.array([])
    for i in ell:
        #integrand=const*(1+z1)*(1+z2)*np.exp(-tau(z1))*np.exp(-tau(z2))*Delta_b(i/chi(z))*uf.f(z1)*uf.f(z2)*uf.D_1(z1)**2*uf.D_1(z2)**2/uf.chi(z1)**2
        #print (np.shape(integrand))

        #integrand=np.resize(integrand,(n_points+1))
        #print (Delta_b(np.sqrt(kpar**2+i**2/chi_z1**2)))
        #print (np.shape(integrand))
        Redshift_dep=const*sp.integrate.trapz(sp.integrate.trapz((1+z1)*(1+z2)*np.exp(-tau_z1)
        *np.exp(-tau_z2)*Delta_b(np.sqrt(kpar**2+i**2/chi_2**2))*np.cos(kpar*(chi_1-chi_2))*f_z1
        *f_z2*D_z1**2*D_z2**2*chi_1*H_z1*H_z2,chi_z,axis=0),chi_z,axis=0)
        #print (Redshift_dep)
        #print (Integral)
        C_l=np.append(C_l,Redshift_dep/(2*i+1)**3)
    return C_l

#plt.plot(ell,ell*(ell+1)*C_l_limb(ell)/(2*np.pi))
plt.plot(ell,ell*(ell+1)*C_l(ell)/(2*np.pi))
#plt.plot(ell,I_limber(ell))
#plt.plot(ell,I_nonlimber(ell,kpar))
#plt.plot(ell,C_l_F(ell))
plt.show()
#print (ell*(ell+1)*C_l(ell)/(2*np.pi))
#print (ell*(ell+1)*C_l(ell,1)/(2*np.pi))
#print (ell*(ell+1)*C_l(ell,1e-1)/(2*np.pi))
#plt.plot(ell,ell*(ell+1)*C_l_old(ell)/(2*np.pi),'b')
#plt.plot(ell,ell*(ell+1)*C_l(ell,1)/(2*np.pi),'g')
#plt.plot(ell,ell*(ell+1)*C_l(ell,1e-1)/(2*np.pi),'r')
#plt.plot(ell,ell*(ell+1)*C_l(ell,10)/(2*np.pi),'k')
#plt.show()

print (C_l_integrand_chi_int(ell,0))
print (C_l_integrand_chi_int(ell,10))
plt.plot(ell,ell*(ell+1)*C_l_integrand_chi_int(ell,0)/(2*np.pi),'g')
plt.plot(ell,ell*(ell+1)*C_l_integrand_chi_int(ell,10)/(2*np.pi),'b')
plt.show()
plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,10),'b')
plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,0),'g')
plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,1),'r')
plt.plot(ell,ell*(ell+1)*C_l_integrand(ell,5),'c')
blue_patch = mpatches.Patch(color='blue', label='k=10')
green_patch = mpatches.Patch(color='green', label='k=0')
red_patch = mpatches.Patch(color='red', label='k=1')
cyan_patch = mpatches.Patch(color='cyan', label='k=5')
plt.legend(handles=[blue_patch,green_patch,red_patch,cyan_patch])
plt.show()
'''
