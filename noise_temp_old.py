import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import useful_functions as uf
import sys
sys.path.insert(0,'/home/zahra/hirax_tools/hirax_tools')
from array_config import HIRAXArrayConfig
from neutral_H_autocorr import Cl_21_func_of_y,Cl_21,Integrand_doppler_21cm,Cl_21_momentum_single,Cl_21,Cl_21_doppler,Cl_21_momentum_integrated
import scipy as sp
from scipy.integrate import trapz
from scipy.interpolate import interp1d,UnivariateSpline,splrep, splev
#import spline
from mpl_toolkits.mplot3d import Axes3D
import pylab
import constants as cc
import density as den
from decimal import Decimal
import pylab


cosmo=uf.cosmo
Mps_interpf=uf.Mps_interpf
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g #in units of g/Mpc^3
sigma_T=cc.sigma_T_Mpc
tau_r=0.055
z_r=10
T_rad=2.725 #In Kelvin

#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
#get matter power spec data
mu_e=1.14
m_p=cc.m_p_g #in grams
rho_g0=cosmo['omega_b_0']*rho_c
#plt.loglog(uf.kabs,Mps_interpf(uf.kabs))
#plt.show()


chi=uf.chi
r=uf.r

D=uf.D_1
H=uf.H
f=uf.f
c=cc.c_light_Mpc_s
T_mean=uf.T_mean
nu_21=1420. #MHz
z=1.26
n=70
delta_z=0.0015

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

ell_large=np.geomspace(1.e-10,1.e6,10000)

kperp_min_orig,kperp_max_orig=uf.ell_lims(z,6.,6*1024)/uf.chi(z)
kperp_min,kperp_max=kperp_min_orig,kperp_max_orig
#print (kperp_min,kperp_max)
kpar_min=uf.kpar_min(z,delta_z)
#print (kpar_min)
kperp_arr=np.geomspace(kperp_min,kperp_max,n)
kpar_arr=np.geomspace(kpar_min,5.,n)

ell_min,ell_max=uf.chi(z)*kperp_min,uf.chi(z)*kperp_max
ell=np.geomspace(ell_min,ell_max,n)

ell=np.geomspace(1.,1.e4,n)
kperp_arr=ell/chi(1.26)
kpar_arr=np.geomspace(1.e-4,.15,n)
y=kpar_arr*r(1.26)

#np.savetxt('Hirax_noise_z_1_Ddish_6_Dsep_7_geom.out',(ell_large,HiraxNoise(ell_large,6.,7.,1.)))
#ell_new,Hirax_noise_z_1=np.genfromtxt('/home/zahra/python_scripts/kSZ_21cm_signal/Hirax_noise_z_1_Ddish_6_Dsep_7.out')
Hirax_noise_z_1pt26_deltaz_pt2 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.26,0.2), bounds_error=False)
Hirax_noise_z_1pt26_deltaz_pt05 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.26,0.05), bounds_error=False)
Hirax_noise_z_1pt26_deltaz_pt0015 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.26,0.0015), bounds_error=False)


'''
#print (Hirax_noise_z_1pt26_deltaz_pt0015(ell)[4000:5000])
plt.loglog(ell,Hirax_noise_z_1pt26_deltaz_pt0015(ell),'o')
plt.ylabel('N(l)')
plt.xlabel('l')
plt.ylim(1.e-5,100.)
plt.show()
'''
def P_v_N_delta(z,kperp,kpar,Noise):
    k=np.sqrt(kperp**2+kpar**2)
    Pv_func_of_k=Mps_interpf(k)/k**4*kpar**2
    return Pv_func_of_k*Noise(uf.chi(z)*kperp)

def P_delta_N_v(z,kperp,kpar,Noise):
    k=np.sqrt(kperp**2+kpar**2)
    Nv_func_of_k=Noise(uf.chi(z)*kperp)/k**4*kpar**2
    return Nv_func_of_k*Mps_interpf(k)


def PN_integrals(ell,z_i,y ,delta_z, Noise):
    Kp=np.geomspace(1.e-10,1.,n)
    U=np.linspace(-.9999,.9999,n)
    Z_min=z_i-delta_z
    Z_max=z_i+delta_z
    Z=np.geomspace(Z_min,Z_max,n)
    u,kp,z1,z2=np.meshgrid(U,Kp,Z,Z)

    T_mean=uf.T_mean
    chi_z1=chi(z1)
    f=uf.f
    D=uf.D_1
    r=uf.r
    H=uf.H

    Kperp_arr=np.array([])
    kpar=y/r(z1)

    kp_perp=kp*np.sqrt(1-u**2)
    Noise_kp_perp=Noise(kp_perp*chi(z1))
    '''
    for i in range(len(Noise_kp_perp)):
        if Noise_kp_perp[i]>1.e4:
            Noise_kp_perp[i]=0.
    '''
    #print (Noise_kp_perp.min(),Noise_kp_perp.max(),'Noise of kp perp')
    mock_arr=np.geomspace(2.e-2,6.e-1,n)
    #print (Noise(mock_arr*chi(z1)).min(),Noise(mock_arr*chi(z1)).max(),'Noise of mock arr')
    kp_par=kp*u
    theta_kp=u

    P_deldel_P_vv=np.array([])
    P_delv_P_delv=np.array([])
    P_vv_N_deldel=np.array([])
    P_deldel_N_vv=np.array([])
    P_delv_N_delv_1=np.array([])
    P_delv_N_delv_2=np.array([])
    N_deldel_N_vv=np.array([])
    N_delv_N_delv=np.array([])
    const=1./(cc.c_light_Mpc_s**2*8*np.pi**2)
    for i in ell:
        k_perp=i/chi_z1
        k=np.sqrt(kpar**2+k_perp**2)
        zeta=(kpar/k*u+k_perp/k*np.sqrt(1-u**2))
        k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
        kperp_dot_kp_perp=k_perp*kp_perp
        K=np.sqrt(k**2+kp**2-2*k_dot_kp)
        K_perp=np.sqrt(k_perp**2+kp_perp**2-2*kperp_dot_kp_perp)
        Kperp_arr=np.append(Kperp_arr,K_perp)
        #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
        theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
        #print (theta_K.min())
        mu_k_sq=kpar**2/k**2
        rsd_1=1.+f(z1)*kpar**2/k**2
        rsd_2=1.+f(z2)*kpar**2/k**2
        Noise_Kperp=Noise(K_perp*chi(z1))
        #print (kp_perp.min())
        max=1.e-1
        redshift_PP=T_mean(z1)**2*T_mean(z2)**2*f(z1)*f(z2)*rsd_1**2*rsd_2**2*H(z1)*H(z2)*D(z1)**2*D(z2)**2/(chi(z1)**2*r(z1)*(1+z1)*(1+z2))
        redshift_PN=T_mean(z1)*T_mean(z2)*rsd_1*rsd_2*f(z1)*f(z2)*H(z1)*H(z2)*D(z1)*D(z2)/((1+z1)*(1+z2))
        redshift_NN=f(z1)*f(z2)*H(z1)*H(z2)*chi(z1)**2*r(z1)/((1+z1)*(1+z2))

        integrand_P_deldel_P_vv=redshift_PP*theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)
        integrand_P_delv_P_delv=redshift_PP*theta_kp*theta_K*kp**2*Mps_interpf(kp)*Mps_interpf(K)/K/kp
        integrand_P_vv_N_deldel=redshift_PN*theta_kp**2*np.ma.masked_greater(Noise_Kperp,max)*Mps_interpf(kp)
        integrand_P_deldel_N_vv=redshift_PN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)
        integrand_P_delv_N_delv_1=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)/K/kp
        integrand_P_delv_N_delv_2=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_Kperp,max) *Mps_interpf(kp)/K/kp
        integrand_N_deldel_N_vv=redshift_NN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)
        integrand_N_delv_N_delv=redshift_NN*theta_kp*theta_K *kp**2/K/kp *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)

        integral_P_deldel_P_vv=const*trapz(trapz(trapz(trapz(integrand_P_deldel_P_vv,U,axis=0),Kp,axis=0),Z,axis=0),Z,axis=0)
        integral_P_delv_P_delv=const*trapz(trapz(trapz(trapz(integrand_P_delv_P_delv,U,axis=0),Kp,axis=0),Z,axis=0),Z,axis=0)
        integral_P_vv_N_deldel=const*trapz(trapz(trapz(trapz(integrand_P_vv_N_deldel,U,axis=0),Kp,axis=0),Z,axis=0),Z,axis=0)
        integral_P_deldel_N_vv=const*trapz(trapz(trapz(trapz(integrand_P_deldel_N_vv,U,axis=0),Kp,axis=0),Z,axis=0),Z,axis=0)
        integral_P_delv_N_delv_1=const*trapz(trapz(trapz(trapz(integrand_P_delv_N_delv_1,U,axis=0),Kp,axis=0),Z,axis=0),Z,axis=0)
        integral_P_delv_N_delv_2=const*trapz(trapz(trapz(trapz(integrand_P_delv_N_delv_2,U,axis=0),Kp,axis=0),Z,axis=0),Z,axis=0)
        integral_N_deldel_N_vv=const*trapz(trapz(trapz(trapz(integrand_N_deldel_N_vv,U,axis=0),Kp,axis=0),Z,axis=0),Z,axis=0)
        integral_N_delv_N_delv=const*trapz(trapz(trapz(trapz(integrand_N_delv_N_delv,U,axis=0),Kp,axis=0),Z,axis=0),Z,axis=0)

        P_deldel_P_vv=np.append(P_deldel_P_vv,integral_P_deldel_P_vv)
        P_delv_P_delv=np.append(P_delv_P_delv, integral_P_delv_P_delv)
        P_vv_N_deldel=np.append(P_vv_N_deldel, integral_P_vv_N_deldel)
        P_deldel_N_vv=np.append(P_deldel_N_vv, integral_P_deldel_N_vv)
        P_delv_N_delv_1=np.append(P_delv_N_delv_1, integral_P_delv_N_delv_1)
        P_delv_N_delv_2=np.append(P_delv_N_delv_2, integral_P_delv_N_delv_2)
        N_deldel_N_vv=np.append(N_deldel_N_vv, integral_N_deldel_N_vv)
        N_delv_N_delv=np.append(N_delv_N_delv, integral_N_delv_N_delv)

    return P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, P_deldel_N_vv, P_delv_N_delv_1, P_delv_N_delv_2, N_deldel_N_vv, N_delv_N_delv, kp_perp, Kperp_arr

def PN_integrals_no_redshift_int(ell,z_i,y , delta_z, Noise):
    Kp=np.geomspace(1.e-6,1.,n)
    U=np.linspace(-.9999,.9999,n)
    #Z_min=z_i-delta_z
    #Z_max=z_i+delta_z
    #z1=np.geomspace(Z_min,Z_max,n)
    z1=z_i
    z2=z1
    u,kp=np.meshgrid(U,Kp)

    T_mean=uf.T_mean
    chi_z1=chi(z1)
    f=uf.f
    D=uf.D_1
    r=uf.r
    H=uf.H

    kpar=y/r(z1)

    kp_perp=kp*np.sqrt(1-u**2)
    Noise_kp_perp=Noise(kp_perp*chi(z1))
    '''
    for i in range(len(Noise_kp_perp)):
        if Noise_kp_perp[i]>1.e4:
            Noise_kp_perp[i]=0.
    '''
    #print (Noise_kp_perp.min(),Noise_kp_perp.max(),'Noise of kp perp')
    mock_arr=np.geomspace(2.e-2,6.e-1,n)
    #print (Noise(mock_arr*chi(z1)).min(),Noise(mock_arr*chi(z1)).max(),'Noise of mock arr')
    kp_par=kp*u
    theta_kp=u

    const=1./(cc.c_light_Mpc_s**2*8*np.pi**2)
    k_perp=ell/chi_z1
    k=np.sqrt(kpar**2+k_perp**2)
    zeta=(kpar/k*u+k_perp/k*np.sqrt(np.abs(1-u**2)))
    k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
    kperp_dot_kp_perp=k_perp*kp_perp
    K=np.sqrt(np.abs(k**2+kp**2-2*k_dot_kp))
    K_perp=np.sqrt(np.abs(k_perp**2+kp_perp**2-2*kperp_dot_kp_perp))
    #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
    theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(np.abs(1-zeta**2))/k/K
    #print (theta_K.min())
    mu_k_sq=kpar**2/k**2
    rsd_1=1.+f(z1)*kpar**2/k**2
    rsd_2=1.+f(z2)*kpar**2/k**2
    Noise_Kperp=Noise(K_perp*chi(z1))
    #print (kp_perp.min())
    max=1.e-1
    redshift_PP=T_mean(z1)**2*T_mean(z2)**2*f(z1)*f(z2)*rsd_1**2*rsd_2**2*H(z1)*H(z2)*D(z1)**2*D(z2)**2/(chi(z1)**2*r(z1)*(1+z1)*(1+z2))
    redshift_PN=T_mean(z1)*T_mean(z2)*rsd_1*rsd_2*f(z1)*f(z2)*H(z1)*H(z2)*D(z1)*D(z2)/((1+z1)*(1+z2))
    redshift_NN=f(z1)*f(z2)*H(z1)*H(z2)*chi(z1)**2*r(z1)/((1+z1)*(1+z2))

    integrand_P_deldel_P_vv=redshift_PP*theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)
    integrand_P_delv_P_delv=redshift_PP*theta_kp*theta_K*kp**2*Mps_interpf(kp)*Mps_interpf(K)/K/kp
    integrand_P_vv_N_deldel=redshift_PN*theta_kp**2*np.ma.masked_greater(Noise_Kperp,max)*Mps_interpf(kp)
    integrand_P_deldel_N_vv=redshift_PN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)
    integrand_P_delv_N_delv_1=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)/K/kp
    integrand_P_delv_N_delv_2=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_Kperp,max) *Mps_interpf(kp)/K/kp
    integrand_N_deldel_N_vv=redshift_NN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)
    integrand_N_delv_N_delv=redshift_NN*theta_kp*theta_K *kp**2/K/kp *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)

    integral_P_deldel_P_vv=const*trapz(trapz(integrand_P_deldel_P_vv,U,axis=0),Kp,axis=0)
    integral_P_delv_P_delv=const*trapz(trapz(integrand_P_delv_P_delv,U,axis=0),Kp,axis=0)
    integral_P_vv_N_deldel=const*trapz(trapz(integrand_P_vv_N_deldel,U,axis=0),Kp,axis=0)
    integral_P_vv_N_deldel=np.ma.filled(integral_P_vv_N_deldel,1.e-100)
    integral_P_deldel_N_vv=const*trapz(trapz(integrand_P_deldel_N_vv,U,axis=0),Kp,axis=0)
    integral_P_deldel_N_vv=np.ma.filled(integral_P_deldel_N_vv,1.e-100)
    integral_P_delv_N_delv_1=const*trapz(trapz(integrand_P_delv_N_delv_1,U,axis=0),Kp,axis=0)
    integral_P_delv_N_delv_1=np.ma.filled(integral_P_delv_N_delv_1,1.e-100)
    integral_P_delv_N_delv_2=const*trapz(trapz(integrand_P_delv_N_delv_2,U,axis=0),Kp,axis=0)
    integral_P_delv_N_delv_2=np.ma.filled(integral_P_delv_N_delv_2,1.e-100)
    integral_N_deldel_N_vv=const*trapz(trapz(integrand_N_deldel_N_vv,U,axis=0),Kp,axis=0)
    integral_N_deldel_N_vv=np.ma.filled(integral_N_deldel_N_vv,1.e-100)
    integral_N_delv_N_delv=const*trapz(trapz(integrand_N_delv_N_delv,U,axis=0),Kp,axis=0)
    integral_N_delv_N_delv=np.ma.filled(integral_N_delv_N_delv,1.e-100)

    P_integral_sum=integral_P_deldel_P_vv+integral_P_delv_P_delv+integral_P_vv_N_deldel+integral_P_deldel_N_vv+integral_P_delv_N_delv_1+integral_P_delv_N_delv_2+integral_N_deldel_N_vv+integral_N_delv_N_delv
    #P_integral_sum=integral_P_delv_N_delv_1
    return P_integral_sum, K_perp

#print (PN_integrals_no_redshift_int(ell[9],1.26,y[9], Hirax_noise_z_1pt26_deltaz_pt0015)[0])
#print (PN_integrals_no_redshift_int(ell[9],1.26,y[9], Hirax_noise_z_1pt26_deltaz_pt0015)[1].min(), PN_integrals_no_redshift_int(ell[9],1.26,y[9], Hirax_noise_z_1pt26_deltaz_pt0015)[1].max())
def PN_integrals_squeezedlim_no_redshift_int(ell,z_i,y , delta_z, Noise):
    Kp=np.geomspace(1.e-6,.1,n)
    U=np.linspace(-.9999,.9999,n)
    #Z_min=z_i-delta_z
    #Z_max=z_i+delta_z
    #z1=np.geomspace(Z_min,Z_max,n)
    z1=z_i
    z2=z1
    u,kp=np.meshgrid(U,Kp)

    T_mean=uf.T_mean
    chi_z1=chi(z1)
    f=uf.f
    D=uf.D_1
    r=uf.r
    H=uf.H

    kpar=y/r(z1)

    kp_perp=kp*np.sqrt(1-u**2)
    Noise_kp_perp=Noise(kp_perp*chi(z1))
    '''
    for i in range(len(Noise_kp_perp)):
        if Noise_kp_perp[i]>1.e4:
            Noise_kp_perp[i]=0.
    '''
    #print (Noise_kp_perp.min(),Noise_kp_perp.max(),'Noise of kp perp')
    mock_arr=np.geomspace(2.e-2,6.e-1,n)
    #print (Noise(mock_arr*chi(z1)).min(),Noise(mock_arr*chi(z1)).max(),'Noise of mock arr')
    kp_par=kp*u
    theta_kp=u

    const=1./(cc.c_light_Mpc_s**2*8*np.pi**2)
    k_perp=ell/chi_z1
    k=np.sqrt(kpar**2+k_perp**2)
    zeta=(kpar/k*u+k_perp/k*np.sqrt(np.abs(1-u**2)))
    k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
    kperp_dot_kp_perp=k_perp*kp_perp
    K=k
    K_perp=k_perp
    #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
    theta_K=kpar/k
    #print (theta_K.min())
    mu_k_sq=kpar**2/k**2
    rsd_1=1.+f(z1)*kpar**2/k**2
    rsd_2=1.+f(z2)*kpar**2/k**2
    Noise_Kperp=Noise(K_perp*chi(z1))
    #print (kp_perp.min())
    max=1.e-1
    redshift_PP=T_mean(z1)**2*T_mean(z2)**2*f(z1)*f(z2)*rsd_1**2*rsd_2**2*H(z1)*H(z2)*D(z1)**2*D(z2)**2/(chi(z1)**2*r(z1)*(1+z1)*(1+z2))
    redshift_PN=T_mean(z1)*T_mean(z2)*rsd_1*rsd_2*f(z1)*f(z2)*H(z1)*H(z2)*D(z1)*D(z2)/((1+z1)*(1+z2))
    redshift_NN=f(z1)*f(z2)*H(z1)*H(z2)*chi(z1)**2*r(z1)/((1+z1)*(1+z2))

    integrand_P_deldel_P_vv=redshift_PP*theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)
    integrand_P_delv_P_delv=redshift_PP*theta_kp*theta_K*kp**2*Mps_interpf(kp)*Mps_interpf(K)/K/kp
    integrand_P_vv_N_deldel=redshift_PN*theta_kp**2*np.ma.masked_greater(Noise_Kperp,max)*Mps_interpf(kp)
    integrand_P_deldel_N_vv=redshift_PN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)
    integrand_P_delv_N_delv_1=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)/K/kp
    integrand_P_delv_N_delv_2=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_Kperp,max) *Mps_interpf(kp)/K/kp
    integrand_N_deldel_N_vv=redshift_NN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)
    integrand_N_delv_N_delv=redshift_NN*theta_kp*theta_K *kp**2/K/kp *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)

    integral_P_deldel_P_vv=const*trapz(trapz(integrand_P_deldel_P_vv,U,axis=0),Kp,axis=0)
    integral_P_delv_P_delv=const*trapz(trapz(integrand_P_delv_P_delv,U,axis=0),Kp,axis=0)
    integral_P_vv_N_deldel=const*trapz(trapz(integrand_P_vv_N_deldel,U,axis=0),Kp,axis=0)
    integral_P_vv_N_deldel=np.ma.filled(integral_P_vv_N_deldel,1.e-100)
    integral_P_deldel_N_vv=const*trapz(trapz(integrand_P_deldel_N_vv,U,axis=0),Kp,axis=0)
    integral_P_deldel_N_vv=np.ma.filled(integral_P_deldel_N_vv,1.e-100)
    integral_P_delv_N_delv_1=const*trapz(trapz(integrand_P_delv_N_delv_1,U,axis=0),Kp,axis=0)
    integral_P_delv_N_delv_1=np.ma.filled(integral_P_delv_N_delv_1,1.e-100)
    integral_P_delv_N_delv_2=const*trapz(trapz(integrand_P_delv_N_delv_2,U,axis=0),Kp,axis=0)
    integral_P_delv_N_delv_2=np.ma.filled(integral_P_delv_N_delv_2,1.e-100)
    integral_N_deldel_N_vv=const*trapz(trapz(integrand_N_deldel_N_vv,U,axis=0),Kp,axis=0)
    integral_N_deldel_N_vv=np.ma.filled(integral_N_deldel_N_vv,1.e-100)
    integral_N_delv_N_delv=const*trapz(trapz(integrand_N_delv_N_delv,U,axis=0),Kp,axis=0)
    integral_N_delv_N_delv=np.ma.filled(integral_N_delv_N_delv,1.e-100)

    P_integral_sum=integral_P_deldel_P_vv+integral_P_delv_P_delv+integral_P_vv_N_deldel+integral_P_deldel_N_vv+integral_P_delv_N_delv_1+integral_P_delv_N_delv_2+integral_N_deldel_N_vv+integral_N_delv_N_delv
    #P_integral_sum=integral_P_delv_N_delv_1
    return P_integral_sum, K_perp

def var_PN_terms_squeezedlim_all_ell_and_y(ell,z_i,y, delta_z, Noise):
    array_var=np.zeros((n,n))
    for i in range(len(y)):
        for j in range(len(ell)):
            array_var[i,j]=PN_integrals_squeezedlim_no_redshift_int(ell[j],z_i,y[i], delta_z,Noise)[0]
    return array_var

'''
def var_PN_terms_all_ell_and_y(ell,z_i,y, Noise):
    array_var=np.zeros((n,n))
    for i in range(len(y)):
        for j in range(len(ell)):
            array_var[i,j]=PN_integrals_no_redshift_int(ell[j],z_i,y[i], Noise)[0]
    return array_var

def var_PN_terms_all_ell(ell,z_i, y, Noise):
    array_var=np.array([])
    for i in ell:
        array_var=np.append(array_var,PN_integrals_no_redshift_int(i,z_i,y, Noise))
    return array_var

#print (ell.max())
#print (y.max())
#print (y)
'''
#plt.plot(ell, var_PN_terms_all_ell(ell,1.26, 100., Hirax_noise_z_1pt26_deltaz_pt0015))
#plt.show()

'''
pylab.pcolormesh(kperp_arr,kpar_arr,np.abs(Z)) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
pylab.xlim([np.min(kperp_arr),np.max(kperp_arr)]) ; pylab.ylim([np.min(kpar_arr),np.max(kpar_arr)])
plt.xlabel(r'$k_\perp$',fontsize=12); plt.ylabel(r'$k_\parallel$',fontsize=12); plt.title(r'$C_{l,y}^{{p21,tot}}(z=1.26)[\mu K^4]}$')
plt.xscale('log')
plt.yscale('log')
plt.show()
'''

def I_kp_perp_kperp(kperp,kp_perp,kpar,kp_par,z1,Noise):
    u=np.linspace(-.9999,.9999,n)
    integrand_arr=np.zeros((n,n))
    theta_kp=u
    chi_z1=chi(z1)
    for i in range(n):
        for j in range(n):
            k=np.sqrt(kperp[i]**2+kpar**2)
            kp=np.sqrt(kp_perp[j]**2+kp_par**2)
            zeta=(kpar/k*u+kperp[i]/k*np.sqrt(1-u**2))
            k_dot_kp=kperp[i]*kp_perp[j]+kpar*kp_par
            kperp_dot_kp_perp=kperp[i]*kp_perp[j]

            if kperp[i]==kp_perp[j]:
                K=1.e-6
                K_perp=1.e-6
                theta_K=1.e-6
            else:
                K=np.sqrt(np.abs(k**2+kp**2-2*k_dot_kp))
                K_perp=np.sqrt(np.abs(kperp[i]**2+kp_perp[j]**2-2*kperp_dot_kp_perp))
                theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*kperp[i]*np.sqrt(np.abs(1.-zeta**2))/k/K

            Noise_kp_perp=Noise(kp_perp[j]*chi(z1))
            Noise_Kperp=Noise(K_perp*chi(z1))
            Mps_arr=np.append(Mps_arr,Mps_interpf(K))
            Noise_Kperp_arr=np.append(Noise_Kperp_arr,Noise_Kperp)
            Noise_kp_perp_arr=np.append(Noise_kp_perp_arr,Noise_kp_perp)
            integrand_P_vv_N_deldel=theta_kp**2*Noise_Kperp*Mps_interpf(kp)
            integrand_P_deldel_N_vv=theta_kp**2 *Noise_kp_perp *Mps_interpf(K)
            integrand_P_delv_N_delv_1=theta_kp*theta_K *kp**2 *Noise_kp_perp *Mps_interpf(K)/K/kp
            integrand_P_delv_N_delv_2=theta_kp*theta_K *kp**2 *Noise_Kperp *Mps_interpf(kp)/K/kp
            integrand_P_delv_N_delv_1_arr=np.append(integrand_P_delv_N_delv_1_arr,integrand_P_delv_N_delv_1)
            #integrand_arr[i,j]=sp.integrate.trapz(integrand_P_deldel_N_vv,u)
            integrand_arr[i,j]=trapz(integrand_P_delv_N_delv_2,u)
    return integrand_arr, integrand_P_delv_N_delv_1_arr, Noise_Kperp_arr, Noise_kp_perp_arr, Mps_arr, theta_K_arr

kperp=np.geomspace(1.e-4,10.,n)
kp_perp=np.geomspace(1.e-4,10,n)

#print (kperp)
#print (kp_perp)
#print (np.meshgrid(kperp,kp_perp))
Z=I_kp_perp_kperp(kperp,kp_perp,1.e-1,1.e-1,1.26,Hirax_noise_z_1pt26_deltaz_pt0015)

print (np.abs(Z[0]).min(), np.abs(Z[0]).max())
print (Z[2])
#print (np.abs(Z[1]).min(),'integral')
#print (Z[1],'integrand')
#print (Z[-2].min(), 'Kperp_arr')
#print (Z[0].min())

pylab.pcolormesh(kp_perp,kperp, np.abs(Z[0]), vmin=np.abs(Z[0]).min(),vmax=1.e5, norm=LogNorm()); cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=12, which='major')
plt.tick_params(axis='both', which='major', labelsize=12)
pylab.xlim([np.min(kp_perp),np.max(kp_perp)]) ; pylab.ylim([np.min(kperp),np.max(kperp)])
plt.xlabel(r'$k_{p_\perp}$',fontsize=12); plt.ylabel(r'$k_\perp$',fontsize=12);
plt.title(r'$P(K) N(kp)$',fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.show()


'''
fig, ax = plt.subplots(2, 1)

Z=I_kp_perp_kperp(kperp,kp_perp,1.e-1,1.e-1,1.26,Hirax_noise_z_1pt26_deltaz_pt0015)

pcm = ax[0].pcolor(kperp, kp_perp, Z,
                   norm=plt.colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                   cmap='PuBu_r')
fig.colorbar(pcm, ax=ax[0], extend='max')

pcm = ax[1].pcolor(kperp, kp_perp, Z, cmap='PuBu_r')
fig.colorbar(pcm, ax=ax[1], extend='max')
plt.show()
'''

def PN_integrals_integral_over_y_no_z_int(ell,z1,delta_z, Noise):
    Kp=np.geomspace(1.e-10,1.,n)
    U=np.linspace(-.9999,.9999,n)
    Kpar=np.geomspace(1.e-6,.15,n)
    Z_min=z1-delta_z
    Z_max=z1+delta_z
    z2=np.geomspace(Z_min,Z_max,n)
    z2=z1
    u,kp,kpar=np.meshgrid(U,Kp,Kpar)

    T_mean=uf.T_mean
    chi_z1=chi(z1)
    f=uf.f
    D=uf.D_1
    r=uf.r
    H=uf.H

    Kperp_arr=np.array([])
    K_arr=np.array([])
    kp_perp=kp*np.sqrt(1-u**2)
    Noise_kp_perp=Noise(kp_perp*chi(z1))
    '''
    for i in range(len(Noise_kp_perp)):
        if Noise_kp_perp[i]>1.e4:
            Noise_kp_perp[i]=0.
    '''
    #print (Noise_kp_perp.min(),Noise_kp_perp.max(),'Noise of kp perp')
    mock_arr=np.geomspace(2.e-2,6.e-1,n)
    #print (Noise(mock_arr*chi(z1)).min(),Noise(mock_arr*chi(z1)).max(),'Noise of mock arr')
    kp_par=kp*u
    theta_kp=u

    P_deldel_P_vv=np.array([])
    P_delv_P_delv=np.array([])
    P_vv_N_deldel=np.array([])
    P_deldel_N_vv=np.array([])
    P_delv_N_delv_1=np.array([])
    P_delv_N_delv_2=np.array([])
    N_deldel_N_vv=np.array([])
    N_delv_N_delv=np.array([])
    const=1./(cc.c_light_Mpc_s**2*16*np.pi**3)
    for i in ell:
        k_perp=i/chi_z1
        k=np.sqrt(kpar**2+k_perp**2)
        zeta=(kpar/k*u+k_perp/k*np.sqrt(1-u**2))
        k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
        kperp_dot_kp_perp=k_perp*kp_perp
        K=np.sqrt(k**2+kp**2-2*k_dot_kp)
        K_arr=np.append(K_arr,K)
        K_perp=np.sqrt(k_perp**2+kp_perp**2-2*kperp_dot_kp_perp)
        Kperp_arr=np.append(Kperp_arr,K_perp)
        #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
        theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
        #print (theta_K.min())
        mu_k_sq=kpar**2/k**2
        rsd_1=1.+f(z1)*kpar**2/k**2
        rsd_2=1.+f(z2)*kpar**2/k**2
        Noise_Kperp=Noise(K_perp*chi(z1))
        #print (kp_perp.min())
        max=1.e-1
        redshift_PP=T_mean(z1)**2*f(z1)*rsd_1**2*H(z1)*D(z1)**2/(chi(z1)**2*(1+z1))*T_mean(z2)**2*f(z2)*rsd_2**2*H(z2)*D(z2)**2/(1+z2)
        redshift_PN=T_mean(z1)*rsd_1*f(z1)*H(z1)*D(z1)*r(z1)/(1+z1)*T_mean(z2)*rsd_2*f(z2)*H(z2)*D(z2)/(1+z2)
        redshift_NN=f(z1)*H(z1)*chi(z1)**2*r(z1)**2/(1+z1)*f(z2)*H(z2)/(1+z2)

        integrand_P_deldel_P_vv=redshift_PP*theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)
        integrand_P_delv_P_delv=redshift_PP*theta_kp*theta_K*kp**2*Mps_interpf(kp)*Mps_interpf(K)/K/kp
        integrand_P_vv_N_deldel=redshift_PN*theta_kp**2*np.ma.masked_greater(Noise_Kperp,max)*Mps_interpf(kp)
        integrand_P_deldel_N_vv=redshift_PN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)
        integrand_P_delv_N_delv_1=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)/K/kp
        integrand_P_delv_N_delv_2=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_Kperp,max) *Mps_interpf(kp)/K/kp
        integrand_N_deldel_N_vv=redshift_NN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)
        integrand_N_delv_N_delv=redshift_NN*theta_kp*theta_K *kp**2/K/kp *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)

        integral_P_deldel_P_vv=const*trapz(trapz(trapz(integrand_P_deldel_P_vv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_delv_P_delv=const*trapz(trapz(trapz(integrand_P_delv_P_delv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_vv_N_deldel=const*trapz(trapz(trapz(integrand_P_vv_N_deldel,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_vv_N_deldel=np.ma.filled(integral_P_vv_N_deldel,1.e-100)
        integral_P_deldel_N_vv=const*trapz(trapz(trapz(integrand_P_deldel_N_vv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_deldel_N_vv=np.ma.filled(integral_P_deldel_N_vv,1.e-100)
        integral_P_delv_N_delv_1=const*trapz(trapz(trapz(integrand_P_delv_N_delv_1,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_delv_N_delv_1=np.ma.filled(integral_P_delv_N_delv_1,1.e-100)
        integral_P_delv_N_delv_2=const*trapz(trapz(trapz(integrand_P_delv_N_delv_2,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_delv_N_delv_2=np.ma.filled(integral_P_delv_N_delv_2,1.e-100)
        integral_N_deldel_N_vv=const*trapz(trapz(trapz(integrand_N_deldel_N_vv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_N_deldel_N_vv=np.ma.filled(integral_N_deldel_N_vv,1.e-100)
        integral_N_delv_N_delv=const*trapz(trapz(trapz(integrand_N_delv_N_delv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_N_delv_N_delv=np.ma.filled(integral_N_delv_N_delv,1.e-100)

        P_deldel_P_vv=np.append(P_deldel_P_vv,integral_P_deldel_P_vv)
        P_delv_P_delv=np.append(P_delv_P_delv, integral_P_delv_P_delv)
        P_vv_N_deldel=np.append(P_vv_N_deldel, integral_P_vv_N_deldel)
        P_deldel_N_vv=np.append(P_deldel_N_vv, integral_P_deldel_N_vv)
        P_delv_N_delv_1=np.append(P_delv_N_delv_1, integral_P_delv_N_delv_1)
        P_delv_N_delv_2=np.append(P_delv_N_delv_2, integral_P_delv_N_delv_2)
        N_deldel_N_vv=np.append(N_deldel_N_vv, integral_N_deldel_N_vv)
        N_delv_N_delv=np.append(N_delv_N_delv, integral_N_delv_N_delv)

    return P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, P_deldel_N_vv, P_delv_N_delv_1, P_delv_N_delv_2, N_deldel_N_vv, N_delv_N_delv, kp_perp, Kperp_arr, K_arr

kperp_arr=ell/uf.chi(1.26)

P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, integral_P_deldel_N_vv, integral_P_delv_N_delv_1, integral_P_delv_N_delv_2, integral_N_deldel_N_vv, integral_N_delv_N_delv,kp_perp_arr,Kperp_arr,K_arr= PN_integrals_integral_over_y_no_z_int(ell,1.26,.0015, Hirax_noise_z_1pt26_deltaz_pt0015)
'''
p21_variance=P_deldel_P_vv +P_delv_P_delv +P_vv_N_deldel +integral_P_deldel_N_vv +integral_P_delv_N_delv_1 +integral_P_delv_N_delv_2 +integral_N_deldel_N_vv +integral_N_delv_N_delv
plt.loglog(kperp_arr,p21_variance)
#plt.ylim(1.e-6,np.abs(P_vv_N_deldel).max())
plt.ylabel(r'$C_{l}^{{p21,tot}}(z=1.26)[\mu K^4]}$')
plt.xlabel('l')
plt.show()
'''
def PN_integrals_integral_over_y(ell,z1,delta_z, Noise):
    Kp=np.geomspace(1.e-10,1.,n)
    U=np.linspace(-.9999,.9999,n)
    Kpar=np.geomspace(1.e-6,.15,n)
    Z_min=z1-delta_z
    Z_max=z1+delta_z
    z2=np.geomspace(Z_min,Z_max,n)
    u,kp,kpar=np.meshgrid(U,Kp,Kpar)

    T_mean=uf.T_mean
    chi_z1=chi(z1)
    f=uf.f
    D=uf.D_1
    r=uf.r
    H=uf.H

    Kperp_arr=np.array([])
    K_arr=np.array([])
    kp_perp=kp*np.sqrt(1-u**2)
    Noise_kp_perp=Noise(kp_perp*chi(z1))
    '''
    for i in range(len(Noise_kp_perp)):
        if Noise_kp_perp[i]>1.e4:
            Noise_kp_perp[i]=0.
    '''
    #print (Noise_kp_perp.min(),Noise_kp_perp.max(),'Noise of kp perp')
    mock_arr=np.geomspace(2.e-2,6.e-1,n)
    #print (Noise(mock_arr*chi(z1)).min(),Noise(mock_arr*chi(z1)).max(),'Noise of mock arr')
    kp_par=kp*u
    theta_kp=u

    P_deldel_P_vv=np.array([])
    P_delv_P_delv=np.array([])
    P_vv_N_deldel=np.array([])
    P_deldel_N_vv=np.array([])
    P_delv_N_delv_1=np.array([])
    P_delv_N_delv_2=np.array([])
    N_deldel_N_vv=np.array([])
    N_delv_N_delv=np.array([])
    const=1./(cc.c_light_Mpc_s**2*16*np.pi**3)
    for i in ell:
        k_perp=i/chi_z1
        k=np.sqrt(kpar**2+k_perp**2)
        zeta=(kpar/k*u+k_perp/k*np.sqrt(1-u**2))
        k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
        kperp_dot_kp_perp=k_perp*kp_perp
        K=np.sqrt(k**2+kp**2-2*k_dot_kp)
        K_arr=np.append(K_arr,K)
        K_perp=np.sqrt(k_perp**2+kp_perp**2-2*kperp_dot_kp_perp)
        Kperp_arr=np.append(Kperp_arr,K_perp)
        #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
        theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
        #print (theta_K.min())
        mu_k_sq=kpar**2/k**2
        rsd_1=1.+f(z1)*kpar**2/k**2
        rsd_2=1.+f(z2)*kpar**2/k**2
        Noise_Kperp=Noise(K_perp*chi(z1))
        #print (kp_perp.min())
        max=1.e-1
        redshift_PP=T_mean(z1)**2*f(z1)*rsd_1**2*H(z1)*D(z1)**2/(chi(z1)**2*(1+z1))*trapz(T_mean(z2)**2*f(z2)*rsd_2**2*H(z2)*D(z2)**2/(1+z2),z2)
        redshift_PN=T_mean(z1)*rsd_1*f(z1)*H(z1)*D(z1)*r(z1)/(1+z1)*trapz(T_mean(z2)*rsd_2*f(z2)*H(z2)*D(z2)/(1+z2),z2)
        redshift_NN=f(z1)*H(z1)*chi(z1)**2*r(z1)**2/(1+z1)*trapz(f(z2)*H(z2)/(1+z2),z2)

        integrand_P_deldel_P_vv=redshift_PP*theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)
        integrand_P_delv_P_delv=redshift_PP*theta_kp*theta_K*kp**2*Mps_interpf(kp)*Mps_interpf(K)/K/kp
        integrand_P_vv_N_deldel=redshift_PN*theta_kp**2*np.ma.masked_greater(Noise_Kperp,max)*Mps_interpf(kp)
        integrand_P_deldel_N_vv=redshift_PN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)
        integrand_P_delv_N_delv_1=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)/K/kp
        integrand_P_delv_N_delv_2=redshift_PN*theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_Kperp,max) *Mps_interpf(kp)/K/kp
        integrand_N_deldel_N_vv=redshift_NN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)
        integrand_N_delv_N_delv=redshift_NN*theta_kp*theta_K *kp**2/K/kp *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)

        integral_P_deldel_P_vv=const*trapz(trapz(trapz(integrand_P_deldel_P_vv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_delv_P_delv=const*trapz(trapz(trapz(integrand_P_delv_P_delv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_vv_N_deldel=const*trapz(trapz(trapz(integrand_P_vv_N_deldel,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_vv_N_deldel=np.ma.filled(integral_P_vv_N_deldel,1.e-6)
        integral_P_deldel_N_vv=const*trapz(trapz(trapz(integrand_P_deldel_N_vv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_deldel_N_vv=np.ma.filled(integral_P_deldel_N_vv,1.e-6)
        integral_P_delv_N_delv_1=const*trapz(trapz(trapz(integrand_P_delv_N_delv_1,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_delv_N_delv_1=np.ma.filled(integral_P_delv_N_delv_1,1.e-6)
        integral_P_delv_N_delv_2=const*trapz(trapz(trapz(integrand_P_delv_N_delv_2,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_P_delv_N_delv_2=np.ma.filled(integral_P_delv_N_delv_2,1.e-6)
        integral_N_deldel_N_vv=const*trapz(trapz(trapz(integrand_N_deldel_N_vv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_N_deldel_N_vv=np.ma.filled(integral_N_deldel_N_vv,1.e-6)
        integral_N_delv_N_delv=const*trapz(trapz(trapz(integrand_N_delv_N_delv,U,axis=0),Kp,axis=0),Kpar,axis=0)
        integral_N_delv_N_delv=np.ma.filled(integral_N_delv_N_delv,1.e-6)

        P_deldel_P_vv=np.append(P_deldel_P_vv,integral_P_deldel_P_vv)
        P_delv_P_delv=np.append(P_delv_P_delv, integral_P_delv_P_delv)
        P_vv_N_deldel=np.append(P_vv_N_deldel, integral_P_vv_N_deldel)
        P_deldel_N_vv=np.append(P_deldel_N_vv, integral_P_deldel_N_vv)
        P_delv_N_delv_1=np.append(P_delv_N_delv_1, integral_P_delv_N_delv_1)
        P_delv_N_delv_2=np.append(P_delv_N_delv_2, integral_P_delv_N_delv_2)
        N_deldel_N_vv=np.append(N_deldel_N_vv, integral_N_deldel_N_vv)
        N_delv_N_delv=np.append(N_delv_N_delv, integral_N_delv_N_delv)

    return P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, P_deldel_N_vv, P_delv_N_delv_1, P_delv_N_delv_2, N_deldel_N_vv, N_delv_N_delv, kp_perp, Kperp_arr, K_arr



P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, integral_P_deldel_N_vv, integral_P_delv_N_delv_1, integral_P_delv_N_delv_2, integral_N_deldel_N_vv, integral_N_delv_N_delv,kp_perp_arr,Kperp_arr,K_arr= PN_integrals_integral_over_y(ell,1.26,.0015, Hirax_noise_z_1pt26_deltaz_pt0015)

p21_variance=P_deldel_P_vv +P_delv_P_delv +P_vv_N_deldel +integral_P_deldel_N_vv +integral_P_delv_N_delv_1 +integral_P_delv_N_delv_2 +integral_N_deldel_N_vv +integral_N_delv_N_delv

PN_only=P_vv_N_deldel +integral_P_deldel_N_vv +integral_P_delv_N_delv_1 +integral_P_delv_N_delv_2
#print (Kperp_arr.min(),'Kperp arr')
#print (K_arr.min(),'K arr')

'''
print (PN_only.min(),PN_only.max(),'PN only')

print (K_arr.max(),Kperp_arr.max(),kp_perp_arr.max())

print (P_deldel_P_vv.min(), P_deldel_P_vv.max())
print (P_delv_P_delv.min(), P_delv_P_delv.max())
print (P_vv_N_deldel.min(), P_vv_N_deldel.max())
print (integral_P_deldel_N_vv.min(), integral_P_deldel_N_vv.max())
print (integral_P_delv_N_delv_1.min(), integral_P_delv_N_delv_1.max())
print (integral_P_delv_N_delv_2.min(), integral_P_delv_N_delv_2.max())
print (integral_N_deldel_N_vv.min(), integral_N_deldel_N_vv.max())
print (integral_N_delv_N_delv.min(), integral_N_delv_N_delv.max())

#plt.plot(ell,P_vv_N_deldel,'r')
plt.plot(ell,integral_P_deldel_N_vv,'b')
plt.plot(ell,integral_P_delv_N_delv_1,'g')
plt.plot(ell,integral_P_delv_N_delv_2,'k')
plt.plot(ell,integral_N_deldel_N_vv,'m')
plt.plot(ell,integral_N_delv_N_delv,'--')
plt.show()
'''

#plt.plot(ell,p21_variance)
#plt.ylabel('var int over y and z2')
#plt.show()

def F_PN_integrals(ell,z1,y, Noise):
    Kp=np.geomspace(1.e-10,10.,n)
    U=np.linspace(-.9999,.9999,n)
    u,kp=np.meshgrid(U,Kp)

    chi_z1=chi(z1)

    Kperp_arr=np.array([])
    kpar=y/r(z1)
    kp_perp=kp*np.sqrt(1-u**2)
    Noise_kp_perp=Noise(kp_perp*chi(z1))

    #print (Noise_kp_perp.min(),Noise_kp_perp.max(),'Noise of kp perp')
    mock_arr=np.geomspace(2.e-2,6.e-1,n)
    #print (Noise(mock_arr*chi(z1)).min(),Noise(mock_arr*chi(z1)).max(),'Noise of mock arr')

    #print (np.argwhere(np.isnan(Noise(kp_perp*chi(z1)))))
    kp_par=kp*u
    theta_kp=u

    P_deldel_P_vv=np.array([])
    P_delv_P_delv=np.array([])
    P_vv_N_deldel=np.array([])
    P_deldel_N_vv=np.array([])
    P_delv_N_delv_1=np.array([])
    P_delv_N_delv_2=np.array([])
    N_deldel_N_vv=np.array([])
    N_delv_N_delv=np.array([])
    for i in ell:
        k_perp=i/chi_z1
        k=np.sqrt(kpar**2+k_perp**2)
        zeta=(kpar/k*u+k_perp/k*np.sqrt(1-u**2))
        k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
        kperp_dot_kp_perp=k_perp*kp_perp
        K=np.sqrt(k**2+kp**2-2*k_dot_kp)
        K_perp=np.sqrt(k_perp**2+kp_perp**2-2*kperp_dot_kp_perp)
        Kperp_arr=np.append(Kperp_arr,K_perp)
        #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
        theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
        #print (theta_K.min())
        mu_k_sq=kpar**2/k**2
        Noise_Kperp=Noise(K_perp*chi(z1))
        #print (kp_perp.min())
        max=5.

        integrand_P_deldel_P_vv=theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)
        #print (integrand_P_deldel_P_vv.min(), integrand_P_deldel_P_vv.max())
        integrand_P_delv_P_delv=theta_kp*theta_K*kp**2*Mps_interpf(kp)*Mps_interpf(K)/K/kp
        #print (integrand_P_delv_P_delv.min(), integrand_P_delv_P_delv.max())
        integrand_P_vv_N_deldel=theta_kp**2*np.ma.masked_greater(Noise_Kperp,max)*Mps_interpf(kp)
        integrand_P_deldel_N_vv=theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)
        integrand_P_delv_N_delv_1=theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)/K/kp
        integrand_P_delv_N_delv_2=theta_kp*theta_K *kp**2 *np.ma.masked_greater(Noise_Kperp,max) *Mps_interpf(kp)/K/kp
        integrand_N_deldel_N_vv=theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)
        integrand_N_delv_N_delv=theta_kp*theta_K *kp**2/K/kp *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)

        integral_P_deldel_P_vv=trapz(trapz(integrand_P_deldel_P_vv,U,axis=0),Kp,axis=0)
        integral_P_delv_P_delv=trapz(trapz(integrand_P_delv_P_delv,U,axis=0),Kp,axis=0)
        integral_P_vv_N_deldel=trapz(trapz(integrand_P_vv_N_deldel,U,axis=0),Kp,axis=0)
        integral_P_deldel_N_vv=trapz(trapz(integrand_P_deldel_N_vv,U,axis=0),Kp,axis=0)
        integral_P_delv_N_delv_1=trapz(trapz(integrand_P_delv_N_delv_1,U,axis=0),Kp,axis=0)
        integral_P_delv_N_delv_2=trapz(trapz(integrand_P_delv_N_delv_2,U,axis=0),Kp,axis=0)
        integral_N_deldel_N_vv=trapz(trapz(integrand_N_deldel_N_vv,U,axis=0),Kp,axis=0)
        integral_N_delv_N_delv=trapz(trapz(integrand_N_delv_N_delv,U,axis=0),Kp,axis=0)


        P_deldel_P_vv=np.append(P_deldel_P_vv,integral_P_deldel_P_vv)
        P_delv_P_delv=np.append(P_delv_P_delv, integral_P_delv_P_delv)
        P_vv_N_deldel=np.append(P_vv_N_deldel, integral_P_vv_N_deldel)
        P_deldel_N_vv=np.append(P_deldel_N_vv, integral_P_deldel_N_vv)
        P_delv_N_delv_1=np.append(P_delv_N_delv_1, integral_P_delv_N_delv_1)
        P_delv_N_delv_2=np.append(P_delv_N_delv_2, integral_P_delv_N_delv_2)
        N_deldel_N_vv=np.append(N_deldel_N_vv, integral_N_deldel_N_vv)
        N_delv_N_delv=np.append(N_delv_N_delv, integral_N_delv_N_delv)

    #print (K_perp.min(),K_perp.max(),'K perp limits')
    #print (Noise_Kperp.min(),Noise_Kperp.max(), 'Noise of Kperp')
    #print (np.argwhere(np.isnan(Noise(K_perp*chi(z1)))))

    return P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, P_deldel_N_vv, P_delv_N_delv_1, P_delv_N_delv_2, N_deldel_N_vv, N_delv_N_delv, kp_perp, Kperp_arr


'''
F_int=F_PN_integrals(ell,z,1., Hirax_noise_z_1pt26_deltaz_pt0015)
F_P_deldel_P_vv, F_P_delv_P_delv, F_P_vv_N_deldel, F_P_deldel_N_vv, F_P_delv_N_delv_1, F_P_delv_N_delv_2, F_N_deldel_N_vv, F_N_delv_N_delv,  kp_perp, K_perp = F_int

print (kp_perp.min()*uf.chi(1.26),kp_perp.max()*uf.chi(1.26), 'kp perp')
print ('%.2E' % Decimal(F_P_deldel_P_vv.min()), '%.2E' % Decimal(F_P_deldel_P_vv.max()), 'P_deldel_P_vv')
print ('%.4E' % Decimal(F_P_delv_P_delv.min()), '%.4E' % Decimal(F_P_delv_P_delv.max()), 'P_delv_P_delv')
print ('%.4E' % Decimal(F_P_vv_N_deldel.min()), '%.4E' % Decimal(F_P_vv_N_deldel.max()), 'P_vv_N_deldel')
print ('%.4E' % Decimal(F_P_deldel_N_vv.min()), '%.4E' % Decimal(F_P_deldel_N_vv.max()), 'P_deldel_N_vv')
print ('%.4E' % Decimal(F_P_delv_N_delv_1.min()), '%.4E' % Decimal(F_P_delv_N_delv_1.max()), 'P_delv_N_delv_1')
print ('%.4E' % Decimal(F_P_delv_N_delv_2.min()), '%.4E' % Decimal(F_P_delv_N_delv_2.max()), 'P_delv_N_delv_2')
print ('%.4E' % Decimal(F_N_deldel_N_vv.min()), '%.4E' % Decimal(F_N_deldel_N_vv.max()), 'N_deldel_N_vv')
print ('%.4E' % Decimal(F_N_delv_N_delv.min()), '%.4E' % Decimal(F_N_delv_N_delv.max()), 'N_delv_N_delv')
'''


'''
P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, integral_P_deldel_N_vv, integral_P_delv_N_delv_1, integral_P_delv_N_delv_2, integral_N_deldel_N_vv, integral_N_delv_N_delv,_,_,_ = PN_integrals(ell,z,1, .0015,Hirax_noise_z_1pt26_deltaz_pt0015)

p21_variance=P_deldel_P_vv +P_delv_P_delv +P_vv_N_deldel +integral_P_deldel_N_vv +integral_P_delv_N_delv_1 +integral_P_delv_N_delv_2 +integral_N_deldel_N_vv +integral_N_delv_N_delv

plt.plot(ell,p21_variance)
plt.show()

print (P_deldel_P_vv.min(), P_deldel_P_vv.max())
print (P_delv_P_delv.min(), P_delv_P_delv.max())
print (P_vv_N_deldel.min(), P_vv_N_deldel.max())
print (integral_P_deldel_N_vv.min(), integral_P_deldel_N_vv.max())
print (integral_P_delv_N_delv_1.min(), integral_P_delv_N_delv_1.max())
print (integral_P_delv_N_delv_2.min(), integral_P_delv_N_delv_2.max())
print (integral_N_deldel_N_vv.min(), integral_N_deldel_N_vv.max())
print (integral_N_delv_N_delv.min(), integral_N_delv_N_delv.max())
'''

'''
plt.loglog(ell,P_vv_N_deldel,'b')
plt.loglog(ell,integral_P_deldel_N_vv,'r')
plt.loglog(ell, integral_P_delv_N_delv_1,'g')
plt.loglog(ell, integral_P_delv_N_delv_2,'m')
plt.loglog(ell,integral_N_deldel_N_vv,'k')
plt.loglog(ell,integral_N_delv_N_delv,'--')
#plt.show()
'''



#print (N_delta_P_vv_proper(kperp_arr,z,1.e-1,Hirax_noise_z_1pt26_deltaz_pt05)[1].min(),N_delta_P_vv_proper(kperp_arr,z,1.e-1,Hirax_noise_z_1pt26_deltaz_pt05)[1].max(),'N_delta_P_v')

#plt.loglog(kperp_arr,Hirax_noise_z_1pt26_deltaz_pt05(ell))
#plt.loglog(kperp_arr,P_v_N_delta(z,kperp_arr,1.e-1,Hirax_noise_z_1pt26_deltaz_pt05))
#plt.loglog(kperp_arr,P_delta_N_v(z,kperp_arr,1.e-1,Hirax_noise_z_1pt26_deltaz_pt05))
#plt.ylim(1.e-4,1.e5)
#plt.xlim(1.e-2,1.e-1)
#plt.show()



'''
print (kp_perp.shape,'kp_perp')
print (Kperp_arr.shape,'Kperp')
print (P_deldel_N_vv)
print (P_vv_N_deldel)

plt.loglog(kperp_arr,P_vv_N_deldel,'b')
plt.loglog(kperp_arr,P_deldel_N_vv,'r')
plt.loglog(kperp_arr, P_delv_N_delv_1,'g')
plt.loglog(kperp_arr, P_delv_N_delv_2,'k')
plt.loglog(kperp_arr, N_deldel_N_vv,'m')
plt.loglog(kperp_arr, N_delv_N_delv,'--')

plt.xlabel('k perp')
#plt.show()
'''
