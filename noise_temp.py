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
z=1.
n=30
delta_z=0.2

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

ell_large=np.geomspace(1.e-6,1.e5,10000)

#np.savetxt('Hirax_noise_z_1_Ddish_6_Dsep_7_geom.out',(ell_large,HiraxNoise(ell_large,6.,7.,1.)))
#ell_new,Hirax_noise_z_1=np.genfromtxt('/home/zahra/python_scripts/kSZ_21cm_signal/Hirax_noise_z_1_Ddish_6_Dsep_7.out')
Hirax_noise_z_1pt26_deltaz_pt2 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.26,0.2), bounds_error=False)
Hirax_noise_z_1pt26_deltaz_pt05 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.26,0.05), bounds_error=False)
Hirax_noise_z_1pt26_deltaz_pt0015 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.26,0.0015), bounds_error=False)
Hirax_noise_z_1_deltaz_pt2 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.,0.2), bounds_error=False)

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
    Kp=np.geomspace(1.e-10,10.,n)
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
        max=5.
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

def I_kp_perp_kperp(kperp,kp_perp,z1,Noise): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    integrand_arr=np.zeros((n,n))
    Noise_Kperp_arr=np.array([])
    Mps_kp_arr=np.array([])
    Theta_K_arr=np.array([])
    zeta_arr=np.array([])
    kpar=1.
    kp_par=1.
    chi_z=chi(z1)
    for i in range(n):
        for j in range(n):
            k=np.sqrt(kperp[i]**2+kpar**2)
            kp=np.sqrt(kp_perp[j]**2+kp_par**2)
            cos_theta=kp_par/kp #cos_theta=u, theta is azimuthal angle between k' and z (line of sight) axis
            sin_theta=kp_perp[j]/kp
            sin_gamma=kpar/k #gamma is measured between k and xy plane, i.e. elevation of k
            cos_gamma=kperp[i]/k
            zeta=sin_gamma*cos_theta+cos_gamma*sin_theta
            zeta_arr=np.append(zeta_arr,zeta)
            k_dot_kp=kperp[i]*kp_perp[j]+kpar*kp_par*zeta
            kperp_dot_kp_perp=kperp[i]*kp_perp[j]
            if kperp[i]==kp_perp[j]:
                K=1.e-6
                K_perp=1.e-6
                theta_K=1.e-6
            else:
                K=np.sqrt(np.abs(k**2+kp**2-2*k_dot_kp))
                K_perp=np.sqrt(kperp[i]**2+kp_perp[j]**2-2*kperp_dot_kp_perp)
                theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*kperp[i]*np.sqrt(np.abs(1.-zeta**2))/k/K
            Mps_kp_arr=np.append(Mps_kp_arr,Mps_interpf(kp))
            Theta_K_arr=np.append(Theta_K_arr,theta_K)
            theta_kp=cos_theta
            Noise_kp_perp=Noise(kp_perp[j]*chi(z1))
            Noise_Kperp=Noise(K_perp*chi(z1))
            Noise_Kperp_arr=np.append(Noise_Kperp_arr,Noise_Kperp)
            I=theta_kp*(theta_kp/kp**2+theta_K/K/kp)
            integrand_P_vv_N_deldel=theta_kp**2*Noise_Kperp*Mps_interpf(kp)/kp**2
            integrand_P_deldel_N_vv=theta_kp**2 *Noise_kp_perp *Mps_interpf(K)/kp**2
            integrand_P_delv_N_delv_1=theta_kp*theta_K *Noise_kp_perp *Mps_interpf(K)/K/kp
            integrand_P_delv_N_delv_2=theta_kp*theta_K  *Noise_Kperp *Mps_interpf(kp)/K/kp
            integrand_N_deldel_N_vv=theta_kp**2 *Noise_kp_perp*Noise_Kperp/kp**2
            integrand_N_delv_N_delv=theta_kp*theta_K*Noise_kp_perp*Noise_Kperp/K/kp
            #integrand_arr[i,j]=sp.integrate.trapz(integrand_P_deldel_N_vv,u)
            integrand_arr[i,j]=integrand_N_delv_N_delv
    return integrand_arr, Noise_Kperp_arr, Mps_kp_arr, Theta_K_arr, zeta_arr
'''
def I_kp_perp_kperp(kperp,kp_perp,kpar,kp_par,z1,Noise):
    integrand_arr=np.zeros((n,n))
    chi_z1=chi(z1)
    Noise_Kperp_arr=np.array([])
    K_perp_arr=np.array([])
    for i in range(n):
        for j in range(n):
            k=np.sqrt(kperp[i]**2+kpar**2)
            kp=np.sqrt(kp_perp[j]**2+kp_par**2)
            cos_theta=kp_par/kp
            sin_theta=kp_perp[j]/kp
            sin_gamma=kpar/k #gamma is measured between k and xy plane, i.e. elevation of k
            cos_gamma=kperp[i]/k
            theta_kp=cos_theta
            zeta=sin_gamma*cos_theta+cos_gamma*sin_theta
            k_dot_kp=kperp[i]*kp_perp[j]+kpar*kp_par*zeta
            kperp_dot_kp_perp=kperp[i]*kp_perp[j]
            K=np.sqrt(k**2+kp**2-2*k_dot_kp)
            K_perp=np.sqrt(kperp[i]**2+kp_perp[j]**2-2*kperp_dot_kp_perp)
            K_perp_arr=np.append(K_perp_arr, K_perp)
            theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*kperp[i]*np.sqrt(1-zeta**2)/k/K
            Noise_kp_perp=Noise(kp_perp[j]*chi(z1))
            Noise_Kperp=Noise(K_perp*chi(z1))
            Noise_Kperp_arr=np.append(Noise_Kperp_arr,Noise_Kperp)
            integrand_P_vv_N_deldel=theta_kp**2*Noise_Kperp*Mps_interpf(kp)/kp**2
            integrand_P_deldel_N_vv=theta_kp**2 *Noise_kp_perp *Mps_interpf(K)/kp**2
            integrand_P_delv_N_delv_1=theta_kp*theta_K *Noise_kp_perp *Mps_interpf(K)/K/kp
            integrand_P_delv_N_delv_2=theta_kp*theta_K  *Noise_Kperp *Mps_interpf(kp)/K/kp
            #integrand_arr[i,j]=sp.integrate.trapz(integrand_P_deldel_N_vv,u)
            integrand_arr[i,j]=integrand_P_deldel_N_vv
    return integrand_arr, Noise_Kperp_arr, K_perp_arr
'''

n=50
kperp=np.geomspace(1.e-4,10.,n)
kp_perp=np.geomspace(1.e-4,10.,n)

#print (kperp)
#print (kp_perp)
#print (np.meshgrid(kperp,kp_perp))
Z=np.zeros((n,n))
Z_init=I_kp_perp_kperp(kperp,kp_perp,1.,Hirax_noise_z_1_deltaz_pt2)[0]


Z=np.where(Z_init>1.e10, 1., Z)

print (Z)
#np.save('N_kp_N_K_mask_30points_z_1pt26',Z[0])

#print (Z[0],'integral')
#print (Z[1],'Noise K_perp')
#print (Z[3],'theta K arr')
#print (Z[0].min())

pylab.pcolormesh(kp_perp,kperp, np.abs(Z_init),vmax=1.e5,cmap='Blues',norm=LogNorm()); cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=12, which='major')
plt.tick_params(axis='both', which='major', labelsize=12)
pylab.xlim([np.min(kp_perp),np.max(kperp)]) ; pylab.ylim([np.min(kp_perp),np.max(kperp)])
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

def PN_integrals_rect_coords_integral_over_y(ell,z1,delta_z, Noise): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    Kp_par=np.geomspace(1.e-6,.15,n)
    Kp_perp=np.geomspace(1.e-6,1.,n)
    Z_min=z1-delta_z
    Z_max=z1+delta_z
    Z=np.geomspace(Z_min,Z_max,n)
    Kpar=np.geomspace(1.e-6,.15,n)
    kp_perp,kp_par,kpar,z2=np.meshgrid(Kp_perp,Kp_par,Kpar,Z)
    T_mean=uf.T_mean
    chi_z1=chi(z1)
    f=uf.f
    D=uf.D_1
    r=uf.r
    H=uf.H
    Noise_kp_perp=Noise(kp_perp*chi(z1))

    P_deldel_P_vv=np.array([])
    P_delv_P_delv=np.array([])
    P_vv_N_deldel=np.array([])
    P_deldel_N_vv=np.array([])
    P_delv_N_delv_1=np.array([])
    P_delv_N_delv_2=np.array([])
    N_deldel_N_vv=np.array([])
    N_delv_N_delv=np.array([])
    const=1./(cc.c_light_Mpc_s**2*4*np.pi**2)
    for i in ell:
        kperp=i/chi_z1
        k=np.sqrt(kperp**2+kpar**2)
        kp=np.sqrt(kp_perp**2+kp_par**2)
        cos_theta=kp_par/kp #cos_theta=u, theta is azimuthal angle between k' and z (line of sight) axis
        sin_theta=kp_perp/kp
        sin_gamma=kpar/k #gamma is measured between k and xy plane, i.e. elevation of k
        cos_gamma=kperp/k
        zeta=sin_gamma*cos_theta+cos_gamma*sin_theta
        k_dot_kp=kperp*kp_perp+kpar*kp_par*zeta
        kperp_dot_kp_perp=kperp*kp_perp
        K=np.sqrt(k**2+kp**2-2*k_dot_kp)
        K_perp=np.sqrt(kperp**2+kp_perp**2-2*kperp_dot_kp_perp)
        theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*kperp*np.sqrt(1-zeta**2)/k/K
        theta_kp=cos_theta
        rsd_1=1.+f(z1)*kpar**2/k**2
        rsd_2=1.+f(z2)*kpar**2/k**2
        Noise_Kperp=Noise(K_perp*chi(z1))
        max=5.
        redshift_PP=T_mean(z1)**2*T_mean(z2)**2*f(z1)*f(z2)*rsd_1**2*rsd_2**2*H(z1)*H(z2)*D(z1)**2*D(z2)**2/(chi(z1)**2*(1+z1)*(1+z2))
        redshift_PN=T_mean(z1)*T_mean(z2)*rsd_1*rsd_2*f(z1)*f(z2)*H(z1)*H(z2)*D(z1)*D(z2)*r(z1)/((1+z1)*(1+z2))
        redshift_NN=f(z1)*f(z2)*H(z1)*H(z2)*chi(z1)**2*r(z1)**2/((1+z1)*(1+z2))
        integrand_P_deldel_P_vv=redshift_PP*theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)/kp**2
        integrand_P_delv_P_delv=redshift_PP*theta_kp*theta_K*Mps_interpf(kp)*Mps_interpf(K)/K/kp
        integrand_P_vv_N_deldel=redshift_PN*theta_kp**2*np.ma.masked_greater(Noise_Kperp,max)*Mps_interpf(kp)/kp**2
        integrand_P_deldel_N_vv=redshift_PN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max) *Mps_interpf(K)/kp**2
        integrand_P_delv_N_delv_1=redshift_PN*theta_kp*theta_K *np.ma.masked_greater(Noise_kp_perp,max)*Mps_interpf(K)/K/kp
        integrand_P_delv_N_delv_2=redshift_PN*theta_kp*theta_K *np.ma.masked_greater(Noise_Kperp,max) *Mps_interpf(kp)/K/kp
        integrand_N_deldel_N_vv=redshift_NN*theta_kp**2 *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)/kp**2
        integrand_N_delv_N_delv=redshift_NN*theta_kp*theta_K /K/kp *np.ma.masked_greater(Noise_kp_perp,max)*np.ma.masked_greater(Noise_Kperp,max)

        integral_P_deldel_P_vv=const*trapz(trapz(trapz(trapz(integrand_P_deldel_P_vv,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0),Z,axis=0)
        integral_P_delv_P_delv=const*trapz(trapz(trapz(trapz(integrand_P_delv_P_delv,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0),Z,axis=0)
        integral_P_vv_N_deldel=const*trapz(trapz(trapz(trapz(integrand_P_vv_N_deldel,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0),Z,axis=0)
        integral_P_deldel_N_vv=const*trapz(trapz(trapz(trapz(integrand_P_deldel_N_vv,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0),Z,axis=0)
        integral_P_delv_N_delv_1=const*trapz(trapz(trapz(trapz(integrand_P_delv_N_delv_1,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0),Z,axis=0)
        integral_P_delv_N_delv_2=const*trapz(trapz(trapz(trapz(integrand_P_delv_N_delv_2,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0),Z,axis=0)
        integral_N_deldel_N_vv=const*trapz(trapz(trapz(trapz(integrand_N_deldel_N_vv,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0),Z,axis=0)
        integral_N_delv_N_delv=const*trapz(trapz(trapz(trapz(integrand_N_delv_N_delv,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0),Z,axis=0)

        P_deldel_P_vv=np.append(P_deldel_P_vv,integral_P_deldel_P_vv)
        P_delv_P_delv=np.append(P_delv_P_delv, integral_P_delv_P_delv)
        P_vv_N_deldel=np.append(P_vv_N_deldel, integral_P_vv_N_deldel)
        P_deldel_N_vv=np.append(P_deldel_N_vv, integral_P_deldel_N_vv)
        P_delv_N_delv_1=np.append(P_delv_N_delv_1, integral_P_delv_N_delv_1)
        P_delv_N_delv_2=np.append(P_delv_N_delv_2, integral_P_delv_N_delv_2)
        N_deldel_N_vv=np.append(N_deldel_N_vv, integral_N_deldel_N_vv)
        N_delv_N_delv=np.append(N_delv_N_delv, integral_N_delv_N_delv)

    return P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, P_deldel_N_vv, P_delv_N_delv_1, P_delv_N_delv_2, N_deldel_N_vv, N_delv_N_delv

'''
P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, integral_P_deldel_N_vv, integral_P_delv_N_delv_1, integral_P_delv_N_delv_2, integral_N_deldel_N_vv, integral_N_delv_N_delv= PN_integrals_rect_coords_integral_over_y(ell,1.26, .0015, Hirax_noise_z_1pt26_deltaz_pt0015)

p21_variance=P_deldel_P_vv +P_delv_P_delv +P_vv_N_deldel +integral_P_deldel_N_vv +integral_P_delv_N_delv_1 +integral_P_delv_N_delv_2 +integral_N_deldel_N_vv +integral_N_delv_N_delv

plt.plot(ell,p21_variance)
plt.ylabel('var int over y and z2')
plt.show()




PN_only=P_vv_N_deldel +integral_P_deldel_N_vv +integral_P_delv_N_delv_1 +integral_P_delv_N_delv_2

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

def F_integral_rect_coords_integral(ell,z1,y, Noise): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    Kp_par=np.geomspace(1.e-6,.15,n)
    Kp_perp=np.geomspace(1.e-6,1.,n)

    chi_z1=chi(z1)
    Kperp=ell/chi_z1
    kperp,kp_perp,kp_par=np.meshgrid(Kperp,Kp_perp,Kp_par)

    Noise_kp_perp=Noise(kp_perp*chi(z1))
    PK_Nkp_mask=np.load('P_K_N_kp_mask_30points_z_1pt26.npy')
    PK_Nkp_mask[PK_Nkp_mask<1.e10]=1.
    PK_Nkp_mask[PK_Nkp_mask>1.e10]=0.
    PK_Nkp_mask=np.append(PK_Nkp_mask.flatten(),np.ones(n**3-n**2)).reshape(n,n,n)
    Pkp_NK_mask=np.load('P_kp_N_K_mask_30points_z_1pt26.npy')
    Pkp_NK_mask[Pkp_NK_mask<1.e10]=1.
    Pkp_NK_mask[Pkp_NK_mask>1.e10]=0.
    Pkp_NK_mask=np.append(Pkp_NK_mask.flatten(),np.ones(n**3-n**2)).reshape(n,n,n)

    NN_mask=np.load('N_kp_N_K_mask_30points_z_1pt26.npy')
    NN_mask[NN_mask<1.e10]=1.
    NN_mask[NN_mask>1.e10]=0.
    NN_mask=np.append(NN_mask.flatten(),np.ones(n**3-n**2)).reshape(n,n,n)

    kpar=y/r(z1)
    k=np.sqrt(kperp**2+kpar**2)
    kp=np.sqrt(kp_perp**2+kp_par**2)
    cos_theta=kp_par/kp #cos_theta=u, theta is azimuthal angle between k' and z (line of sight) axis
    sin_theta=kp_perp/kp
    sin_gamma=kpar/k #gamma is measured between k and xy plane, i.e. elevation of k
    cos_gamma=kperp/k
    zeta=sin_gamma*cos_theta+cos_gamma*sin_theta
    k_dot_kp=kperp*kp_perp+kpar*kp_par*zeta
    kperp_dot_kp_perp=kperp*kp_perp
    K=np.sqrt(k**2+kp**2-2*k_dot_kp)
    K_perp=np.sqrt(kperp**2+kp_perp**2-2*kperp_dot_kp_perp)
    theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*kperp*np.sqrt(1-zeta**2)/k/K
    theta_kp=cos_theta
    Noise_Kperp=Noise(K_perp*chi(z1))


    integrand_P_deldel_P_vv=theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)/kp**2
    integrand_P_delv_P_delv=theta_kp*theta_K*Mps_interpf(kp)*Mps_interpf(K)/K/kp
    integrand_P_vv_N_deldel=theta_kp**2*Noise_Kperp*Mps_interpf(kp)/kp**2*Pkp_NK_mask
    integrand_P_deldel_N_vv=theta_kp**2 *Noise_kp_perp *Mps_interpf(K)/kp**2*PK_Nkp_mask
    integrand_P_delv_N_delv_1=theta_kp*theta_K *Noise_kp_perp*Mps_interpf(K)/K/kp*PK_Nkp_mask
    integrand_P_delv_N_delv_2=theta_kp*theta_K *Noise_Kperp *Mps_interpf(kp)/K/kp*Pkp_NK_mask
    integrand_N_deldel_N_vv=theta_kp**2 *Noise_kp_perp*Noise_Kperp/kp**2*NN_mask
    integrand_N_delv_N_delv=theta_kp*theta_K /K/kp *Noise_kp_perp*Noise_Kperp*NN_mask

    integral_P_deldel_P_vv=trapz(trapz(integrand_P_deldel_P_vv,Kp_perp,axis=0),Kp_par,axis=0)
    integral_P_delv_P_delv=trapz(trapz(integrand_P_delv_P_delv,Kp_perp,axis=0),Kp_par,axis=0)
    integral_P_vv_N_deldel=trapz(trapz(integrand_P_vv_N_deldel,Kp_perp,axis=0),Kp_par,axis=0)
    integral_P_deldel_N_vv=trapz(trapz(integrand_P_deldel_N_vv,Kp_perp,axis=0),Kp_par,axis=0)
    integral_P_delv_N_delv_1=trapz(trapz(integrand_P_delv_N_delv_1,Kp_perp,axis=0),Kp_par,axis=0)
    integral_P_delv_N_delv_2=trapz(trapz(integrand_P_delv_N_delv_2,Kp_perp,axis=0),Kp_par,axis=0)
    integral_N_deldel_N_vv=trapz(trapz(integrand_N_deldel_N_vv,Kp_perp,axis=0),Kp_par,axis=0)
    integral_N_delv_N_delv=trapz(trapz(integrand_N_delv_N_delv,Kp_perp,axis=0),Kp_par,axis=0)

    return integral_P_deldel_P_vv, integral_P_delv_P_delv, integral_P_vv_N_deldel, integral_P_deldel_N_vv, integral_P_delv_N_delv_1, integral_P_delv_N_delv_2, integral_N_deldel_N_vv, integral_N_delv_N_delv

def F_integral_rect_coords_integral(ell,z1,z2,y, Noise): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    Kp_par=np.geomspace(1.e-6,.15,n)
    Kp_perp=np.geomspace(1.e-6,1.,n)

    kp_perp,kp_par=np.meshgrid(Kp_perp,Kp_par)
    T_mean=uf.T_mean
    chi_z1=chi(z1)
    f=uf.f
    D=uf.D_1
    r=uf.r
    H=uf.H
    Noise_kp_perp=Noise(kp_perp*chi(z1))
    PK_Nkp_mask=np.load('P_K_N_kp_mask_30points_z_1pt26.npy')
    PK_Nkp_mask[PK_Nkp_mask<1.e10]=1.
    PK_Nkp_mask[PK_Nkp_mask>1.e10]=0.
    Pkp_NK_mask=np.load('P_kp_N_K_mask_30points_z_1pt26.npy')
    Pkp_NK_mask[Pkp_NK_mask<1.e10]=1.
    Pkp_NK_mask[Pkp_NK_mask>1.e10]=0.
    NN_mask=np.load('N_kp_N_K_mask_30points_z_1pt26.npy')
    NN_mask[NN_mask<1.e10]=1.
    NN_mask[NN_mask>1.e10]=0.

    '''
    P_deldel_P_vv=np.array([])
    P_delv_P_delv=np.array([])
    P_vv_N_deldel=np.array([])
    P_deldel_N_vv=np.array([])
    P_delv_N_delv_1=np.array([])
    P_delv_N_delv_2=np.array([])
    N_deldel_N_vv=np.array([])
    N_delv_N_delv=np.array([])
    '''
    F_integral=np.zeros((n,n,n))
    for i in range(len(ell)):
        for j in range(len(y)):
            for rdsft in range(len(z2)):
                kperp=ell[i]/chi_z1
                kpar=y[j]/r(z1)
                k=np.sqrt(kperp**2+kpar**2)
                kp=np.sqrt(kp_perp**2+kp_par**2)
                cos_theta=kp_par/kp #cos_theta=u, theta is azimuthal angle between k' and z (line of sight) axis
                sin_theta=kp_perp/kp
                sin_gamma=kpar/k #gamma is measured between k and xy plane, i.e. elevation of k
                cos_gamma=kperp/k
                zeta=sin_gamma*cos_theta+cos_gamma*sin_theta
                k_dot_kp=kperp*kp_perp+kpar*kp_par*zeta
                kperp_dot_kp_perp=kperp*kp_perp
                K=np.sqrt(k**2+kp**2-2*k_dot_kp)
                K_perp=np.sqrt(kperp**2+kp_perp**2-2*kperp_dot_kp_perp)
                theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*kperp*np.sqrt(1-zeta**2)/k/K
                theta_kp=cos_theta
                Noise_Kperp=Noise(K_perp*chi(z1))
                rsd_1=1.+f(z1)*kpar**2/k**2
                rsd_2=1.+f(z2[rdsft])*kpar**2/k**2

                redshift_PP=T_mean(z1)**2*T_mean(z2[rdsft])**2*f(z1)*f(z2[rdsft])*rsd_1**2*rsd_2**2*H(z1)*H(z2[rdsft])*D(z1)**2*D(z2[rdsft])**2/(chi(z1)**2*(1+z1)*(1+z2[rdsft]))
                redshift_PN=T_mean(z1)*T_mean(z2[rdsft])*rsd_1*rsd_2*f(z1)*f(z2[rdsft])*H(z1)*H(z2[rdsft])*D(z1)*D(z2[rdsft])*r(z1)/((1+z1)*(1+z2[rdsft]))
                redshift_NN=f(z1)*f(z2[rdsft])*H(z1)*H(z2[rdsft])*chi(z1)**2*r(z1)**2/((1+z1)*(1+z2[rdsft]))

                integrand_P_deldel_P_vv=redshift_PP*theta_kp**2*Mps_interpf(kp)*Mps_interpf(K)/kp**2
                integrand_P_delv_P_delv=redshift_PP*theta_kp*theta_K*Mps_interpf(kp)*Mps_interpf(K)/K/kp
                integrand_P_vv_N_deldel=redshift_PN*theta_kp**2*Noise_Kperp*Mps_interpf(kp)/kp**2*Pkp_NK_mask
                integrand_P_deldel_N_vv=redshift_PN*theta_kp**2 *Noise_kp_perp *Mps_interpf(K)/kp**2*PK_Nkp_mask
                integrand_P_delv_N_delv_1=redshift_PN*theta_kp*theta_K *Noise_kp_perp*Mps_interpf(K)/K/kp*PK_Nkp_mask
                integrand_P_delv_N_delv_2=redshift_PN*theta_kp*theta_K *Noise_Kperp *Mps_interpf(kp)/K/kp*Pkp_NK_mask
                integrand_N_deldel_N_vv=redshift_NN*theta_kp**2 *Noise_kp_perp*Noise_Kperp/kp**2*NN_mask
                integrand_N_delv_N_delv=redshift_NN*theta_kp*theta_K /K/kp *Noise_kp_perp*Noise_Kperp*NN_mask

                integral_P_deldel_P_vv=trapz(trapz(integrand_P_deldel_P_vv,Kp_perp,axis=0),Kp_par,axis=0)
                integral_P_delv_P_delv=trapz(trapz(integrand_P_delv_P_delv,Kp_perp,axis=0),Kp_par,axis=0)
                integral_P_vv_N_deldel=trapz(trapz(integrand_P_vv_N_deldel,Kp_perp,axis=0),Kp_par,axis=0)
                integral_P_deldel_N_vv=trapz(trapz(integrand_P_deldel_N_vv,Kp_perp,axis=0),Kp_par,axis=0)
                integral_P_delv_N_delv_1=trapz(trapz(integrand_P_delv_N_delv_1,Kp_perp,axis=0),Kp_par,axis=0)
                integral_P_delv_N_delv_2=trapz(trapz(integrand_P_delv_N_delv_2,Kp_perp,axis=0),Kp_par,axis=0)
                integral_N_deldel_N_vv=trapz(trapz(integrand_N_deldel_N_vv,Kp_perp,axis=0),Kp_par,axis=0)
                integral_N_delv_N_delv=trapz(trapz(integrand_N_delv_N_delv,Kp_perp,axis=0),Kp_par,axis=0)

                F_integral[i,j,rdsft]=integral_P_deldel_P_vv+integral_P_delv_P_delv+integral_P_vv_N_deldel+integral_P_deldel_N_vv+integral_P_delv_N_delv_1+integral_P_delv_N_delv_2+integral_N_deldel_N_vv+integral_N_delv_N_delv
    return F_integral
'''
kpar=np.geomspace(1.e-4,.15,n)
y=kpar*chi(1.26)
z1=1.26
delta_z=.0015
Z_min=z1-delta_z
Z_max=z1+delta_z
z=np.geomspace(Z_min,Z_max,n)
const=1./(cc.c_light_Mpc_s**2*16*np.pi**3)

PN_int_over_y=const*trapz(trapz(F_integral_rect_coords_integral(ell,1.26,y,z, Hirax_noise_z_1pt26_deltaz_pt0015),y,axis=1),z,axis=1)
#print (F_integral_rect_coords_integral(ell,1.26,y,z, Hirax_noise_z_1pt26_deltaz_pt0015),'F int rect coords')
#print (const*PN_int_over_y)
#plt.plot(ell,const*PN_int_over_y)
#plt.show()

            P_deldel_P_vv=np.append(P_deldel_P_vv,integral_P_deldel_P_vv)
            P_delv_P_delv=np.append(P_delv_P_delv, integral_P_delv_P_delv)
            P_vv_N_deldel=np.append(P_vv_N_deldel, integral_P_vv_N_deldel)
            P_deldel_N_vv=np.append(P_deldel_N_vv, integral_P_deldel_N_vv)
            P_delv_N_delv_1=np.append(P_delv_N_delv_1, integral_P_delv_N_delv_1)
            P_delv_N_delv_2=np.append(P_delv_N_delv_2, integral_P_delv_N_delv_2)
            N_deldel_N_vv=np.append(N_deldel_N_vv, integral_N_deldel_N_vv)
            N_delv_N_delv=np.append(N_delv_N_delv, integral_N_delv_N_delv)

    return P_deldel_P_vv, P_delv_P_delv, P_vv_N_deldel, P_deldel_N_vv, P_delv_N_delv_1, P_delv_N_delv_2, N_deldel_N_vv, N_delv_N_delv
'''

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
