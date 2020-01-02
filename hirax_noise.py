import numpy as np
import matplotlib.pyplot as plt
import useful_functions as uf
import sys
sys.path.insert(0,'/home/zahra/hirax_tools/hirax_tools')
from array_config import HIRAXArrayConfig
from neutral_H_autocorr import Cl_21_func_of_y,Cl_21,Integrand_doppler_21cm,Cl_21_momentum_single,Cl_21,Cl_21_doppler,Cl_21_momentum_integrated
import scipy as sp
from scipy.interpolate import interp1d,UnivariateSpline,splrep, splev
#import spline
from mpl_toolkits.mplot3d import Axes3D
import pylab
import constants as cc
import density as den
from bispec import crosscorr_squeezedlim
from lin_kSZ_redshift_dep import C_l_mu_integral
from noise_temp_old import PN_integrals_no_redshift_int


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
n=100
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

ell_large=np.geomspace(1.e-10,1.e6,10000)

ell=np.geomspace(1.,1.e4,n)
kpar_arr=np.geomspace(1.e-4,.15,n)
y=kpar_arr*r(1.26)


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

Hirax_noise_z_1pt26_deltaz_pt0015 = interp1d(ell_large, HiraxNoise(ell_large,6.,7.,1.26,0.0015), bounds_error=False)


SKA_noise_dish_2d=np.genfromtxt('/home/zahra/Downloads/SKA1mid_single_dish/SKA1Mid_single_dish_noise_zc_1_26.dat',
                   dtype=float,unpack=True)
SKA_noise_interferom_2d=np.genfromtxt('/home/zahra/Downloads/SKAmid1_interferometer_Noise/SKA1Mid_interferometer_noise_zc_1_26.dat',
                dtype=float,unpack=True)

ell_dish=SKA_noise_dish_2d[:,0]
SKA_noise_dish=SKA_noise_dish_2d[:,1]
ell_interferom=SKA_noise_interferom_2d[:,0]
SKA_noise_interferom=SKA_noise_interferom_2d[:,1]

SKA_noise_dish_interp = interp1d(ell_dish, SKA_noise_dish) #original values are in mK^2 so we convert to microK^2

#tck = splrep(ell_dish, SKA_noise_dish, s=0)
#SKA_noise_dish_spline=splev(ell, tck,der=0)
#SKA_noise_dish_spline.set_smoothing_factor(0.5)
SKA_noise_interferom_interp=interp1d(ell_interferom, SKA_noise_interferom)
#SKA_noise_interferom_spline=UnivariateSpline(ell_interferom, SKA_noise_interferom)

SKA_noise_total_interp=interp1d(ell_interferom,1./(1./(1.e6*SKA_noise_dish_interp(ell_interferom))+1./(1.e6*SKA_noise_interferom_interp(ell_interferom))))
#SKA_noise_total_spline=UnivariateSpline(ell_interferom,1./(1./(1.e6*SKA_noise_dish_interp(ell_interferom))+1./(1.e6*SKA_noise_interferom_interp(ell_interferom))))

'''
kperp_min=1.e-3
kpar_min=.075
delta_kperp=.03
delta_kpar=.03

kperp_arr=np.linspace(kperp_min,.2,n)
ell=kperp_arr*chi(z)
kpar_arr=np.linspace(kpar_min,.2,n)
y=kpar_arr*uf.r(z)
'''

OV_1pt21_10=np.load('OV_1pt21_to_10_200pts.npy')

OV_1pt31_10=np.load('OV_1pt31_to_10_200pts.npy')
ell_loaded=OV_1pt21_10[0,:]
OV_1pt21_loaded=OV_1pt21_10[1,:]
OV_1pt31_loaded=OV_1pt31_10[1,:]


OV_1pt21_interp=interp1d(ell_loaded,OV_1pt21_loaded)
OV_1pt31_interp=interp1d(ell_loaded,OV_1pt31_loaded)

OV_interp=interp1d(ell_loaded,OV_1pt21_interp(ell_loaded)-OV_1pt31_interp(ell_loaded))



'''
plt.loglog(kperp_arr,Hirax_noise_z_1pt26_deltaz_pt2(ell))
plt.loglog(kperp_arr,Hirax_noise_z_1pt11_deltaz_pt05(ell)+Hirax_noise_z_1pt21_deltaz_pt05(ell)+Hirax_noise_z_1pt31_deltaz_pt05(ell)
            +Hirax_noise_z_1pt41_deltaz_pt05(ell))
plt.ylabel(r'$\rm N_l^{HIRAX}(z) (\mu K^2)$')
plt.xlabel(r'$\rm k_\perp$')
plt.ylim(1e-4,10)
plt.xlim(1e-2,1)
plt.legend(('Noise in a 0.4 redshift bin','Sum of noise in four redshift bins of 0.1'))
plt.show()


plt.semilogx(kperp_arr,crosscorr_squeezedlim_all_ell(ell,1.26,100.,0.2))
plt.semilogx(kperp_arr,crosscorr_squeezedlim_all_ell(ell,1.11,100.,0.05)+crosscorr_squeezedlim_all_ell(ell,1.21,100.,0.05)
            +crosscorr_squeezedlim_all_ell(ell,1.31,100.,0.05)+crosscorr_squeezedlim_all_ell(ell,1.41,100.,0.05))
plt.ylabel(r'$\rm B_l^{21-kSZ}(y=100,z)$')
plt.legend(('Signal in a 0.4 redshift bin','Sum of signals in four redshift bins of 0.1'))
#plt.ylim(1e-10,3e-10)
plt.xlabel(r'$\rm k_\perp$')
plt.show()

plt.loglog(kperp_arr,1.e6*SKA_noise_dish_interp(ell))
plt.loglog(kperp_arr,1.e6*SKA_noise_interferom_interp(ell))
plt.ylim(1e-5,1e10)
plt.show()


plt.semilogy(ell_dish,SKA_noise_dish_interp(ell_dish))
plt.show()

SKA_dish_inv=1./(1.e6*SKA_noise_dish_interp(ell))
SKA_int_inv=1./(1.e6*SKA_noise_interferom_interp(ell))


plt.loglog(kperp_arr,HiraxNoise(ell,6.,7.,1.26,0.05))
plt.ylim(1e-5,100)

plt.show()

#plt.loglog(kperp_arr,Hirax_noise_z_1pt26_deltaz_pt05(ell),'b')
plt.loglog(kperp_arr,1.e6*SKA_noise_dish_interp(ell),'r')
plt.loglog(kperp_arr,1.e6*SKA_noise_dish_spline,'g')

#plt.loglog(kperp_arr,1.e6*SKA_noise_interferom_interp(ell),'g')
#plt.loglog(kperp_arr,SKA_noise_total_interp(ell),'k')
#plt.semilogy(ell,SKA_noise_total_spline(ell),'k')
plt.ylabel(r'$\rm N_l^{HI}(z=1.26) (\mu K^2)$')
plt.xlabel(r'$\rm k_\perp$')
#plt.ylim(1e-5,100)
plt.legend(('Hirax noise','SKA dish','SKA interferometer','SKA total'))
plt.show()
'''

#kSZ_OV_single_bin=C_l_mu_integral(ell,z+delta_z)-C_l_mu_integral(ell,z-delta_z)

def Noise_21cm_vel(ell,z,y,delta_z,Noise):
    k=np.sqrt(y**2/r(z)**2+ell**2/chi(z)**2)
    mu_k=(y/r(z))/k
    v_expression_dimless=1/c*f(z)*H(z)/(1+z)*mu_k/k ##took out the D^2 factor-don't know if I need it
    return Noise(ell)*v_expression_dimless**2

def Noise_21cm_vel_integrated_over_y(ell,z,delta_z,Noise):
    N_vv=np.array([])
    for i in ell:
        k=np.sqrt(y**2/r(z)**2+i**2/chi(z)**2)
        mu_k=(y/r(z))/k
        v_expression_dimless=1/c*f(z)*H(z)/(1+z)*mu_k/k ##took out the D^2 factor-don't know if I need it
        integral=sp.integrate.trapz(Noise(i)*v_expression_dimless**2,y)
        N_vv=np.append(N_vv,integral)
    return N_vv



def Func_2d(ell,z,y,Func,delta_z):
    Func_mat=np.zeros((len(ell),len(y)))
    for i in range(len(ell)):
        for j in range(len(y)):
            Func_ind=Func(ell[i],z,y[j],delta_z)
            Func_mat[i][j]=Func_ind
    return Func_mat


def Func_noise(ell,z,y,Func,delta_z,Noise):
    Func_mat=np.zeros((len(ell),len(y)))
    for i in range(len(ell)):
        for j in range(len(y)):
            Func_ind=Func(ell[i],z,y[j],delta_z,Noise)
            Func_mat[i][j]=Func_ind
    return Func_mat


def Cl_vel(ell,z,y): ###Just a check that this is the same as the Integrand_doppler_21cm function, which it now is
    k=np.sqrt(y**2/r(z)**2+ell**2/chi(z)**2)
    mu_k=(y/r(z))/k
    v_expression_dimless=1./c*f(z)*H(z)/(1+z)*mu_k/k ##took the D^2 factor out because the 21cm density Cl has this factor already
    print (v_expression_dimless,'v_fac')
    return Cl_21_func_of_y(ell,y)*v_expression_dimless**2

#print (Cl_vel(2000,3000))

def HI_den_SNR(ell,z,y,delta_z,ell_2d,Noise):
    S=Func_2d(ell,z,y,Cl_21_func_of_y,delta_z)
    N=Noise(ell_2d)
    sigma=S+N
    return S/sigma

def HI_vel_SNR(ell,z,y,ell_2d,delta_z,Noise_21cm_vel,Noise):
    S=Func_2d(ell,z,y,Integrand_doppler_21cm,delta_z)
    N=Func_2d_vel_noise(ell,z,y,Noise_21cm_vel,Noise)
    sigma=S+N
    return S/sigma

Kp_min=1.e-6
Kp_max=1.e-1

def N_P_4(ell,z_1,y,delta_z,Noise):
    z_2=z_1
    kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp_perp=np.linspace(Kp_min,Kp_max,n)
    Kp_par=np.linspace(Kp_min,Kp_max,n)
    Kp=np.sqrt(Kp_perp**2+Kp_par**2)
    mu,kp=np.meshgrid(Mu,Kp)
    k_perp=ell/chi(z_1)
    k=np.sqrt(kpar**2+k_perp**2)
    K_perp=np.sqrt(np.abs(k_perp**2+Kp_perp**2-2*k_perp*Kp_perp*mu))
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
    theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
    theta_K=np.abs(kpar/K/k*(k-kp*mu)-kp*k_perp*np.sqrt(1-mu**2)/k/K)
    mu_k_sq=kpar**2/k**2
    a=uf.b_HI+f(z_1)*mu_k_sq
    const=1/(8*np.pi**2*cc.c_light_Mpc_s**2)*T_mean(z_1)**2*D(z_1)**2*f(z_1)**2*H(z_1)**2/(1+z_1)**2#/chi(z_1)**2/r(z_1)
    #integrand=a**2*theta_kp**2*Noise(K*chi(z_1))*Mps_interpf(kp)
    t1=theta_kp*(Noise(K_perp*chi(z_1))*Mps_interpf(kp)/kp**2+Noise(Kp_perp*chi(z_1))*Mps_interpf(K)/Kp_perp**2)
    t2=theta_K*(Noise(K_perp*chi(z_1))*Mps_interpf(kp)/(kp*K_perp)+Noise(Kp_perp*chi(z_1))*Mps_interpf(K)/(kp*K_perp))
    integrand=a**2*theta_kp*kp**2*(t1+t2)
    integral=const*sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0)
    #integrand=Noise(K*chi(z_1))
    #integral1=sp.integrate.trapz(integrand,mu)
    #integral=sp.integrate.trapz(integral1*Mps_interpf(kp),kp)
    #integral=sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0)*Mps_interpf(kp),Kp,axis=0)
    return integral

def N_N_2(ell,z_1,y,delta_z,Noise):
    z_2=z_1
    kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp_perp=np.linspace(Kp_min,Kp_max,n)
    Kp_par=np.linspace(Kp_min,Kp_max,n)
    Kp=np.sqrt(Kp_perp**2+Kp_par**2)
    mu,kp=np.meshgrid(Mu,Kp)
    k_perp=ell/chi(z_1)
    k=np.sqrt(kpar**2+k_perp**2)
    K_perp=np.sqrt(np.abs(k_perp**2+Kp_perp**2-2*k_perp*Kp_perp*mu))
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
    theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
    theta_K=np.abs(kpar/K/k*(k-kp*mu)-kp*k_perp*np.sqrt(1-mu**2)/k/K)
    mu_k_sq=kpar**2/k**2
    a=uf.b_HI+f(z_1)*mu_k_sq
    const=1/(8*np.pi**2*cc.c_light_Mpc_s**2)*f(z_1)**2*H(z_1)**2/(1+z_1)**2*chi(z_1)**2*r(z_1)
    t1=theta_kp*Noise(K_perp*chi(z_1))*Noise(kp*chi(z_1))/Kp_perp**2
    t2=theta_K*Noise(K_perp*chi(z_1))*Noise(kp*chi(z_1))/(Kp_perp*K_perp)
    integrand=a**2*theta_kp*kp**2*(t1+t2)
    integral=const*sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0)
    #integrand=Noise(K*chi(z_1))
    #integral1=sp.integrate.trapz(integrand,mu)
    #integral=sp.integrate.trapz(integral1*Mps_interpf(kp),kp)
    #integral=sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0)*Mps_interpf(kp),Kp,axis=0)
    return integral

def N_delta_P_vv_full_ell(ell,z_1,y,delta_z,Noise):
    z_2=z_1
    kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp_perp=np.linspace(1.e-6,1.e-1,n)
    Kp_par=np.linspace(1.e-6,1.e-1,n)
    Kp=np.sqrt(Kp_perp**2+Kp_par**2)
    mu,kp=np.meshgrid(Mu,Kp)
    full_integral=np.array([])
    for i in ell:
        k_perp=i/chi(z_1)
        k=np.sqrt(kpar**2+k_perp**2)
        K=np.sqrt(np.abs(k_perp**2+Kp_perp**2-2*k_perp*Kp_perp*mu))
        theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
        theta_K=kpar/K/k*(k-kp*mu)-kp*k_perp*np.sqrt(1-mu**2)/k/K
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f(z_1)*mu_k_sq
        const=1/(8*np.pi**2*cc.c_light_Mpc_s**2)*T_mean(z_1)**2*D(z_1)**2*f(z_1)**2*H(z_1)**2/(1+z_1)**2#/chi(z_1)**2/r(z_1)
        integrand=a**2*theta_kp**2*Noise(K*chi(z_1))*Mps_interpf(kp)
        integral=const*sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0)
        full_integral=np.append(full_integral,integral)
    return full_integral

def P_delta_N_vv_integrate_over_y(ell,z_1,delta_z,Noise):
    z_2=z_1
    y=np.geomspace(1.,3000.,n)
    Kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp_perp=np.geomspace(1.e-10,1.e-1,n)
    Kp_par=np.geomspace(1.e-10,1.e-1,n)
    Kp=np.sqrt(Kp_perp**2+Kp_par**2)
    mu,kp,kpar=np.meshgrid(Mu,Kp,Kpar)
    SN_21_y_integ=np.array([])
    Kperp_arr=np.array([])
    for i in ell:
        k_perp=i/chi(z_1)
        k=np.sqrt(kpar**2+k_perp**2)
        zeta=(kpar/k*mu+k_perp/k*np.sqrt(np.abs(1-mu**2)))
        K_perp=np.sqrt(np.abs(k_perp**2+Kp_perp**2-2*k_perp*Kp_perp*zeta))
        Kperp_arr=np.append(Kperp_arr,K_perp)
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*zeta))
        theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
        theta_K=kpar/K/k*(k-kp*zeta)-kp*k_perp*np.sqrt(np.abs(1-zeta**2))/k/K
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f(z_1)*mu_k_sq
        const=1/(16*np.pi**3*cc.c_light_Mpc_s**2)*T_mean(z_1)**2*D(z_1)**2*f(z_1)**2*H(z_1)**2/(1+z_1)**2*r(z_1)
        #integrand=a**2*theta_kp**2*kp**2*(Noise(K_perp*chi(z_1))*Mps_interpf(kp)/kp**2)
        integrand=a**2*theta_kp**2*kp**2 *Noise(Kp_perp*chi(z_1)) *Mps_interpf(K)/kp**2
        integral=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0),Kpar,axis=0)
        SN_21_y_integ=np.append(SN_21_y_integ,integral)
    return SN_21_y_integ,Kperp_arr

def N_delta_P_vv_P_delta_N_vv_integrate_over_y(ell,z_1,delta_z,Noise):
    z_2=z_1
    y=np.linspace(1.,3000.,n)
    Kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp_perp=np.linspace(1.e-6,1.e-1,n)
    Kp_par=np.linspace(1.e-6,1.e-1,n)
    Kp=np.sqrt(Kp_perp**2+Kp_par**2)
    mu,kp,kpar=np.meshgrid(Mu,Kp,Kpar)
    SN_21_y_integ=np.array([])
    for i in ell:
        k_perp=i/chi(z_1)
        k=np.sqrt(kpar**2+k_perp**2)
        K_perp=np.sqrt(np.abs(k_perp**2+Kp_perp**2-2*k_perp*Kp_perp*mu))
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
        theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
        theta_K=kpar/K/k*(k-kp*mu)-kp*k_perp*np.sqrt(1-mu**2)/k/K
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f(z_1)*mu_k_sq
        const=1/(16*np.pi**3*cc.c_light_Mpc_s**2)*T_mean(z_1)**2*D(z_1)**2*f(z_1)**2*H(z_1)**2/(1+z_1)**2*r(z_1)
        integrand=a**2*theta_kp**2*kp**2*(Noise(K_perp*chi(z_1))*Mps_interpf(kp)/kp**2+Noise(Kp_perp*chi(z_1))*Mps_interpf(K)/Kp_perp**2)
        integral=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0),Kpar,axis=0)
        SN_21_y_integ=np.append(SN_21_y_integ,integral)
    return SN_21_y_integ

def N_delta_N_vv_integrate_over_y(ell,z_1,delta_z,Noise):
    z_2=z_1
    y=np.linspace(1.,3000.,n)
    Kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp_perp=np.linspace(1.e-6,1.e-1,n)
    Kp_par=np.linspace(1.e-6,1.e-1,n)
    Kp=np.sqrt(Kp_perp**2+Kp_par**2)
    mu,kp,kpar=np.meshgrid(Mu,Kp,Kpar)
    SN_21_y_integ=np.array([])
    for i in ell:
        k_perp=i/chi(z_1)
        k=np.sqrt(kpar**2+k_perp**2)
        K_perp=np.sqrt(np.abs(k_perp**2+Kp_perp**2-2*k_perp*Kp_perp*mu))
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
        theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
        theta_K=kpar/K/k*(k-kp*mu)-kp*k_perp*np.sqrt(1-mu**2)/k/K
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f(z_1)*mu_k_sq
        const=1/(16*np.pi**3*cc.c_light_Mpc_s**2)*f(z_1)**2*H(z_1)**2/(1+z_1)**2*r(z_1)**2*chi(z_1)**2
        integrand=theta_kp**2*kp**2*(Noise(K_perp*chi(z_1))*Noise(Kp_perp*chi(z_1)))/Kp_perp**2
        integral=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0),Kpar,axis=0)
        SN_21_y_integ=np.append(SN_21_y_integ,integral)
    return SN_21_y_integ

def N_delta_v_P_delta_v_sq_integrate_over_y(ell,z_1,delta_z,Noise):
    z_2=z_1
    y=np.linspace(1.,3000.,n)
    Kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp_perp=np.linspace(1.e-6,1.e-1,n)
    Kp_par=np.linspace(1.e-6,1.e-1,n)
    Kp=np.sqrt(Kp_perp**2+Kp_par**2)
    mu,kp,kpar=np.meshgrid(Mu,Kp,Kpar)
    SN_21_y_integ=np.array([])
    for i in ell:
        k_perp=i/chi(z_1)
        k=np.sqrt(kpar**2+k_perp**2)
        K_perp=np.sqrt(np.abs(k_perp**2+Kp_perp**2-2*k_perp*Kp_perp*mu))
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
        theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
        theta_K=kpar/K/k*(k-kp*mu)-kp*k_perp*np.sqrt(1-mu**2)/k/K
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f(z_1)*mu_k_sq
        const=1/(16*np.pi**3*cc.c_light_Mpc_s**2)*T_mean(z_1)**2*D(z_1)**2*f(z_1)**2*H(z_1)**2/(1+z_1)**2*r(z_1)
        integrand=a**2*theta_kp*theta_K*kp**2*(Noise(K_perp*chi(z_1))*Mps_interpf(kp)/(kp*K_perp)+Noise(Kp_perp*chi(z_1))*Mps_interpf(K)/(Kp_perp*K))
        integral=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0),Kpar,axis=0)
        SN_21_y_integ=np.append(SN_21_y_integ,integral)
    return SN_21_y_integ

def N_delta_N_vv_integrate_over_y_second(ell,z_1,delta_z,Noise):
    z_2=z_1
    y=np.linspace(1.,3000.,n)
    Kpar=y/uf.r(z_1)
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp_perp=np.linspace(1.e-6,1.e-1,n)
    Kp_par=np.linspace(1.e-6,1.e-1,n)
    Kp=np.sqrt(Kp_perp**2+Kp_par**2)
    mu,kp,kpar=np.meshgrid(Mu,Kp,Kpar)
    SN_21_y_integ=np.array([])
    for i in ell:
        k_perp=i/chi(z_1)
        k=np.sqrt(kpar**2+k_perp**2)
        K_perp=np.sqrt(np.abs(k_perp**2+Kp_perp**2-2*k_perp*Kp_perp*mu))
        K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*mu))
        theta_kp=kpar*mu/k+k_perp*np.sqrt(1-mu**2)/k
        theta_K=np.abs(kpar/K/k*(k-kp*mu)-kp*k_perp*np.sqrt(1-mu**2)/k/K)
        mu_k_sq=kpar**2/k**2
        a=uf.b_HI+f(z_1)*mu_k_sq
        const=1/(16*np.pi**3*cc.c_light_Mpc_s**2)*f(z_1)**2*H(z_1)**2/(1+z_1)**2*r(z_1)**2*chi(z_1)**2
        integrand=theta_kp*theta_K*kp**2*(Noise(K_perp*chi(z_1))*Noise(Kp_perp*chi(z_1)))/(K_perp*Kp_perp)
        integral=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0),Kpar,axis=0)
        SN_21_y_integ=np.append(SN_21_y_integ,integral)
    return SN_21_y_integ

Noise= Hirax_noise_z_1pt26_deltaz_pt0015 #SKA_noise_total_interp

'''
plt.loglog(ell,HiraxNoise(ell,6,7,z,.05))
plt.loglog(ell,HiraxNoise(ell,6,7,z,.2))
plt.ylim(1e-5,1e2)
plt.xlim(10,1e4)
plt.show()
#print ((N_delta_P_vv_full_ell(ell,z,y,delta_z,Noise)[1]*chi(z))[:100],'ell arr')
#print (N_delta_P_vv_integrate_over_y(ell,z,delta_z,Noise))
'''
#kp=np.linspace(1.e-6,2.,n)
#print (ell.max())
#print ((kp*chi(z)).max())
#Kp_perp=np.linspace(2.e-6,1.e-1,n)

#print (P_delta_N_vv_integrate_over_y(ell,z,delta_z,Noise)[1][:100]-Kp_perp)
'''
print (K_arr[:100]*chi(z))

print (Noise(ell))
print (Mps_interpf(kp))
print (Noise(ell)*Mps_interpf(kp))
print (N_delta_P_vv(ell,z,100,delta_z,Noise)[0])


print (N_delta_P_vv_integrate_over_y(ell,z,delta_z,Noise))
'''

#plt.loglog(ell,Noise_21cm_vel_integrated_over_y(ell,z,delta_z,Noise),'r')
#plt.loglog(ell,Noise(ell),'b')
#plt.show()
#print (P_delta_N_vv_integrate_over_y(ell,z,delta_z,Noise)[0])

#plt.plot(ell,P_delta_N_vv_integrate_over_y(ell,z,delta_z,Noise)[0],'k')
#plt.loglog(ell,Cl_21_doppler(ell,z)*Noise(ell))
#plt.loglog(ell,Noise_21cm_vel_integrated_over_y(ell,z,delta_z,Noise)*Cl_21(ell,z),'b')
#plt.loglog(ell,N_delta_P_vv_P_delta_N_vv_integrate_over_y(ell,z,delta_z,Noise)+N_delta_N_vv_integrate_over_y(ell,z,delta_z,Noise),'r')
#plt.plot(ell,N_delta_N_vv_integrate_over_y_second(ell,z,delta_z,Noise))
#plt.plot(ell,N_delta_v_P_delta_v_sq_integrate_over_y(ell,z,delta_z,Noise))
#plt.loglog(ell,N_delta_v_P_delta_v_sq_integrate_over_y(ell,z,delta_z,Noise)+N_delta_N_vv_integrate_over_y_second(ell,z,delta_z,Noise),'b')
#plt.loglog(ell,Cl_21_momentum_integrated(ell,z),'g')
#plt.legend(('Ndelta Pv','Ndelta Pv Pdelta Nv + Ndelta Nv','Ndelta Pv + Ndelta Nv','Cl p21'))
#plt.show()



sys.path.insert(0,'/home/zahra/python_scripts/CMB_noise')
import scal_power_spectra as cmb

cmb_spec=cmb.cl_tt_func_bis
cmb_noise=cmb.nl_tt_ref_func_bis



def Bispec_SNR(ell,z,y,delta_z,ell_2d,Noise):
    S = Func_2d(ell,z,y,crosscorr_squeezedlim,delta_z)
    N_ksz=cmb_spec(ell_2d)+cmb_noise(ell_2d)
    S_ksz=
    S_N_p21=Func_noise(ell,z,y,PN_integrals_no_redshift_int,delta_z, Noise)
    #S_21_den=Func_2d(ell,z,y,Cl_21_func_of_y,delta_z)
    #S_21_vel=Func_2d(ell,z,y,Integrand_doppler_21cm,delta_z)
    #S_21_momentum=Func_2d(ell,z,y,Cl_21_momentum_single,delta_z)
    #N_21_den_Pvv_2d=Func_noise(ell,z,y,N_delta_P_vv,delta_z,Noise)
    #PN_4_NN_2=Func_noise(ell,z,y,N_P_4,delta_z,Noise)+Func_noise(ell,z,y,N_N_2,delta_z,Noise)
    #N_21_den=Noise(ell_2d)
    #N_21_momentum=N_21_den*N_21_vel*S_21_momentum/(S_21_den*S_21_vel)
    #variance=(S_ksz+N_ksz)*(S_21_momentum+N_21_den_Pvv_2d)+3.*S**2
    variance=(S_ksz+N_ksz)*(S_N_p21)+3.*S**2
    #variance=(S_ksz+N_ksz)*(S_21_vel+N_21_vel)*(S_21_den+N_21_den)+3.*S**2
    sigma=np.sqrt(variance)
        #y_ind=np.linspace(y[j],y[j+1],n_new)
    #Ell,Y=np.meshgrid(ell_ind,y_ind)
    return S/sigma,S,sigma

kperp_arr=ell/chi(z)
kpar_arr=y/r(z)

ell_2d=np.outer(ell,np.ones(len(y)))
pylab.pcolormesh(kperp_arr,kpar_arr,Bispec_SNR(ell,z,y,delta_z,ell_2d,Hirax_noise_z_1pt26_deltaz_pt05)[2]) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
pylab.xlim([np.min(kperp_arr),np.max(kperp_arr)]) ; pylab.ylim([np.min(kpar_arr),np.max(kpar_arr)])
plt.xlabel(r'$k_\perp$',fontsize=12); plt.ylabel(r'$k_\parallel$',fontsize=12); plt.title('2D sigma', x=1.13, y=1.05)
plt.show()

ell_ind=592.
kperp=ell_ind/chi(1)
y_ind=581.
kpar=y_ind/r(1)
'''
print (Func_2d(ell,y,crosscorr_squeezedlim).max(),'bispec_signal')
print (cmb_spec(ell).max(),'cmb_spec')
print (cmb_noise(ell).max(),'cmb_noise')
print (interp_OV_full_signal(ell).max(),'OV_signal')
print (Func_2d(ell,y,Cl_21_func_of_y).max(),'S_den')
print (Func_2d(ell,y,Integrand_doppler_21cm).max(),'S_vel')
print (Hirax_noise_z_1_interp(ell).max(),'N_den')
print (Func_2d(ell,y,Hirax_noise_21cm_vel).max(),'N_vel')
print (HI_den_SNR(ell,y,ell).max(),'HI_den_snr')
print (Bispec_SNR(ell,y,ell).max(),'SNR')


#######################################################

#CHECKING ORDER OF MAGNITUDES FOR WORK WRITE UP

Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
fsky=Sarea/4./np.pi
delta_nu_dimless=400./1420.
Mode_Volume_Factor=fsky*Sarea*delta_nu_dimless#*2
const=Mode_Volume_Factor*0.5/4./np.pi**2
br1=interp_OV_full_signal(ell_ind)+cmb_spec(ell_ind)+cmb_noise(ell_ind)
print (interp_OV_full_signal(ell_ind),'kSZ_signal')
print (cmb_spec(ell_ind),'cmbspec')
print (cmb_noise(ell_ind),'cmbnoise')
br2=Hirax_noise_z_1_interp(ell_ind)+Hirax_noise_21cm_vel(ell_ind,y_ind)+Cl_21_func_of_y(ell_ind,y_ind)+Integrand_doppler_21cm(ell_ind,y_ind)
print (Hirax_noise_z_1_interp(ell_ind),'21_den_noise')
print (Hirax_noise_21cm_vel(ell_ind,y_ind),'21_vel_noise')
print (Cl_21_func_of_y(ell_ind,y_ind),'21_den_sig')
print (Integrand_doppler_21cm(ell_ind,y_ind),'21_vel_signal')
t1=br1*br2
t2=3*crosscorr_squeezedlim(ell_ind,y_ind)**2
var=t1+t2
print (var)
S=crosscorr_squeezedlim(ell_ind,y_ind)
print (S,'bispec_signal')
bin_widths=0.01**2*r(1)*chi(1)
SNR=np.sqrt(const*bin_widths*S**2/var*ell_ind)
print (SNR,kperp,kpar)

S_21_vel=Integrand_doppler_21cm(ell_ind,y_ind)
N_21_vel=Hirax_noise_21cm_vel(ell_ind,y_ind)
sigma_vel=S_21_vel+N_21_vel
SNR=np.sqrt(const*bin_widths*S_21_vel**2/sigma_vel**2*ell_ind)
print (SNR,'21_vel')

S_21_den=Cl_21_func_of_y(ell_ind,y_ind)
N_21_den=Hirax_noise_z_1_interp(ell_ind)
sigma_den=S_21_den+N_21_den
SNR=np.sqrt(const*bin_widths*S_21_den**2/sigma_den**2*ell_ind)
print (SNR,'21_den')
######################################################################

plt.plot(ell,Cl_vel(ell,72),'b')
plt.plot(ell,Integrand_doppler_21cm(ell,1,72),'g')
plt.show()

#print (ell)
#print (Hirax_noise_21cm_vel(ell,72.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,72.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,572.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,1072.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,2072.))
plt.loglog(ell,Hirax_noise_21cm_vel(ell,2572.))
plt.loglog(ell,Hirax_noise_z_1_interp(ell),'k')
plt.legend(('y=72','y=572','y=1072','y=2072','y=2572'))
plt.xlabel('l')
plt.ylabel(r'$\rm{N_l^{HI}(y,z=1)/\sqrt{f_{sky}l [\mu K^2]}$')
plt.xlim(100,5000)
plt.ylim(1e-16,1)
plt.show()
#print (Cl_21_func_of_y(1,277,ell)+HiraxNoise(ell,6,7,1))
#plt.plot(ell,HiraxNoise(ell,6,7,1))
#plt.plot(ell,Hirax_noise_z_1_interp(ell))
#plt.show()
'''


'''
plt.loglog(ell,Noise(ell)*Noise_21cm_vel(ell,z,100,Noise)*Cl_21_momentum_integrated(ell,z)/(Cl_21(ell,z)*Cl_21_doppler(ell,z)),'r')
plt.loglog(ell,crosscorr_squeezedlim_integral_y_all_ell(ell,z,delta_z),'b')
plt.xlabel('l')
plt.ylim(1e-8,1e-3)
plt.ylabel('Bispectrum')
plt.legend(('Bispectrum noise','Bispectrum signal'))
plt.show()

'''
#pylab.pcolormesh(Bispec_2d_interp(ell,y))
#pylab.show()
#Cl_21_2d_interp = sp.interpolate.interp2d(ell, y, Cl_21_2d(ell,1,y)) #get same result from using mesh grid

def SNR_binned(z,delta_z,Noise,SNR,Sarea_deg,Dsep):
    z_min=z-delta_z
    z_max=z+delta_z
    nu_min=nu_21/(1+z_max) #743. #585. # #freq in MHz, these min and max values are based on HIRAX
    nu_max=nu_21/(1+z_min) #784. #665. #
    nubar=(nu_min+nu_max)/2.
    cp= HIRAXArrayConfig.from_n_elem_compact(1024,Dsep)
    Fov_str=(cp.fov(frequency=nubar))
    print (Fov_str,'fov')
    Sarea=Sarea_deg*(np.pi/180.)**2 #converting from square degrees to square radians
    print (Sarea,'sarea')
    fsky=Sarea/4./np.pi
    delta_nu_dimless=(nu_max-nu_min)/nu_21
    N_patches=Sarea/Fov_str
    Mode_Volume_Factor=fsky*Sarea*delta_nu_dimless#*2
    num_k_bins_y=25
    num_k_bins_ell=25
    n=100
    kpar_arr = np.zeros(num_k_bins_y) ; kperp_arr= np.zeros(num_k_bins_ell) ; SNR_arr = np.zeros((num_k_bins_y,num_k_bins_ell))
    kpar_min = .01
    kperp_min =.01
    delta_kpar = .03
    delta_kperp = .03
    y_min=kpar_min*r(z)
    ell_min=kperp_min*chi(z)
    delta_y=delta_kpar*r(z)
    delta_ell=delta_kperp*chi(z)
    y_arr=np.linspace(y_min,y_min+num_k_bins_y*delta_y,n)
    ell_arr=np.linspace(ell_min,ell_min+num_k_bins_ell*delta_ell,n)
    for bin_number_y in np.linspace(0,num_k_bins_y-1,num_k_bins_y):
        kpar_bin_min, kpar_bin_max = kpar_min + delta_kpar*np.array([bin_number_y, bin_number_y+1])    # bin_number starts at 0
        #print (kpar_bin_min*r(z),kpar_bin_max*r(z),'kpar')
        for bin_number_ell in np.linspace(0,num_k_bins_ell-1,num_k_bins_ell):
            kperp_bin_min, kperp_bin_max = kperp_min + delta_kperp*np.array([bin_number_ell, bin_number_ell+1])
            #print (kperp_bin_min*r(z),kperp_bin_max*r(z),'kperp')
            kpar_arr[np.int(bin_number_y)] = kpar_bin_min  ; kperp_arr[np.int(bin_number_ell)] = kperp_bin_min
            y_bin_arr = y_arr[(y_arr > kpar_bin_min*r(z)) & (y_arr < kpar_bin_max*r(z))]
            ell_bin_arr = ell_arr[(ell_arr > kperp_bin_min*chi(z)) & (ell_arr < kperp_bin_max*chi(z))]
            ell_bin_2d_arr=np.outer(ell_bin_arr,np.ones(len(y_bin_arr)))
            #SN_ratio_2d_arr = SNR(ell_bin_arr,y_bin_arr) #SNR(Ell,Y)
            SN_ratio_2d_arr=SNR(ell_bin_arr,z,y_bin_arr,delta_z,ell_bin_2d_arr,Noise)[0]
            integrand=SN_ratio_2d_arr**2*ell_bin_2d_arr
            int1=sp.integrate.trapz(integrand, ell_bin_arr, axis=0)
            SNR_sq = 0.5 * Mode_Volume_Factor/(4.*np.pi**2)*N_patches * sp.integrate.trapz(int1, y_bin_arr, axis=0)
            SNR_arr[np.int(bin_number_y),np.int(bin_number_ell)] = np.sqrt(SNR_sq)
                # print 'kparmin=', kpar_bin_min, 'kperpmin=', kperp_bin_min, np.sqrt(SNR_sq)
    return kperp_arr,kpar_arr,SNR_arr

'''
plt.loglog(ell,1e-6*Cl_21_func_of_y(ell,z,72))
plt.loglog(ell,1e-6*Cl_21_func_of_y(ell,z,572))
plt.loglog(ell,1e-6*Cl_21_func_of_y(ell,z,1072))
plt.loglog(ell,1e-6*Cl_21_func_of_y(ell,z,1572))
plt.loglog(ell,1e-6*Cl_21_func_of_y(ell,z,2072))


plt.legend(('y=72','y=572','y=1072','y=1572','y=2072'))
plt.loglog(ell,1e-6*Noise(ell)/np.sqrt(ell*0.36),'--')

#plt.loglog(ell,Hirax_noise_z_1_interp(ell))
plt.xlabel('l')
#plt.ylabel('S/N')
plt.xlim(50,5e3)
plt.ylim(1e-12,1e-8)
plt.ylabel(r'$\rm C_l^{HI}(y,z=1.25)$ vs $N_l^{HI-SKA}(y,z=1.25)/\sqrt{f_{sky}l} [mK^2]$')

#plt.legend(('HI density power spectrum','HIRAX noise'))
plt.show()
'''

def cumulative_SNR(SNR_2D_binned):
    sum_each_kperp=np.array([])
    SNR_y,SNR_x=SNR_2D_binned.shape
    for i in range(SNR_x):
        sum_one_kperp=np.sum(SNR_2D_binned[:,i])
        sum_each_kperp=np.append(sum_each_kperp,sum_one_kperp)
    return sum_each_kperp


S_area=15000
SNR=Bispec_SNR
kperp_arr,kpar_arr,SNR_arr=SNR_binned(z,delta_z,Noise,SNR,S_area,Dsep=7.)
ell=kperp_arr*uf.chi(z)


#print (np.cumsum(cum_SNR(SNR_arr)))

pylab.pcolormesh(kperp_arr,kpar_arr,SNR_arr) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
pylab.xlim([np.min(kperp_arr),np.max(kperp_arr)]) ; pylab.ylim([np.min(kpar_arr),np.max(kpar_arr)])
plt.xlabel(r'$k_\perp$',fontsize=12); plt.ylabel(r'$k_\parallel$',fontsize=12); plt.title('Pixel SN for bispectrum', x=1.13, y=1.05)
pylab.show()

plt.plot(kperp_arr,np.cumsum(cumulative_SNR(SNR_arr)))
plt.ylabel('Cumulative S/N')
plt.xlabel(r'$k_\perp$')
plt.show()
'''

plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,z,y=72.),'b')
plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,z,572.),'g')
plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,z,1072.),'r')
plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,z,1572.),'m')
plt.loglog(ell,1.e-6*Integrand_doppler_21cm(ell,z,2072.),'k')


plt.imshow((ell,SNR(ell,1,y)), cmap='hot', interpolation='nearest')

def SNR_integrand_mat(ell,z_i,y,S):
    kperp=ell/chi(z_i)
    kpar=y/r(z_i)
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    integrand_arr=np.array([])
    for i in ell:
        for j in y:
            cl_21=S(i,z_i,j)
            integrand=i*(cl_21)**2/(cl_21+Hirax_noise_z_1_interp(i))**2
            integrand_arr=np.append(integrand_arr,integrand)
    integrand_mat=np.reshape(integrand_arr,(n,n))
    return np.sqrt(const*integrand_mat),kperp,kpar


kperp=SNR_integrand_mat(ell,1,y,Cl_21_func_of_y)[1]
kpar=SNR_integrand_mat(ell,1,y,Cl_21_func_of_y)[2]
pylab.pcolormesh(kperp,kpar,SNR_integrand_mat(ell,1,y,Cl_21_func_of_y)[0]) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel(r'$k_{\perp}$')
plt.ylabel(r'$k_{\parallel}$')
plt.title('Differential S/N')
plt.show()

def SNR_tot(ell,z_i,y):
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    Ell,Y=np.meshgrid(ell,y)
    integrand=Ell*(Cl_21_func_of_y(Ell,z_i,Y))**2/(Cl_21_func_of_y(Ell,z_i,Y)+Hirax_noise_z_1_interp(Ell))**2
    #integral=sp.integrate.cumtrapz(integrand,ell,initial=0)
    integral=sp.integrate.trapz(sp.integrate.trapz(integrand,ell,axis=0),y,axis=0)
    integral=const*integral
    return np.sqrt(integral)

#print (SNR_tot(ell,1,y))



def SNR(ell,z_i,y,S):
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    #snr_mat=np.zeros((n-1,n-1))
    integral_tot=np.zeros((n-1,n-1))
    n_new=10
    for i in range(n-1):
        ell_ind=np.linspace(ell[i],ell[i+1],n_new)
        for j in range(n-1):
            y_ind=np.linspace(y[j],y[j+1],n_new)
            Ell,Y=np.meshgrid(ell_ind,y_ind)
            cl_21 = S(ell_ind,y_ind)#S(Ell,z_i,Y)
            nl_21=Hirax_noise_z_1_interp(Ell)
            #nl_21 = np.reshape(HiraxNoise(Ell,6,7,1),(n_new,n_new))
            integrand=Ell*(cl_21**2.)/(cl_21+nl_21)**2
            integral=sp.integrate.trapz(sp.integrate.trapz(integrand,y_ind,axis=0),ell_ind,axis=0)
            integral_tot[i][j]=integral
            integral_tot=np.sqrt(const*integral_tot)
    #integral_tot=np.flip(integral_tot,0)
    return integral_tot

#print (SNR(ell,1,y,Cl_21_func_of_y))

#pylab.pcolormesh(SNR(ell,1,y,Cl_21_2d_interp)) ;  cbar=plt.colorbar()
#plt.show()
Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
fsky=Sarea/4./np.pi
delta_nu_dimless=400./1420.
const=2.*fsky/(2*np.pi)**2*Sarea*delta_nu_dimless
integrand=Ell*Cl_21_func_of_y(Ell,1,Y)**2/(Cl_21_func_of_y(Ell,1,Y)+Hirax_noise_z_1_interp(Ell))**2
integral=np.trapz(np.trapz(integrand,ell,axis=0),y,axis=0)
#print (np.sqrt(const*integral))

def SNR_area(ell,z_i,y,S):
    #n=100
    #kperp=np.linspace(0.01,0.2,n)
    #kpar=kperp
    kperp=ell/chi(z_i)
    kpar=y/r(z_i)
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2*np.pi)**2*Sarea*delta_nu_dimless
    snr_mat=np.zeros((n-1,n-1))
    for i in range(n-1):
        nl_21 = Hirax_noise_z_1_interp(ell[i+1])
        #ell_ind=np.linspace(ell[i],ell[i+1],n_new)
        for j in range(n-1):
            #y_ind=np.linspace(y[j],y[j+1],n_new)
            #Ell,Y=np.meshgrid(ell_ind,y_ind)
            area=(ell[i+1]-ell[i])*(y[j+1]-y[j])
            #print (area)
            cl_21 = S(ell[i+1],z_i,y[j+1])
            integrand=ell[i+1]*(cl_21**2.)/(cl_21+nl_21)**2
            snr_mat[i][j]=integrand*area
            #integral=sp.integrate.trapz(sp.integrate.trapz(integrand,ell_ind,axis=0),y_ind,axis=0)
            #integral_tot=np.append(integral_tot,integral)
            #integral_tot=np.sqrt(const*integral_tot)
    #integral_tot=np.flip(integral_tot,0)
    return np.sqrt(const*snr_mat),kperp,kpar

SNR_area_21_den=SNR_area(ell,1,y,Cl_21_func_of_y)[0]

N, M = SNR_area_21_den.shape
div=10
assert N % div == 0
assert M % div == 0
A1 = np.zeros((N//div, M//div))
for i in range(N//div):
    for j in range(M//div):
         A1[i,j] = np.mean(SNR_area_21_den[2*i:2*i+2, 2*j:2*j+2])

#print (np.sum(SNR_area(ell,1,y,Cl_21_func_of_y)[0]))

kperp=SNR_area(ell,1.,y,Cl_21_func_of_y)[1]#[::div]
kpar=SNR_area(ell,1.,y,Cl_21_func_of_y)[2]#[::div]
#pylab.pcolormesh(kperp,kpar,SNR_area_21_den) ;  cbar=plt.colorbar()
#plt.show()

def SNR_bispec_integrand(ell,z_i,y,delta_z):
    kperp=ell/chi(z_i)
    kpar=y/r(z_i)
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    integrand_arr=np.array([])
    for i in ell:
        for j in y:
            S_bispec = crosscorr_squeezedlim(i,z_i,j,delta_z)
            N_ksz=cmb_spec(i)+cmb_noise(i)
            S_ksz=interp_OV_full_signal(i)
            S_21_den=Cl_21_func_of_y(i,z_i,j)
            S_21_vel=Integrand_doppler_21cm(i,z_i,j)
            N_21=Hirax_noise_z_1_interp(i)
            sigma=(S_ksz+N_ksz)*(2.*N_21+S_21_den+S_21_vel)+3.*S_bispec**2
            #y_ind=np.linspace(y[j],y[j+1],n_new)
            #Ell,Y=np.meshgrid(ell_ind,y_ind)
            integrand=i*(S_bispec**2.)/(sigma)**2
            integrand_arr=np.append(integrand_arr,integrand)
    integrand_mat=np.reshape(integrand_arr,(n,n))
    return np.sqrt(const*integrand_mat),kperp,kpar


def SNR_bispectrum(ell,z_i,y,delta_z):
    kperp=ell/chi(z_i)
    kpar=y/r(z_i)
    Sarea=15000.*(np.pi/180.)**2 #converting from square degrees to square radians
    fsky=Sarea/4./np.pi
    delta_nu_dimless=400./1420.
    const=2.*fsky/(2.*np.pi)**2*Sarea*delta_nu_dimless
    snr_mat=np.zeros((n-1,n-1))
    n_new=100
    for i in range(n-1):
        #ell_ind=np.linspace(ell[i],ell[i+1],n_new)
        for j in range(n-1):
            S_bispec = crosscorr_squeezedlim(ell[i+1],z_i,y[j+1],delta_z)
            N_ksz=cmb_spec(ell[i+1])+cmb_noise(ell[i+1])
            S_ksz=interp_OV_full_signal(ell[i+1])
            S_21_den=Cl_21_func_of_y(ell[i+1],z_i,y[j+1])
            S_21_vel=Integrand_doppler_21cm(ell[i+1],z_i,y[j+1])
            N_21=Hirax_noise_z_1_interp(ell[i+1])
            sigma=(S_ksz+N_ksz)*(2.*N_21+S_21_den+S_21_vel)+3.*S_bispec**2
            #y_ind=np.linspace(y[j],y[j+1],n_new)
            #Ell,Y=np.meshgrid(ell_ind,y_ind)
            area=(ell[i+1]-ell[i])*(y[j+1]-y[j])
            #print (area)

            integrand=ell[i+1]*(S_bispec**2.)/(sigma)**2
            snr_mat[i][j]=integrand*area
            #integral=sp.integrate.trapz(sp.integrate.trapz(integrand,ell_ind,axis=0),y_ind,axis=0)
            #integral_tot=np.append(integral_tot,integral)
            #integral_tot=np.sqrt(const*integral_tot)
    #integral_tot=np.flip(integral_tot,0)
    return np.sqrt(const*snr_mat),kperp,kpar

kperp=SNR_bispectrum(ell,1.,y,0.3)[1]
kpar=SNR_bispectrum(ell,1.,y,0.3)[2]
pylab.pcolormesh(kperp,kpar,SNR_bispec_integrand(ell,1.,y,0.3)[0]) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel(r'$k_{\perp}$')
plt.ylabel(r'$k_{\parallel}$')
plt.title('Differential S/N')
plt.show()

#print (np.sum(SNR_area(ell,1,y,Integrand_doppler_21cm)[0]),'sum_snr_area')
kperp=SNR_area(ell,1,y,Cl_21_func_of_y)[1]
kpar=SNR_area(ell,1,y,Cl_21_func_of_y)[2]
pylab.pcolormesh(kperp,kpar,SNR_area(ell,1,y,Cl_21_func_of_y)[0]) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel(r'$k_{\perp}$')
plt.ylabel(r'$k_{\parallel}$')
plt.title('S/N')
plt.show()


#print (np.sum(SNR(ell,1,y)),'sum')



#plt.ylim(1e-20,1e-7)
#plt.xlim(100,5000)
#plt.show()

#SNR=Cl_21(1,ell)/HiraxNoise(ell,6,7,1)
#print (HiraxNoise(ell,6.,20.,2.))
#plt.ylim(1e-10,1)
#plt.ylim(1,100)


'''
