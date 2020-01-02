import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import useful_functions as uf
import density as den
import constants as cc
from scipy.interpolate import interp1d
import pylab
'''changed the crosscorr_squeezedlim function to return a single ell i.e. took out the for loop'''

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
n_points=50
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
#get matter power spec data
mu_e=1.14
m_p=cc.m_p_g #in grams
rho_g0=cosmo['omega_b_0']*rho_c
#plt.loglog(uf.kabs,Mps_interpf(uf.kabs))
#plt.show()
r=uf.r
chi=uf.chi

kperp_min=1.e-3
kpar_min=1.e-3
delta_kperp=.03
delta_kpar=.03
kperp_arr=np.linspace(kperp_min,0.2,n_points)
kpar_arr=np.linspace(kpar_min,kpar_min+delta_kpar*n_points,n_points)
ell=kperp_arr*chi(1.26)
y=kpar_arr*r(1.26)

ell=np.geomspace(1.,1.e4,n_points)
kperp_arr=ell/chi(1.26)
kpar_arr=np.geomspace(1.e-4,.15,n_points)
y=kpar_arr*r(1.26)

def ionized_elec(Yp,N_He):
    x=(1-Yp*(1-N_He/4))/(1-Yp/2)
    return x
x=ionized_elec(0.24,0)



def crosscorr_integral_y(ell,z_i,delta_z): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    n=n_points #number of points over which to integrate
    #y=np.geomspace(1.,3000.,n)
    #Kpar=y/uf.r(z)
    Kpar=np.geomspace(1.e-6,.15,n)
    U=np.linspace(-.9999,.9999,n)
    Kp=np.geomspace(1.e-6,1.,n)
    Z_min=z_i-delta_z
    Z_max=z_i+delta_z
    z=np.geomspace(Z_min,Z_max,n)
    u,kp,kpar=np.meshgrid(U,Kp,Kpar)
    T_mean_zi=uf.T_mean(z_i)
    chi_zi=chi(z_i)
    chi_z=chi(z)
    f_zi=uf.f(z_i)
    f_z=uf.f(z)
    D_zi=uf.D_1(z_i)
    r_zi=uf.r(z_i)
    D_z=uf.D_1(z)
    H_zi=uf.H(z_i)
    tau_z=uf.tau_inst(z)
    const=1.e6/(8.*np.pi**3)*T_rad*T_mean_zi**2/cc.c_light_Mpc_s*f_zi*D_zi**2*H_zi/(chi_zi**2)/(1.+z_i)*x*(sigma_T*rho_g0/(mu_e*m_p))
    #Cl=np.array([])
    kp_perp=kp*np.sqrt(1-u**2)
    kp_par=kp*u
    k_perp=ell/chi_zi
    k=np.sqrt(k_perp**2+kpar**2)
    rsd=1.+f_zi*kpar**2/k**2
    zeta=(kpar/k*u+k_perp/k*np.sqrt(1-u**2))
    k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
    K=np.sqrt(k**2+kp**2-2*k_dot_kp)
    #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
    theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
    theta_kp=u
    #theta_K=np.where(theta_K > 0, theta_K, 0)
    I=theta_kp*(theta_kp/kp**2+theta_K/K/kp)
    z_dep_integ=sp.integrate.trapz(f_z*D_z**2*(1+z)*np.exp(-tau_z),z)
    integrand=Mps_interpf(kp)*Mps_interpf(K)*rsd**2*kp**2*I  #+theta_K/K/kp)#-mu*kp*np.gradient(Mps_interpf(k),axis=0))
    integral_sing=const*z_dep_integ*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand,U,axis=0),Kp,axis=0),Kpar,axis=0)
    #Cl=np.append(Cl,integral)
    return integral_sing

def crosscorr_squeezed_integral_y(ell,z_i,delta_z):
    n=n_points #number of points over which to integrate
    #y=np.geomspace(1.,3000.,n)
    #Kpar=y/uf.r(z)
    Kpar=np.geomspace(1.e-6,.15,n)
    U=np.linspace(-.9999,.9999,n)
    Kp=np.geomspace(1.e-6,.1,n)
    Z_min=z_i-delta_z
    Z_max=z_i+delta_z
    z=np.geomspace(Z_min,Z_max,n)
    u,kp,kpar=np.meshgrid(U,Kp,Kpar)
    T_mean_zi=uf.T_mean(z_i)
    chi_zi=chi(z_i)
    chi_z=chi(z)
    f_zi=uf.f(z_i)
    f_z=uf.f(z)
    D_zi=uf.D_1(z_i)
    r_zi=uf.r(z_i)
    D_z=uf.D_1(z)
    H_zi=uf.H(z_i)
    tau_z=uf.tau_inst(z)
    const=1.e6/(8.*np.pi**3)*T_rad*T_mean_zi**2/cc.c_light_Mpc_s*f_zi*D_zi**2*H_zi/(chi_zi**2)/(1.+z_i)*x*(sigma_T*rho_g0/(mu_e*m_p))
    #Cl=np.array([])
    kp_perp=kp*np.sqrt(1-u**2)
    kp_par=kp*u
    k_perp=ell/chi_zi
    k=np.sqrt(k_perp**2+kpar**2)
    rsd=1.+f_zi*kpar**2/k**2
    zeta=(kpar/k*u+k_perp/k*np.sqrt(1-u**2))
    k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
    K=np.sqrt(k**2+kp**2-2*k_dot_kp)
    #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
    theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
    theta_kp=u
    #theta_K=np.where(theta_K > 0, theta_K, 0)
    I=theta_kp*(theta_kp/kp**2+theta_K/K/kp)
    z_dep_integ=sp.integrate.trapz(f_z*D_z**2*(1+z)*np.exp(-tau_z),z)
    integrand=Mps_interpf(kp)*Mps_interpf(k)*rsd**2*kp**2*I  #+theta_K/K/kp)#-mu*kp*np.gradient(Mps_interpf(k),axis=0))
    integral_sing=const*z_dep_integ*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand,U,axis=0),Kp,axis=0),Kpar,axis=0)
    #Cl=np.append(Cl,integral)
    return integral_sing


def crosscorr_integral_y_all_ell(ell,z_i,delta_z):
    array_bispec=np.array([])
    for i in ell:
        array_bispec=np.append(array_bispec,crosscorr_integral_y(i,z_i,delta_z))
    return array_bispec

def crosscorr_squeezed_integral_y_all_ell(ell,z_i,delta_z):
    array_bispec=np.array([])
    for i in ell:
        array_bispec=np.append(array_bispec,crosscorr_squeezed_integral_y(i,z_i,delta_z))
    return array_bispec

'''
plt.loglog(ell,ell*(ell+1)*crosscorr_integral_y_all_ell(ell,1.26,.0015)/2/np.pi,'r')
plt.loglog(ell,ell*(ell+1)*crosscorr_squeezed_integral_y_all_ell(ell,1.26,.0015)/2/np.pi,'b')
plt.legend(('Full bispectrum','Squeezed bispectrum'))
plt.ylabel(r'$\rm{l(l+1)B_l^{{21-OV}}(z=1.26,\Delta_z=0.003)/2\pi[\mu K^3]}$')
#plt.xlabel(r'$\rm{k_\perp}$')
plt.xlabel('l')
plt.show()
'''



#plt.plot(ell,ell*(ell+1)*crosscorr_integral_y_rec_coords_all_ell(ell,1.26,.0015)/2/np.pi)
#plt.ylabel(r'$\rm{l(l+1)B_l^{{21-OV}}(z=1.26,\Delta_z=0.003)/2\pi[\mu K^3]}$')
#plt.show()

def crosscorr(ell,z_i,y, delta_z): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    n=n_points #number of points over which to integrate
    #y=np.geomspace(1.,3000.,n)
    U=np.linspace(-.9999,.9999,n)
    Kp=np.geomspace(1.e-6,1.,n)
    Z_min=z_i-delta_z
    Z_max=z_i+delta_z
    z=np.geomspace(Z_min,Z_max,n)
    u,kp=np.meshgrid(U,Kp)
    T_mean_zi=uf.T_mean(z_i)
    chi_zi=chi(z_i)
    chi_z=chi(z)
    f_zi=uf.f(z_i)
    f_z=uf.f(z)
    D_zi=uf.D_1(z_i)
    r_zi=uf.r(z_i)
    D_z=uf.D_1(z)
    H_zi=uf.H(z_i)
    tau_z=uf.tau_inst(z)
    kpar=y/uf.r(z)
    const=1.e6/(4.*np.pi**2)*T_rad*T_mean_zi**2/cc.c_light_Mpc_s*f_zi*D_zi**2*H_zi/(chi_zi**2*r_zi)/(1.+z_i)*x*(sigma_T*rho_g0/(mu_e*m_p))
    #Cl=np.array([])
    kp_perp=kp*np.sqrt(1-u**2)
    kp_par=kp*u
    k_perp=ell/chi_zi
    k=np.sqrt(k_perp**2+kpar**2)
    rsd=1.+f_zi*kpar**2/k**2
    zeta=(kpar/k*u+k_perp/k*np.sqrt(1-u**2))
    k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
    K=np.sqrt(k**2+kp**2-2*k_dot_kp)
    #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
    theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
    #print (theta_K.min(),theta_K.max())
    theta_kp=u
    #theta_K=np.where(theta_K > 0, theta_K, 0)
    I=theta_kp*(theta_kp/kp**2+theta_K/K/kp)
    z_integral=sp.integrate.trapz(f_z*D_z**2*(1+z)*np.exp(-tau_z),z)
    integrand_1=z_integral*Mps_interpf(kp)*Mps_interpf(K)*rsd**2*theta_kp**2
  #+theta_K/K/kp)#-mu*kp*np.gradient(Mps_interpf(k),axis=0))
    integrand_2=z_integral*Mps_interpf(kp)*Mps_interpf(K)*rsd**2*kp**2*theta_kp*theta_K/kp/K

    integral_sing_1=const*sp.integrate.trapz(sp.integrate.trapz(integrand_1,U,axis=0),Kp,axis=0)
    integral_sing_2=const*sp.integrate.trapz(sp.integrate.trapz(integrand_2,U,axis=0),Kp,axis=0)
    #Cl=np.append(Cl,integral)
    return integral_sing_1+integral_sing_2



def crosscorr_all_ell(ell,z_i,y, delta_z):
    array_bispec=np.array([])
    for i in ell:
        array_bispec=np.append(array_bispec,crosscorr(i,z_i,y, delta_z))
    return array_bispec

def crosscorr_all_ell_all_y(ell,z_i,y, delta_z):
    array_bispec=np.zeros((n_points,n_points))
    for i in range(len(y)):
        for j in range(len(ell)):
            array_bispec[i,j]=crosscorr(ell[j],z_i,y[i], delta_z)
    return array_bispec

def crosscorr_squeezedlim(ell,z_i,y, delta_z): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    n=n_points #number of points over which to integrate
    #y=np.geomspace(1.,3000.,n)
    U=np.linspace(-.9999,.9999,n)
    Kp=np.geomspace(1.e-6,.1,n)
    Z_min=z_i-delta_z
    Z_max=z_i+delta_z
    z=np.geomspace(Z_min,Z_max,n)
    u,kp=np.meshgrid(U,Kp)
    T_mean_zi=uf.T_mean(z_i)
    chi_zi=chi(z_i)
    chi_z=chi(z)
    f_zi=uf.f(z_i)
    f_z=uf.f(z)
    D_zi=uf.D_1(z_i)
    r_zi=uf.r(z_i)
    D_z=uf.D_1(z)
    H_zi=uf.H(z_i)
    tau_z=uf.tau_inst(z)
    kpar=y/uf.r(z)
    const=1.e6/(4.*np.pi**2)*T_rad*T_mean_zi**2/cc.c_light_Mpc_s*f_zi*D_zi**2*H_zi/(chi_zi**2*r_zi)/(1.+z_i)*x*(sigma_T*rho_g0/(mu_e*m_p))
    #Cl=np.array([])
    kp_perp=kp*np.sqrt(1-u**2)
    kp_par=kp*u
    k_perp=ell/chi_zi
    k=np.sqrt(k_perp**2+kpar**2)
    rsd=1.+f_zi*kpar**2/k**2
    zeta=(kpar/k*u+k_perp/k*np.sqrt(1-u**2))
    k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
    K=np.sqrt(k**2+kp**2-2*k_dot_kp)
    #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
    theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
    #print (theta_K.min(),theta_K.max())
    theta_kp=u
    #theta_K=np.where(theta_K > 0, theta_K, 0)
    I=theta_kp*(theta_kp/kp**2+theta_K/K/kp)
    z_integral=sp.integrate.trapz(f_z*D_z**2*(1+z)*np.exp(-tau_z),z)
    integrand_1=z_integral*Mps_interpf(kp)*Mps_interpf(K)*rsd**2*theta_kp**2
  #+theta_K/K/kp)#-mu*kp*np.gradient(Mps_interpf(k),axis=0))
    integrand_2=z_integral*Mps_interpf(kp)*Mps_interpf(k)*rsd**2*kp**2*theta_kp*theta_K/kp/K

    integral_sing_1=const*sp.integrate.trapz(sp.integrate.trapz(integrand_1,U,axis=0),Kp,axis=0)
    integral_sing_2=const*sp.integrate.trapz(sp.integrate.trapz(integrand_2,U,axis=0),Kp,axis=0)
    #Cl=np.append(Cl,integral)
    return integral_sing_1+integral_sing_2


'''
pylab.pcolormesh(kperp_arr,kpar_arr,crosscorr_all_ell_all_y(ell,1.26,y, .0015)) ;  cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=12)
plt.tick_params(axis='both', which='major', labelsize=12)
pylab.xlim([np.min(kperp_arr),np.max(kperp_arr)]) ; pylab.ylim([np.min(kpar_arr),np.max(kpar_arr)])
plt.xlabel(r'$k_\perp$',fontsize=12); plt.ylabel(r'$k_\parallel$',fontsize=12); plt.title(r'$B_{l,y}^{{21-OV}}(z=1.26,\Delta_z=0.003)[\mu K^3]}$')
plt.xscale('log')
plt.yscale('log')
#plt.show()
'''
def crosscorr_integral_y_rec_coords(ell,z_i,delta_z): #This is not working out-don't try this for now. Just leave it.
    n=n_points #number of points over which to integrate
    Kp_par=np.geomspace(1.e-6,.15,n)
    Kp_perp=np.geomspace(1.e-6,1.,n)
    Z_min=z_i-delta_z
    Z_max=z_i+delta_z
    z=np.geomspace(Z_min,Z_max,n)
    Kpar=np.geomspace(1.e-6,.15,n)
    kp_perp,kp_par,kpar=np.meshgrid(Kp_perp,Kp_par,Kpar)
    chi_z=chi(z)
    T_mean_zi=uf.T_mean(z_i)
    chi_zi=chi(z_i)
    f_zi=uf.f(z_i)
    f_z=uf.f(z)
    D_zi=uf.D_1(z_i)
    r_zi=uf.r(z_i)
    D_z=uf.D_1(z)
    H_zi=uf.H(z_i)
    tau_z=uf.tau_inst(z)
    kpar=y/uf.r(z)
    const=1.e6/(8.*np.pi**3)*T_rad*T_mean_zi**2/cc.c_light_Mpc_s*f_zi*D_zi**2*H_zi/(chi_zi**2)/(1.+z_i)*x*(sigma_T*rho_g0/(mu_e*m_p))
    kperp=ell/chi_zi
    k=np.sqrt(kperp**2+kpar**2)
    kp=np.sqrt(kp_perp**2+kp_par**2)
    cos_theta=kp_par/kp #cos_theta=u, theta is azimuthal angle between k' and z (line of sight) axis
    sin_theta=kp_perp/kp
    sin_gamma=kpar/k #gamma is measured between k and xy plane, i.e. elevation of k
    cos_gamma=kperp/k
    zeta=sin_gamma*cos_theta+cos_gamma*sin_theta
    k_dot_kp=kperp*kp_perp+kpar*kp_par*zeta
    K=np.sqrt(k**2+kp**2-2*k_dot_kp)
    theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*kperp*np.sqrt(1-zeta**2)/k/K
    theta_kp=cos_theta
    rsd=1.+f_zi*kpar**2/k**2
    I=theta_kp*(theta_kp/kp**2+theta_K/K/kp)
    z_dep_integ=sp.integrate.trapz(f_z*D_z**2*(1+z)*np.exp(-tau_z),z)
    integrand_1=rsd**2*Mps_interpf(kp)*Mps_interpf(K)*theta_kp**2/kp**2*kp_perp
    integrand_2=rsd**2*Mps_interpf(kp)*Mps_interpf(K)*theta_kp*theta_K/kp/K*kp_perp
    integral_1=const*z_dep_integ*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand_1,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0)
    integral_2=const*z_dep_integ*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand_2,Kp_perp,axis=0),Kp_par,axis=0),Kpar,axis=0)
    return integral_1 +integral_2


def crosscorr_integral_y_rec_coords_all_ell(ell,z_i,delta_z):
    array_bispec=np.array([])
    for i in ell:
        array_bispec=np.append(array_bispec,crosscorr_integral_y_rec_coords(i,z_i,delta_z))
    return array_bispec


def crosscorr_rec_coords(ell,z_i,y,delta_z): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    n=n_points #number of points over which to integrate
    Kp_par=np.geomspace(1.e-6,.15,n)
    Kp_perp=np.geomspace(1.e-6,1.,n)
    Z_min=z_i-delta_z
    Z_max=z_i+delta_z
    Z=np.geomspace(Z_min,Z_max,n)
    kp_perp,kp_par,z=np.meshgrid(Kp_perp,Kp_par,Z)
    chi_z=chi(z)
    T_mean_zi=uf.T_mean(z_i)
    chi_zi=chi(z_i)
    chi_z=chi(z)
    f_zi=uf.f(z_i)
    f_z=uf.f(z)
    D_zi=uf.D_1(z_i)
    r_zi=uf.r(z_i)
    D_z=uf.D_1(z)
    H_zi=uf.H(z_i)
    tau_z=uf.tau_inst(z)
    kpar=y/uf.r(z)
    const=1.e6/(4.*np.pi**2)*T_rad*T_mean_zi**2/cc.c_light_Mpc_s*f_zi*D_zi**2*H_zi/(chi_zi**2*r_zi)/(1.+z_i)*x*(sigma_T*rho_g0/(mu_e*m_p))
    kperp=ell/chi_z
    k=np.sqrt(kperp**2+kpar**2)
    kp=np.sqrt(kp_perp**2+kp_par**2)
    cos_theta=kp_par/kp #cos_theta=u, theta is azimuthal angle between k' and z (line of sight) axis
    sin_theta=kp_perp/kp
    sin_gamma=kpar/k #gamma is measured between k and xy plane, i.e. elevation of k
    cos_gamma=kperp/k
    zeta=sin_gamma*cos_theta+cos_gamma*sin_theta
    k_dot_kp=kperp*kp_perp+kpar*kp_par*zeta
    K=np.sqrt(k**2+kp**2-2*k_dot_kp)
    theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*kperp*np.sqrt(1-zeta**2)/k/K
    theta_kp=cos_theta
    rsd=1.+f_zi*kpar**2/k**2
    I=theta_kp*(theta_kp/kp**2+theta_K/K/kp)
    integrand_1=f_z*D_z**2*(1+z)*np.exp(-tau_z)*rsd**2*Mps_interpf(kp)*Mps_interpf(K)*theta_kp**2/kp**2
    integrand_2=f_z*D_z**2*(1+z)*np.exp(-tau_z)*rsd**2*Mps_interpf(kp)*Mps_interpf(K)*theta_kp*theta_K/kp/K
    integral_1=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand_1,Kp_perp,axis=0),Kp_par,axis=0),Z,axis=0)
    integral_2=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand_2,Kp_perp,axis=0),Kp_par,axis=0),Z,axis=0)
    return integral_1 +integral_2


def crosscorr_rec_coords_all_ell(ell,z_i,y, delta_z):
    array_bispec=np.array([])
    for i in ell:
        array_bispec=np.append(array_bispec,crosscorr_rec_coords(i,z_i,y, delta_z))
    return array_bispec


def F_crosscorr(ell,z,y): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    n=n_points #number of points over which to integrate
    #y=np.geomspace(1.,3000.,n)
    U=np.linspace(-.9999,.9999,n)
    Kp=np.geomspace(1.e-10,.1,n)

    u,kp=np.meshgrid(U,Kp)
    chi_z=chi(z)
    kpar=y/uf.r(z)
    #Cl=np.array([])
    kp_perp=kp*np.sqrt(1-u**2)
    kp_par=kp*u
    integral_1=np.array([])
    integral_2=np.array([])
    for i in ell:
        k_perp=i/chi_z
        k=np.sqrt(k_perp**2+kpar**2)
        zeta=(kpar/k*u+k_perp/k*np.sqrt(1-u**2))
        k_dot_kp=k_perp*kp_perp+kpar*kp_par*zeta
        K=np.sqrt(k**2+kp**2-2*k_dot_kp)
        #theta_kp=kpar*zeta/k+k_perp*np.sqrt(np.abs(1-zeta**2))/k
        theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*k_perp*np.sqrt(1-zeta**2)/k/K
        #print (theta_K.min(),theta_K.max())
        theta_kp=u
        #theta_K=np.where(theta_K > 0, theta_K, 0)
        I=theta_kp*(theta_kp/kp**2+theta_K/K/kp)
        integrand_1=Mps_interpf(kp)*Mps_interpf(K)*theta_kp**2
      #+theta_K/K/kp)#-mu*kp*np.gradient(Mps_interpf(k),axis=0))
        integrand_2=Mps_interpf(kp)*Mps_interpf(K)*kp**2*theta_kp*theta_K/kp/K

        integral_sing_1=sp.integrate.trapz(sp.integrate.trapz(integrand_1,U,axis=0),Kp,axis=0)
        integral_sing_2=sp.integrate.trapz(sp.integrate.trapz(integrand_2,U,axis=0),Kp,axis=0)
        integral_1=np.append(integral_1,integral_sing_1)
        integral_2=np.append(integral_2, integral_sing_2)
    #Cl=np.append(Cl,integral)
    return integral_1, integral_2

def F_crosscorr_kp_rect_coords(ell,z,y): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    n=n_points #number of points over which to integrate
    Kp_par=np.geomspace(1.e-6,.15,n)
    Kp_perp=np.geomspace(1.e-6,1.,n)
    kp_perp,kp_par=np.meshgrid(Kp_perp,Kp_par)
    chi_z=chi(z)
    kpar=y/uf.r(z)
    integral_1=np.array([])
    integral_2=np.array([])
    for i in ell:
        kperp=i/chi_z
        k=np.sqrt(kperp**2+kpar**2)
        kp=np.sqrt(kp_perp**2+kp_par**2)
        cos_theta=kp_par/kp #cos_theta=u, theta is azimuthal angle between k' and z (line of sight) axis
        sin_theta=kp_perp/kp
        sin_gamma=kpar/k #gamma is measured between k and xy plane, i.e. elevation of k
        cos_gamma=kperp/k
        zeta=sin_gamma*cos_theta+cos_gamma*sin_theta
        k_dot_kp=kperp*kp_perp+kpar*kp_par*zeta
        K=np.sqrt(k**2+kp**2-2*k_dot_kp)
        theta_K=kpar/K/k**2*(k**2-k_dot_kp)-kp*kperp*np.sqrt(1-zeta**2)/k/K
        theta_kp=cos_theta
        I=theta_kp*(theta_kp/kp**2+theta_K/K/kp)
        integrand_1=Mps_interpf(kp)*Mps_interpf(K)*theta_kp**2/kp**2
        integrand_2=Mps_interpf(kp)*Mps_interpf(K)*theta_kp*theta_K/kp/K
        integral_sing_1=sp.integrate.trapz(sp.integrate.trapz(integrand_1,Kp_perp,axis=0),Kp_par,axis=0)
        integral_sing_2=sp.integrate.trapz(sp.integrate.trapz(integrand_2,Kp_perp,axis=0),Kp_par,axis=0)
        integral_1=np.append(integral_1,integral_sing_1)
        integral_2=np.append(integral_2, integral_sing_2)
    return integral_1, integral_2


'''
plt.loglog(ell,ell*(ell+1)*crosscorr_integral_y_all_ell(ell,1.26,.05)/2/np.pi,'r')
plt.loglog(ell,ell*(ell+1)*crosscorr_squeezedlim_integral_y_all_ell(ell,1.26,.05)/2/np.pi,'b')
plt.ylabel(r'$\rm{l(l+1)C_l^{21-OV}}(z)/(2\pi)[\mu K^3]$')
plt.xlabel('l')
plt.legend(('Full signal','Squeezed limit'))
plt.show()
'''


#plt.plot(ell,ell*(ell+1)*crosscorr_all_ell(ell,1.26,y=1000.,delta_z=.003))
#plt.ylabel(r'$\rm l(l+1)B_l^{21-kSZ}(z_i=1.26)$')
#plt.xlabel('l')
#plt.show()

'''
plt.loglog(ell,ell*(ell+1)*crosscorr_squeezedlim_all_ell(ell,1.,277.,0.3)/2/np.pi,'r')
plt.loglog(ell,ell*(ell+1)*crosscorr_squeezedlim_all_ell(ell,1.,750.,0.3)/2/np.pi,'b')
plt.loglog(ell,ell*(ell+1)*crosscorr_squeezedlim_all_ell(ell,1.,2032.,0.3)/2/np.pi,'g')
plt.xlabel('l')
plt.legend(('y=277','y=750','y=2032'))
plt.ylabel(r'$\rm{l(l+1)B_l(y,z=1,\Delta_z=0.3)^{{21-OV}}/2\pi[\mu K^3]}$')
plt.show()


plt.loglog(ell,ell*(ell+1)*crosscorr_squeezedlim_integral_y_all_ell(ell,1.,0.3)/2/np.pi)
plt.xlabel('l')
plt.ylabel(r'$\rm{l(l+1)B_l(z=1,\Delta_z=0.3)^{{21-OV}}/2\pi[\mu K^3]}$')
plt.show()
'''

'''def crosscorr_squeezedlim(ell,z,y,delta_z): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    n=100 #number of points over which to integrate
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp=np.geomspace(1.e-4,.1,n)
    mu,kp=np.meshgrid(Mu,Kp)
    z_i=z
    T_mean_zi=uf.T_mean(z_i)
    chi_zi=chi(z_i)
    chi_z=chi(z)
    f_zi=uf.f(z_i)
    f_z=uf.f(z)
    D_zi=uf.D_1(z_i)
    r_zi=uf.r(z_i)
    D_z=uf.D_1(z)
    H_zi=uf.H(z_i)
    tau_z=uf.tau_inst(z)
    const=1.e6/(8.*np.pi**2)*T_rad*T_mean_zi**2/cc.c_light_Mpc_s*f_zi*D_zi**2*H_zi/(chi_zi**2*r_zi)/(1.+z_i)*x*(sigma_T*rho_g0/(mu_e*m_p))*(2*delta_z)*f_z*D_z**2*(1+z)*np.exp(-tau_z)
    #Cl=np.array([])
    kpar=y/r_zi
    k_perp=ell/chi_zi
    k=np.sqrt(k_perp**2+kpar**2)
    rsd=1.+f_zi*kpar**2/k**2
    zeta=(kpar/k*mu+k_perp/k*np.sqrt(1-mu**2))
    theta_kp=kpar*zeta/k+k_perp*np.sqrt(1.-zeta**2)/k
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*zeta))
    integrand=Mps_interpf(kp)*rsd**2*theta_kp**2*(Mps_interpf(K))#-mu*kp*np.gradient(Mps_interpf(k),axis=0))
    integral_sing=const*sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0)
    #Cl=np.append(Cl,integral)
    return integral_sing


def crosscorr_squeezedlim_all_ell(ell,z,y,delta_z):
    array_bispec=np.array([])
    for i in ell:
        array_bispec=np.append(array_bispec,crosscorr_squeezedlim(i,z,y,delta_z))
    return array_bispec

def crosscorr_squeezedlim_integral_y(ell,z_i,delta_z): #with the assumption that zi=z so no cos factor, and we have dz=redshift bin width=2*delta_z defined above
    n=50 #number of points over which to integrate
    y=np.geomspace(1.,3000.,n)
    Kpar=y/uf.r(z_i)
    Mu=np.linspace(-0.9999,0.9999,n)
    Kp=np.geomspace(1.e-10,.1,n)
    Z_min=z_i-delta_z
    Z_max=z_i+delta_z
    Z=np.geomspace(Z_min,Z_max,n)
    mu,kp,z,kpar=np.meshgrid(Mu,Kp,Z,Kpar)
    T_mean_zi=uf.T_mean(z_i)
    chi_zi=chi(z_i)
    chi_z=chi(z)
    f_zi=uf.f(z_i)
    f_z=uf.f(z)
    D_zi=uf.D_1(z_i)
    r_zi=uf.r(z_i)
    D_z=uf.D_1(z)
    H_zi=uf.H(z_i)
    tau_z=uf.tau_inst(z)
    const=1.e6/(8.*np.pi**2)*T_rad*T_mean_zi**2/cc.c_light_Mpc_s*f_zi*D_zi**2*H_zi/(chi_zi**2)/(1.+z_i)*x*(sigma_T*rho_g0/(mu_e*m_p))
    #Cl=np.array([])
    kpar=y/r_zi
    k_perp=ell/chi_zi
    k=np.sqrt(k_perp**2+kpar**2)
    rsd=1.+f_zi*kpar**2/k**2
    zeta=(kpar/k*mu+k_perp/k*np.sqrt(1-mu**2))
    theta_kp=kpar*zeta/k+k_perp*np.sqrt(1.-zeta**2)/k
    K=np.sqrt(np.abs(k**2+kp**2-2*k*kp*zeta))
    integrand=f_z*D_z**2*(1+z)*np.exp(-tau_z)*Mps_interpf(kp)*rsd**2*theta_kp**2*(Mps_interpf(K))#-mu*kp*np.gradient(Mps_interpf(k),axis=0))
    integral_sing=const*sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(integrand,Mu,axis=0),Kp,axis=0),Z,axis=0),Kpar,axis=0)
    #Cl=np.append(Cl,integral)
    return integral_sing

def crosscorr_squeezedlim_integral_y_all_ell(ell,z_i,delta_z):
    array_bispec=np.array([])
    for i in ell:
        array_bispec=np.append(array_bispec,crosscorr_squeezedlim_integral_y(i,z_i,delta_z))
    return array_bispec
'''
