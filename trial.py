# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:08:59 2018

@author: zahra
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import useful_functions as uf
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
'''
n=10
z=np.linspace(1,4,n)
x=np.linspace(0,2,n)
d=np.linspace(0,6,n)
l=np.linspace(0,5,n)
def A(x,z):
    A=np.array([])
    for i in x:
        print (i)
        integrand=i*l/(z)
        print (integrand)
        integral=sp.integrate.trapz(integrand,l)
        print (integral)
        A_res=np.append(A,integral)
        #print (A_res)
    #print (A_res)
    return A_res

def B(d):
    B=np.array([])
    for i in d:
        print (i)
        print (A(i/z,z),'A(i/z)')
        integral=sp.integrate.trapz(A(i/(z),z)*(z)**3,z)
        B_res=np.append(B,integral)
        print (B_res)
    return B_res

def C(d):
    A1,A2=np.meshgrid(l,z)
    C=np.array([])
    for i in d:
        integrand=A1*A2*i
        integral=sp.integrate.trapz(sp.integrate.trapz(integrand,l,axis=0),z,axis=0) 
        print (integral) 
        C_res=np.append(C,integral)
    return C_res
    
#x=np.linspace(0,100,n)
#y=2*x
#np.savetxt('test3.out',(x,y))


x,y=np.genfromtxt('C:\\Users\\zahra\\.spyder-py3\\test.out')
print (x[5])
print (y[5])
interp = interp1d(x, y, bounds_error=False,fill_value=0.)
x_new=np.linspace(0,5,n)
plt.plot(x_new,interp(x_new))
plt.show()




z=np.linspace(1e-4,10,n)
kpar=np.linspace(1e-2,10,n)

def integral():
    func=np.array([])
    for i in kpar:
        z1,z2=np.meshgrid(z,z)
        integral=[sp.integrate.trapz(sp.integrate.trapz(np.cos(i*(uf.chi(z1)-uf.chi(z2))),z,axis=0),z,axis=0)]
        func=np.append(func,integral)
    return func
#print (integral())
#ax.plot3D(x, y, c)
plt.plot(kpar,integral())
plt.show()

x=np.linspace(-10,10,10000)
y=np.ones(10000)
y_ft=np.fft.fft(y)/10000
y_ft=np.fft.fftshift(y_ft)
#plt.plot(x,np.real(y_ft))
sp.integrate.trapz(x*y_ft,x)
sp.integrate.trapz((x+1)*y_ft,x)

def f(x):
    array=np.array([])
    for i in x:
        y=i+1
        array=np.append(array,y)
    return array
'''
chi=uf.chi
Mps_interpf=uf.Mps_interpf
n_points=10

x=np.linspace(0,4,n_points)
y=np.linspace(0,3,n_points)
z=np.linspace(0,5,n_points)

X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
function=X+Y+Z

my_interpolating_function = RegularGridInterpolator(points=[x, y, z], values=function)

tst=(1.4,1.5)
xyz=(X,Y,Z)

#print (1.2+1.4+1.5,'simple')

#print (my_interpolating_function(tst),'simple interp')
#print (function,'function')
#print (my_interpolating_function(xyz),'interp function')
#print (sp.integrate.trapz(sp.integrate.trapz(function,x,axis=0),y,axis=0))
#print (sp.integrate.trapz(sp.integrate.trapz(my_interpolating_function(xyz),x,axis=0),y,axis=0))


'''trying the quadrature integration'''

D=np.linspace(0,1,n_points)


def trial_check(d):
    a=np.linspace(-1,1,n_points)
    b=np.linspace(1e-5,1e-1,n_points)
    r=np.linspace(0,1,n_points)
    e=np.linspace(1e-4,1e-2,n_points)
    A,B,R,S,E=np.meshgrid(a,b,r,r,e) 
    array=np.array([]) 
    for i in d:
        integrand=sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz
                            ((i*A+E)/(B**2+i**2/uf.chi(R)**2+E**2-2*B*np.sqrt(i**2/uf.chi(R)**2+E**2)*A)*R*S,a,axis=0),b,axis=0),r,axis=0),r,axis=0),e,axis=0)
        array=np.append(array,integrand)
    return array   

print (trial_check(np.linspace(1e-5,1e-1,10)))

def Integrand(A,B,R1,R2,E,d):
    a,b,r1,r2,e=np.meshgrid(A,B,R1,R2,E) 
    tau=uf.tau_inst
    D=uf.D_1
    f=uf.f
    k_perp=d/chi(r1)
    k=np.sqrt(k_perp**2+e**2)
    K=np.sqrt(k**2+b**2-2*k*b*e)
    integrand=(e*a*b**2/k+k_perp*b**2*np.sqrt(1-a**2)/k)*((e*a+k_perp*np.sqrt(1-a**2))*(k-2*b*a)/b**2/K**2+e/b/K**2)*Mps_interpf(b)*Mps_interpf(K)*(1+r1)*(1+r2)*np.exp(-tau(r1))*np.exp(-tau(r2))*f(r1)*f(r2)*D(r1)**2*D(r2)**2/chi(r1)**2*np.cos(e*(chi(r1)-chi(r2)))
    return integrand


def func(d):
    array=np.array([])
    for i in d:
        int1=lambda b,r1,r2,e,i: sp.integrate.fixed_quad(Integrand,-1,1,args=(b,r1,r2,e,i, ),n=5)[0]
        int2=lambda r1,r2,e,i: sp.integrate.fixed_quad(int1,1e-5,1e-1,args=(r1,r2,e,i, ),n=5)[0]
        int3=lambda r2,e,i: sp.integrate.fixed_quad(int2,1e-4,10,args=(r2,e,i,),n=5)[0]
        int4=lambda e,i: sp.integrate.fixed_quad(int3,1e-4,10,args=(e,i,),n=5)[0]
        int5=sp.integrate.fixed_quad(int4,1e-4,1e-2,args=(i,),n=5)[0]
        array=np.append(array,int5)
    return array
        
print (func(np.linspace(1e-2,1e4,50)))
def first_int(b,r,e):
    return sp.integrate.quadrature(Integrand, 0, 5, args=(b,r,e), maxiter=5,vec_func=False)[0]

def second_int(r,e):
    return sp.integrate.quadrature(first_int, 0, 4, args=(r,e), maxiter=5,vec_func=False)[0]

def third_int(e):
    return sp.integrate.quadrature(second_int,0,1,args=(e),maxiter=5,vec_func=False)[0]

def fourth(D):
    res=sp.integrate.quadrature(lambda e:third_int(e),0,2,maxiter=5,vec_func=False)
    return res

print (fourth(D))

'''tests!'''

def integrand_double_int_d(X,Y):
    x,y=np.meshgrid(X,Y)
    integrand=x*y
    return integrand

def function_double_int():
    integral1=lambda Y: sp.integrate.fixed_quad(integrand_double_int_d,0,2,args=(Y,),n=6)[0]
    integral2=sp.integrate.fixed_quad(integral1,0,2,n=6)[0]
    return integral2

def function_double_int_forloop(Y):
    array=np.array([])
    for i in Y:
        integral1=sp.integrate.fixed_quad(integrand_double_int_d,0, 2, args=(i, ), n=6)[0]
        array=np.append(array,integral1)
    return array

print (function_double_int_forloop(np.linspace(0,1,2)))

def integrand_triple_int(W,X,Y):
    w,x,y=np.meshgrid(W,X,Y)
    return x*y*z

    

W=np.linspace(-0.9999,0.9999,100)
X=np.geomspace(1e-4,10,100)
Y=np.geomspace(1e-4,10,100)
int1=sp.integrate.trapz(integrand_triple_int(W,X,Y),W,axis=0)
print (int1)
print (sp.integrate.trapz(sp.integrate.trapz(int1,X,axis=0),Y,axis=0))

def function_triple_int():
    inside=lambda Y,Z: sp.integrate.quadrature(integrand_triple_int,-0.9999,0.9999,args=(Y,Z,),vec_func=False)[0]
    outside=lambda Z: sp.integrate.quadrature(inside,1e-4,10,args=(Z,),vec_func=False)[0]
    outside_final=sp.integrate.quadrature(outside,1e-4,10,vec_func=False)[0]
    return outside_final

print (function_triple_int())

def func_trip_int_split(Z,A,d):
    array=np.array([])
    for i in d:
        inside1=lambda X,Y,Z,A,i: sp.integrate.fixed_quad(integrand_triple_int,0,1,args=(X,Y,Z,A,i,),n=5)[0]
        inside2=lambda Y,Z,A,i: sp.integrate.fixed_quad(inside1,0,1,args=(Y,Z,A,i,),n=5)[0]
        for j in Z:
            for m in A:
                outside=sp.integrate.fixed_quad(inside2,0,1,args=(i,j,m,),n=5)[0]
                array=np.append(array,outside)
    return array

    
Z=np.linspace(0,5,5)
A=np.linspace(0,3,5)
d=np.linspace(0,2,5)
reshape=np.reshape(func_trip_int_split(Z,A,d),(5,5,5))

print (sp.integrate.trapz(sp.integrate.trapz(reshape,Z,axis=0),A,axis=0))

def integrand_trip_int_d(X,Y,Z,d):
    x,y,z=np.meshgrid(X,Y,Z)
    integrand=x*y*z*d
    return integrand
    

def function_triple_int_forloop(d):
    #Z=np.linspace(0,1,5)
    err=np.array([])
    array=np.array([])
    for i in d:    
        inside=lambda Y,Z,i: sp.integrate.quadrature(integrand_trip_int_d,1e-4,10,args=(Y,Z,i, ),vec_func=False)[0]
        outside=lambda Z,i: sp.integrate.quadrature(inside,-0.9999,0.9999,args=(Z,i, ),vec_func=False)[0]
        outside_final=sp.integrate.quadrature(outside,1e-4,10,args=(i,),vec_func=False)[0]
        error=sp.integrate.quadrature(outside,0,2,args=(i,),vec_func=False)[1]
        err=np.append(err,error)
        array=np.append(array,outside_final)
    return array,err  

print (function_triple_int_forloop(np.linspace(0,1,5))[0])

def function_tripl_int_nquad(Z,d):
    array=np.array([])
    for i in d:
        integral=[sp.integrate.nquad(lambda X,Y: integrand_trip_int_d(X,Y,i),[[1e-4,10],[-0.9999,0.9999]]) for j in Z][0]
        array=np.append(array,integral)
    return array
print (function_tripl_int_nquad(np.linspace(0,1,5)))

'''end of tests'''

def trial4_6dmesh(r,e,d):
    a=np.linspace(0,5,n_points)
    b=np.linspace(0,4,n_points)
    A,B,R,S,E,D=np.meshgrid(a,b,r,r,e,d,indexing='ij')     
    integral=sp.integrate.trapz(sp.integrate.trapz((D*A+E)*B*R*S,a,axis=0),b,axis=0)
    return integral

#print (trial4_6dmesh(d),np.shape(trial4_6dmesh(d)),'trial4_6dmesh')

#print (trial_check(d))


def trial4(r,e,d):
    a=np.linspace(0,5,n_points)
    b=np.linspace(0,4,n_points)
    A,B,R,S=np.meshgrid(a,b,r,r) 
    array=np.array([])
    for i in d:
        for j in e:
            integrand=sp.integrate.trapz(sp.integrate.trapz((i*A+j)*B*R*S,a,axis=0),b,axis=0)
            array=np.append(array,integrand)
    F=np.reshape(array,(n_points,n_points,n_points,n_points))
    F_trans=np.transpose(F)
    return F_trans


R,S,E,D=np.meshgrid(r,r,e,d,indexing='ij') 


#print (np.shape(trial4(d,r,e)))

#print (F,'reshaped for loops')
#np.savez('trial5',F=trial4(d,r,e))
npzfile = np.load('trial5.npz')
#print (npzfile['F'])
#F_new=np.genfromtxt('C:\\Users\\zahra\\.spyder-py3\\F_trial3.out')
my_interpolating_function = RegularGridInterpolator(points=[r, r, e, d], values=trial4_6dmesh(r,e,d))

pts=(R,S,E,D)
#print (my_interpolating_function(pts),'interp for trial4')
#print (trial4_6dmesh(r,e,d),'actual fxn for trial4')

n_new=2
d_new=np.linspace(0.1,0.5,n_new)
r_new=np.linspace(0.6,0.9,n_new)
e_new=np.linspace(1.5,1.9,n_new)

R_new,S_new,E_new,D_new=np.meshgrid(r_new,r_new,e_new,d_new,indexing='ij')
pts=(R_new,S_new,E_new,D_new)
#print (my_interpolating_function(pts),'interp 2 for trial4')
#print (trial4_6dmesh(r_new,e_new,d_new),'actual fxn 2 for trial4')

#print (interp(pts))
'''
my_interpolating_function = RegularGridInterpolator((d, r, e), trial4(d,r,e))
pts = np.array([[0.1, 0.2, 0.3]])
#print (my_interpolating_function(pts))
#print (trial4(0.1,0.2,0.3))
print (trial4(0.1,r,e),'trial4')

for i in d:
    print (trial4(i,r,e),'trial4')
    
plt.plot(r,my_interpolating_function(0.1,r,0.3))
plt.show()    
print (my_interpolating_function(np.array([[0.1,r,e]])),'interp')
for i in d:
    print (my_interpolating_function(np.array([[i,r,e]])),'interp')
'''
#np.savetxt('F_trial3.out',trial4(d,r,e))
    
def trial4_2(d):
    integral=sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(trial4(r,d,e),r,axis=0),r,axis=0),e,axis=0)
    return integral

#print (trial4_2(d))
import scipy.integrate as sp
from scipy import LowLevelCallable
from numba import cfunc, types
import useful_functions as uf
import numpy as np
import ctypes


x_min=0
x_max=5
#chi=uf.chi(x)
def func(x):
    return x**2


print (sp.quad(lambda x: x*func(x),x_min,x_max))
x=np.linspace(x_min,x_max,10)

func=func(x)

print (sp.trapz(x*func,x))

c_sig = types.double(types.intc, types.CPointer(types.double))
@cfunc(c_sig)
def f(n_args, x):
    x1=x[0]
    func=x[1]
    return x1*func


print (sp.nquad(LowLevelCallable(f.ctypes), [[0,25],[0,5]], full_output=True))

import numpy as np
import scipy.integrate as si
import numba
from numba import cfunc
from numba.types import intc, CPointer, float64
from scipy import LowLevelCallable

def func(x):
    return x**2


x=np.linspace(x_min,x_max,10)

func=func(x)

import scipy as sp
import useful_functions as uf

ell=np.geomspace(1,1e4,1000)
Mps_interpf=uf.Mps_interpf
def power_spec(ell):
    kpar=np.geomspace(1e-4,1e-1,1000)
    array=np.array([])
    z=1
    for i in ell:
        k=np.sqrt(kpar**2+i**2/uf.chi(z)**2)
        integrand=Mps_interpf(k)
        integral=sp.integrate.trapz(integrand,kpar,axis=0)
        array=np.append(array,integral)
    return array

plt.plot(ell,ell**2*power_spec(ell)/2/np.pi)

