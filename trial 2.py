# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 11:10:54 2018

@author: zahra
"""

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


n=100

x=np.linspace(0,4,n)
y=np.linspace(0,3,n)
z=np.linspace(0,5,n)

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


d=np.linspace(0,1,n)
r=np.linspace(0,1,n) 
e=np.linspace(0,2,n)

def trial_check(d):
    a=np.linspace(0,5,n)
    b=np.linspace(0,4,n)
    A,B,R,S,E=np.meshgrid(a,b,r,r,e) 
    array=np.array([]) 
    for i in d:
     integrand=sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz
                            ((i*A+E)*B*R*S,a,axis=0),b,axis=0),r,axis=0),r,axis=0),e,axis=0)
     array=np.append(array,integrand)
    return array   

def trial4_6dmesh(r,e,d):
    a=np.linspace(0,5,n)
    b=np.linspace(0,4,n)
    A,B,R,S,E,D=np.meshgrid(a,b,r,r,e,d,indexing='ij')     
    integral=sp.integrate.trapz(sp.integrate.trapz((D*A+E)*B*R*S,a,axis=0),b,axis=0)
    return integral

#print (trial4_6dmesh(d),np.shape(trial4_6dmesh(d)),'trial4_6dmesh')

print (trial_check(d))


def trial4(r,e,d):
    a=np.linspace(0,5,n)
    b=np.linspace(0,4,n)
    A,B,R,S=np.meshgrid(a,b,r,r) 
    array=np.array([])
    for i in d:
        for j in e:
            integrand=sp.integrate.trapz(sp.integrate.trapz((i*A+j)*B*R*S,a,axis=0),b,axis=0)
            array=np.append(array,integrand)
    F=np.reshape(array,(n,n,n,n))
    F_trans=np.transpose(F)
    return F_trans

R,S,E,D=np.meshgrid(r,r,e,d,indexing='ij') 


print (np.shape(trial4(d,r,e)))

#print (F,'reshaped for loops')
#np.savez('trial5',F=trial4(d,r,e))
npzfile = np.load('trial5.npz')
#print (npzfile['F'])
#F_new=np.genfromtxt('C:\\Users\\zahra\\.spyder-py3\\F_trial3.out')
my_interpolating_function = RegularGridInterpolator(points=[r, r, e, d], values=trial4(r,e,d))

pts=(R,S,E,D)
print (my_interpolating_function(pts),'interp for trial4')
print (trial4(r,e,d),'actual fxn for trial4')

d_new=np.linspace(0.1,0.5,5)
r_new=np.linspace(0.6,0.9,5)
e_new=np.linspace(1.5,1.9,5)
#pts=np.array([0.1,0.2,0.3])
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
    integral=sp.integrate.trapz(sp.integrate.trapz(sp.integrate.trapz(my_interpolating_function(pts),r,axis=0),r,axis=0),e,axis=0)
    return integral

print (trial4_2(d))