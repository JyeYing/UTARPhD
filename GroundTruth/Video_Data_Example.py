# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:36:07 2020

@author: jyeyings
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
import matplotlib.pyplot as plt

fname = 'data/frame38.jpg'
img = imageio.imread(fname)
#plt.imshow(img[:,:,0],cmap='bone')

img_r = img[40:,27:,0]
img_g = img[40:,27:,1]
img_b = img[40:,27:,2]

img_gray =0.2989*img_r + 0.5870*img_g + 0.1140*img_b
plt.imshow(img_gray, cmap='Greys_r')

np.save('tensor_data.npy', img_gray)
np.load('tensor_data.npy')

print(np.shape(img_gray))

class params:
#    alpha = 0.002011
    x_0=0
    x_end=10
    t_0=0
    t_end=820
    
    nx=439
    nt=820

    x = np.linspace(x_0, x_end, nx+1)   # mesh points in space
    dx = x[1] - x[0]
    t = np.linspace(t_0, t_end, nt+1)
    dt = t[1] - t[0]
#        
#    const=alpha*dt/(dx**2)
#    
    lamda = 1

class functional_var:
    parameters = params()
   
    #u = np.zeros((parameters.nx+1,parameters.nt+1))
    u=img_gray
    print('u is', u)
    print('shape of u is', np.shape(u))
    u_t = np.zeros_like(u)
#    u_x =  np.zeros_like(u)
#    u_xx = np.zeros_like(u)

    u_remove=u[2:,1:]

    syy=np.ones((parameters.nx-1,parameters.nt))
    sx1x1=np.ones_like(syy)
    sxy=np.ones_like(syy)
    sx2x2=np.ones_like(syy)
    sx2y=np.ones_like(syy)
    sx3y=np.ones_like(syy)
    sx3x3=np.ones_like(syy)

def plot_u_values(v):
    fig1=plt.figure(1)
    [T,X] = np.meshgrid(v.parameters.t,v.parameters.x)
    ax1 = plt.axes(projection='3d')
    ax1.plot_surface(X,T,v.u)
    fig2 = plt.figure(2)
    ax2 = plt.plot(v.parameters.x,v.u[:,0])   #2D plot
    fig3 = plt.figure(3)
    ax3 = plt.plot(v.parameters.x,v.u[:,-1])
    
def ut_value(f):
    f.u_t=(f.u[2:, 1:] - f.u[2:, :-1])/f.parameters.dt
    ut_bar=np.average(f.u_t)
    #print (f.u_t)
    return (f.u_t,ut_bar)
#
def Sut_ut_value(f):
    (u_t, ut_bar)=ut_value(f)
#    print(("Values of ut_bar is"), (ut_bar))
#    print(("Size of ut is"),(np.shape(f.u_t)))
#    print (np.shape(functional_var.syy))
    
    for j in range (0,f.parameters.nt):
        for i in range (0,f.parameters.nx-1):
            functional_var.syy[i,j]=(f.u_t[i,j]-ut_bar)**2
        sum_syy=np.sum(functional_var.syy[:,:])
#    print(("Values of sum of squares, Su_t u_t is"), (sum_syy))
#    print(("Size of u_t u_t is"), np.shape(functional_var.syy))
#    print(functional_var.syy[:,:])
    return (sum_syy)

def Suu_value(f):
    u_bar=np.average(f.u_remove)
#    print(("Values of u_bar is"), (u_bar))
#    print(("Size of u is"),(np.shape(f.u_remove)))
#    print(f.u_remove)
    
    functional_var.sx1x1[:,:-1]=(f.u_remove[:,:-1]-u_bar)**2
    sum_sx1x1=np.sum(functional_var.sx1x1[:,:])
#    print(("Values of sum of squares, Suu is"), (sum_sx1x1))
#    print(("Size of uu is"), np.shape(functional_var.sx1x1))
#    print(functional_var.sx1x1[:,:])
    return (sum_sx1x1)

def Su_ut_value(f):
    u_bar=np.average(f.u_remove)
    (u_t, ut_bar)=ut_value(f)
    
    functional_var.sxy[:,:-1]=(f.u_remove[:,:-1]*f.u_t[:,:-1])-(3*u_bar*ut_bar)
    sum_sxy=np.sum(functional_var.sxy[:,:])
#    print(("Values of sum of squares, Su_ut is"), (sum_sxy))
#    print(("Size of u_ut is"), np.shape(functional_var.sxy))
#    print(functional_var.sxy[:,:])
    return (sum_sxy)

def ux_value(f):
    f.ux=(f.u[2:, :-1] - f.u[:-2, :-1])/f.parameters.dx
    ux_bar=np.average(f.ux)
    return (f.ux,ux_bar)

def Sux_ux_value(f):
    (ux, ux_bar)=ux_value(f)
#    print(("Size of ux is"), np.shape(f.ux))
    
    functional_var.sx2x2[:,:]=(f.ux[:,:]-(ux_bar)**2)
    sum_sx2x2=np.sum(functional_var.sx2x2[:,:])
#    print(("Values of sum of squares, Sux_ux is"), (sum_sx2x2))
#    print(("Size of ux_ux is"), np.shape(functional_var.sx2x2))
#    print(functional_var.sx2x2[:,:])
    return (sum_sx2x2)

def uxx_value(f):
    f.u_xx=(f.u[2:, :-1] + f.u[:-2,:-1] - 2*f.u[1:-1,:-1])/(f.parameters.dx**2)
    uxx_bar=np.average(f.u_xx)
    return (f.u_xx,uxx_bar)

def Sux_ut_value(f):
    (ux, ux_bar)=ux_value(f)
    (u_t, ut_bar)=ut_value(f)
       
    functional_var.sx2y[:,:]=(ux[:,:]*f.u_t[:,:])-(3*ux_bar*ut_bar)
    sum_sx2y=np.sum(functional_var.sx2y[:,:])
#    print(("Values of sum of squares, Sux_ut is"), (sum_sx2y))
#    print(("Size of ux_ut is"), np.shape(functional_var.sx2y))
#    print(functional_var.sx2y[:,:])
    return (sum_sx2y)

def Suxx_ut_value(f):
    (u_t, ut_bar)=ut_value(f)
    (u_xx,uxx_bar)=uxx_value(f)
       
    functional_var.sx3y[:,:]=(f.u_xx[:,:]*f.u_t[:,:])-(3*uxx_bar*ut_bar)
    sum_sx3y=np.sum(functional_var.sx3y[:,:])
#    print(("Values of sum of squares, Suxx_ut is"), (sum_sx3y))
#    print(("Size of uxx_ut is"), np.shape(functional_var.sx3y))
#    print(functional_var.sx3y[:,:])
    return (sum_sx3y)

def Suxx_uxx_value(f):
    (u_xx,uxx_bar)=uxx_value(f)
    
    functional_var.sx3x3[:,:]=(f.u_xx[:,:]-uxx_bar)**2
    sum_sx3x3=np.sum(functional_var.sx3x3[:,:])
#    print(("Values of sum of squares, Suxx_uxx is"), (sum_sx3x3))
#    print(("Size of uxx_uxx is"), np.shape(functional_var.sx3x3))
#    print(functional_var.sx3x3[:,:])
    return (sum_sx3x3)

def update_u(v):
#    set_bound_cond(v)
#    setup_pde(v)
    pass
#    Sut_ut_value(v)
#    Suu_value(v)
#    Su_ut_value(v)
#    Sux_ux_value(v)
#    Sux_ut_value(v)
#    Suxx_ut_value(v)
#    Suxx_uxx_value(v)
    
def coeff_u(v):
    eqn1=Sut_ut_value(v)- v.parameters.lamda*Suu_value(v)
    eqn2=np.sqrt(((eqn1)**2) + (4*v.parameters.lamda*(Su_ut_value(v)**2)))
    eqn3=2*Su_ut_value(v)
    u_coeff=(eqn1+eqn2)/eqn3
#    print (u_coeff)
    return (u_coeff)

def coeff_ux(v):
    eqn4=Sut_ut_value(v)- v.parameters.lamda*Sux_ux_value(v)
    eqn5=np.sqrt(((eqn4)**2) + (4*v.parameters.lamda*(Sux_ut_value(v)**2)))
    eqn6=2*Sux_ut_value(v)
    ux_coeff=(eqn4+eqn5)/eqn6
#    print(ux_coeff)
    return (ux_coeff)

def coeff_uxx(v):
    eqn7=Sut_ut_value(v)- v.parameters.lamda*Suxx_uxx_value(v)
    eqn8=np.sqrt(((eqn7)**2) + (4*v.parameters.lamda*(Suxx_ut_value(v)**2)))
    eqn9=2*Suxx_ut_value(v)
    uxx_coeff=(eqn7+eqn8)/eqn9
#    print (uxx_coeff)
    return (uxx_coeff)

def const_coeff(v):
    (u_coeff, ux_coeff,uxx_coeff)=(coeff_u(v), coeff_ux(v),coeff_uxx(v))
    u_bar=np.average(v.u_remove)
    (u_t, ut_bar)=ut_value(v)
    (ux, ux_bar)=ux_value(v)
    (u_xx,uxx_bar)=uxx_value(v)
    const_coeff = ut_bar - u_coeff *u_bar - ux_coeff * ux_bar - uxx_coeff * uxx_bar
##    const=ut_bar-a1_u*u_bar-a2_ux*ux_bar-a3_uxx*uxx_bar
#    print(const_coeff)
    return (const_coeff)
#    
def printing(v):
    print(("Values of sum of squares, Sut_ut is"), (Sut_ut_value(v)))
    print(("Values of sum of squares, Suu is"), (Suu_value(v)))
    print(("Values of sum of squares, Su_ut is"), (Su_ut_value(v)))
    print(("Values of sum of squares, Sux_ux is"), (Sux_ux_value(v)))
    print(("Values of sum of squares, Sux_ut is"), (Sux_ut_value(v)))
    print(("Values of sum of squares, Suxx_ut is"), (Suxx_ut_value(v)))
    print(("Values of sum of squares, Suxx_uxx is"), (Suxx_uxx_value(v)))
    
    print ('Constant value is', (const_coeff(v)))
    print ('Coefficient of U is', (coeff_u(v)))
    print ('Coefficient of Ux is', (coeff_ux(v)))
    print ('Coefficient of Uxx is', (coeff_uxx(v)))
    
# main workflow
def main():
#    #v = initialisation()
    v = functional_var()
    #print (v.u)  
    
    ut_value(v)
#    set_bound_cond(v)
#    setup_pde(v)
    plot_u_values(v)
#    
    update_u(v)

#    coeff_u(v)
    coeff_ux(v)
    coeff_uxx(v)
    const_coeff(v)
    printing(v)
    
main()