# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:58:34 2026

ground: pinn + admm 
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy as sp
import pywt
import time
import matplotlib.pyplot as plt
import scipy.integrate as scint
from scipy.interpolate import CubicSpline
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense #, Lambda
from keras.saving import save_model
from tensorflow.keras import backend as K
from tensorflow import keras
from scipy.optimize import minimize
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from scipy.linalg import solve
from box import Box
import pickle

from sklearn.linear_model import Lasso

start_time = time.time()

lamda = 1e-2
rho = ρ = tf.constant(1, dtype=tf.float64)
learn_rate = -2e-6
adam_learn = 1e-5
losses_sparse = []
N = 99
t_end = 1.30

t0 = tf.constant([[0]],dtype=tf.float64)
x0 = tf.constant([[25]],dtype=tf.float64)
y0 = tf.constant([[31]],dtype=tf.float64)

t_bc = tf.constant([[1.56]],dtype=tf.float64)
x_bc = tf.constant([[820]],dtype=tf.float64)
y_bc = tf.constant([[410]],dtype=tf.float64)

α = 1 #np.sqrt(5) / 2

row = 5
col = 1

sigma = 1e-3  #>sigma-->keep
sigma1 = -1e-3 #proximal 
sigma2 = 5e-5

α1 = 0.001 #lasso weight search for dxdt
α2 = 0.001 #lasso weight search for dydt 
αf1 = 1e-4 #loss func cal for dxdt
αf2 = 1e-4 #loss func cal for dydt

data = pd.read_csv("coor_txy_inter.csv")
train_data = tf.convert_to_tensor(data.values)

def make_data():    
    with open ('ground_inter.pkl','rb') as f:
        pdata = pickle.load(f)
       
    # global data
    libdata ={
        
           't_given' : train_data[1:-1,0],
           'x_given' : train_data[1:-1,1],
           'y_given' : train_data[1:-1,2],
           
           't_pred' : train_data[:-1,0],
           'x_admm' : train_data[:-1,1],
           'y_admm' : train_data[:-1,2],
           #'dxdt_manual' : train_data[1:-1,3],
           #'dydt_manual' : train_data[1:-1,4],
           
           'u_pred':pdata.u_pred,
           
           'output': pdata.output,
           #'t_spline' :pdata.t_spline,
           'out_spline' :pdata.out_spline,
           'der_spline' :pdata.der_spline,
           # 'dxdt_given' : train_data[:N,3],
           # 'dydt_given' : train_data[:N,4],
           # 'ori_x': train_data[:N,5],
           # 'ori_y' : train_data[:N,6],
           # 'ori_dxdt':train_data[:N,7],
           # 'ori_dydt':train_data[:N,8],
           
           # 'va_t' : train_data[N:0],
           # 'va_ori_x' : train_data[N:,5],
           # 'va_ori_y' : train_data[N:,6],
           
           'ddt':pdata.ddt,  #pinn's u_t
           'pinn_dxdt' :pdata.pinn_dx,
           'pinn_dydt': pdata.pinn_dy,
           
           'model': None,
           
           'ori_ddt' :[],
           'w1':[],
           'w2':[],
           'iv1':[],
           'iv2':[],
           'sol_lasso':[],
           
           'identity' : [],
           'phi1' : [],
           'phi2':[],
           'lambda_T' :[],
           'y_ij1' : [],
           'y_ij2' :[],
           'z_T1' : [],
           'z_T2' :[],
           'lambda_LU' : [],
           'sol_LU' :[],
           
           'z_T3' : [],
           'z_T4' :[],
         
           'end_dx' :[],
           'end_dy' :[],
           
           #'model': pdata.model, 
           'fname':None,
           
           't_vec' : [],
           #'x_vec' : pdata.x_vec,
           #'y_vec' : pdata.y_vec,
           
           
           'lambda_phi': [],
           'y_lam' : [],
           'term1_tra' : [],
           'term2_lamz' :[],
           'y_lam_changes' : [],
           'rho_lam_change' : [],
           'L_grad' : [],
           
           'sol_lambda' : [],
        
           'beta1' :[],
           'beta2' :[],
           'sol_func':[],
           
           'beta3' :[],
           'beta4' :[],
           'sol_func_em':[],
           #'u_test': [],
           #'numerical_rhs':[],
           #'grads':[],
           'losses':[],
           'loss_sparse':[],
           'losses_sparse':[],
           
           }
    return Box(libdata)

def loading_model(libdata):   
   
    def custom_loss(y_true, y_pred):
        
        # Initial condition
        #y_pred0 = model(t0)
        
        # Boundary condition
        #y_pred_bc = model(t_bc)
        
        #to_take = sampling + N_f - 5
        #loss_ic = K.mean(K.square(y_true[:to_take] - y_pred[:to_take]))
        loss_data = K.mean(K.square(tf.cast(y_true, tf.float32) - (tf.cast(y_pred, tf.float32))))

        with tf.GradientTape() as tape:
            tape.watch(t0)
            y_pred0 = model(t0)       # shape [1,2]
            
        J = tape.jacobian(y_pred0, t0)  #dx/dt is [0,0,0,0]; dy/dt is [0,1,0,0]
        print(J)
 
        loss_ic = tf.reduce_mean(tf.square(y_pred0 - [x0,y0]))
        #loss_bc = tf.reduce_mean(tf.square(y_pred_bc - [x_bc,y_bc]))
        #print(loss_ic.numpy(), loss_f.numpy())
        
        #loss = loss_ic + loss_f
        #loss = tf.cast(loss_data, tf.float32) + tf.cast(loss_ic, tf.float32) + tf.cast(loss_f, tf.float32)
        loss = tf.cast(loss_data, tf.float32) + tf.cast(loss_ic, tf.float32) #+ tf.cast(loss_bc, tf.float32)

        tf.print('loss on data is:', loss_data)
        tf.print('loss on ic is:', loss_ic)
        #tf.print('loss on bc is:', loss_bc)
        #tf.print('loss on derivative(GTape) is:', loss_f)
        tf.print('total loss without bc is:', loss)
        
        return loss
    
    model = load_model('ground_fitted_model.keras', compile=False)
        
    """3 steps: compile, fit, predict.       """
    #model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss=custom_loss, optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
    
    return model
    

def prepare_data(libdata):
    
    u_pred = libdata['u_pred']
    out_spline = libdata['out_spline']
    x = out_spline[:,0]
    y = out_spline[:,1]
    
    const_one = tf.ones([N,])
    const_one = tf.cast(const_one,dtype=tf.float64)
    
    x2 = tf.square(x)
    y2 = tf.square(y)
    
    x3 = tf.pow(x, 3)
    y3 = tf.pow(y, 3)
    
    xy = tf.multiply(x,y)
    
    x2y = tf.multiply(x2,y)
    xy2 = tf.multiply(x,y2)
    
    identity = tf.eye(row,dtype=tf.float64)  
           
    #array = tf.stack([x,x_sqr,y_cube,x2y2],axis=0)
    #array = tf.stack([const_one,x,y],axis=0)
    array1 = tf.stack([x,x2,y,y2,xy],axis=0)
    array2 = tf.stack([x,x2,y,y2,xy],axis=0)
    #print(array)   #(5x100)
    iv1 = np.c_[x,x2, y,y2, xy]  #+ xy
    iv2 = np.c_[x, x2, y,y2, xy]  # + xy 
    
    #array_var_T = tf.transpose(array)
    #print(array_var_T)
    
    u_vec = tf.ones([row,col])
    v_vec = tf.ones([row,col])
    lammda = np.c_[u_vec,v_vec]
    lambda_T = lammda.T   #dtype:float32
    lambda_T = tf.cast(lambda_T,dtype=tf.float64) #convert tensor to tensor
    #print(matrix_T)
    y_coeff1 = y_coeff2 = tf.cast(tf.ones([2,row])/10,dtype=tf.float64)
    z_T1 = z_T2 = z_T3 = z_T4 = tf.convert_to_tensor(np.ones_like(lambda_T), dtype=tf.float64)
    #z_T = tf.convert_to_tensor(np.random.rand(2,6), dtype=tf.float64)
 
    libdata['iv1'] = iv1
    libdata['iv2'] = iv2
    libdata['identity'] = identity
    libdata['phi1'] = array1
    libdata['phi2'] = array2
    libdata['lambda_T'] = lambda_T
    libdata['y_ij1'] = y_coeff1
    libdata['y_ij2'] = y_coeff2
    libdata['z_T1'] = z_T1
    libdata['z_T2'] = z_T2
    libdata['z_T3'] = z_T3
    libdata['z_T4'] = z_T4


#--------lasso fitting to get initial coefficient matrix------
def lass_fit(libdata):
    #given_xy = las_data['output']
    ddt = libdata['der_spline']
    iv1 = libdata['iv1']
    iv2 = libdata['iv2']
    
    u_pred = libdata['u_pred']
    out_spline = libdata['out_spline']
    x = out_spline[:,0]
    y = out_spline[:,1]
    
    #x2 = tf.square(x)
    #y2 = tf.square(y)   
    #xy = tf.multiply(x,y)
    
    #iv = np.c_[x,y, x2, y2, xy]
    
    dxdt = ddt[:,0]
    
    lasso1 = Lasso(alpha=α1, fit_intercept=True, tol=1e-4, max_iter=1000)
    lasso1.fit(iv1,dxdt)
    w1 = np.array(list(lasso1.coef_))# + [lasso1.intercept_])
    print('lasso for y1(dxdt) is', w1)
    #print('lasso loss for y1(dxdt) is',0.5*sum((lasso1.predict(iv)-dxdt)**2) + 1*sum(np.abs(w1)))
    
    def fun1(w1, α1=α1):
        
        #iv = np.c_[x,y, x2, y2, xy]
        
        dxdt = ddt[:,0]
        #n=100
        
        XX1 = np.c_[iv1]#, np.ones_like(dxdt)]
        #print(XX1)
        y_predict1 = XX1 @ w1 #+ 0.1 * np.random.randn(n)  #(100,1)
        f1 = np.sum( (dxdt - y_predict1)**2 )
        g1 = np.abs(w1).sum()
        loss1 = 0.5*f1 + α1*g1
      
        return loss1    
    
    
    dxdt_result = minimize(fun1, w1, tol=0.0001)
    #print(dxdt_result)
    coeff_min = dxdt_result.x
    print('Non-sparse coeff for dxdt',coeff_min)
    loss_min = dxdt_result.fun
    #print('loss_scipy_min_dxdt',loss_min)
    #y_pred_dxdt = (w1 @ coeff_min).flatten()
    #print('non-sparse loss',y_pred_dxdt)
    
    
    dydt = ddt[:,1]
    
    lasso2 = Lasso(alpha=α2, fit_intercept=True, tol=1e-4, max_iter=1000)
    lasso2.fit(iv2,dydt)
    w2 = np.array(list(lasso2.coef_))# + [lasso2.intercept_])
    print('lasso for y2(dydt) is',w2)
    #print('lasso loss for y2(dydt) is',0.5*sum((lasso2.predict(iv)-dydt)**2) + 1*sum(np.abs(w2)))
    
    def fun2(w2, α2=α2):
        
        dydt = ddt[:,1]
        
        XX2 = np.c_[iv2]#, np.ones_like(dydt)]
        y_predict2 = XX2 @ w2 - np.log(np.abs(tf.multiply(x,y))) #* np.random.randn(n)
        f2 = np.sum( (dydt - y_predict2)**2 )
        g2 = np.abs(w2).sum()
        loss2 = 0.5*f2 + αf2*g2
        
        return loss2
    
    dydt_result = minimize(fun2, w2, tol=0.0001)
    #print(dxdt_result)
    coeffdy_min = dydt_result.x
    print('Non-sparse coeff for dydt',coeffdy_min)
    loss_mindy = dydt_result.fun
    #print('loss_scipy_min_dydt',loss_mindy)
    
    libdata['w1'] = w1
    libdata['w2'] = w2
    #libdata['iv'] = iv
    
    return w1, w2 #, iv

#---------------- functional as initial--------
def em_mlr(X, y, num_iter=100, tol=1e-6):

    n, p = X.shape

    beta_est = np.random.randn(p)
    variance = np.var(y) + 1e-8
    penalty = 1e-6

    for _ in range(num_iter):

        # E-step
        y_pred = X @ beta_est
        residual = (y - y_pred)**2

        log_r = -0.5 * residual / variance
        log_r -= np.max(log_r)

        responsibilities = np.exp(log_r)

        sum_r = np.sum(responsibilities)
        if sum_r < 1e-12:
            responsibilities = np.ones(n) / n
        else:
            responsibilities /= sum_r

        # M-step
        W = np.diag(responsibilities)

        X_weighted = X.T @ W @ X + penalty * np.eye(p)
        y_weighted = X.T @ W @ y

        new_beta_est = np.linalg.solve(X_weighted, y_weighted)

        if np.linalg.norm(new_beta_est - beta_est) < tol:
            break

        beta_est = new_beta_est

    return beta_est
#----------------end of sparse_noise ------------------

def sparse_LU(libdata):
    iv1 = libdata['iv1']
    iv2 = libdata['iv2']
    ddt = libdata['der_spline']
    
    z_T1 = libdata['z_T1']
    z_T2 = libdata['z_T2']
    
    pinn_dxdt = libdata['pinn_dxdt']
    pinn_dydt = libdata['pinn_dydt']
    
    #-----------------------------------
    #dxdt = ori_ddt[:,0]
    #dydt = ori_ddt[:,1]
    #XX1 = np.c_[np.ones_like(dxdt),iv1]    #(100,6)
    XX1 = iv1
    XX1_ddt = ddt.T @ XX1
    #print('XX1_ddt is', XX1_ddt)   #values equal to ut_phi1
    XTX1 = XX1.T @ XX1 
    #print('XTX1', XTX1)   #values equal to phi_phiT
    #aa = ρ* np.eye( *XTX.shape)
    #print('ρI', aa)
    A1 = XTX1 + ρ* np.eye( *XTX1.shape) 
    #print('LU_method A1 matrix', A1)   #values equal ppT_rI
    invr1 = tf.linalg.pinv(A1)
    #print('inverse1', invr1)

    #XX2 = np.c_[np.ones_like(dxdt),iv2]    #(100,6)
    XX2 = iv2
    XX2_ddt = ddt.T @ XX2
    #print('XX2_ddt is', XX2_ddt)   #values equal to ut_phi2
    XTX2 = XX2.T @ XX2 
    #print('XTX2', XTX2)   #values equal to phi_phiT
    #aa = ρ* np.eye( *XTX.shape)
    #print('ρI', aa)
    A2 = XTX2 + ρ* np.eye( *XTX2.shape) 
    #print('LU_method A2 matrix', A2)   #values equal ppT_rI
    invr2 = tf.linalg.pinv(A2)
    #print('inverse2', invr2) 

    lambda_LU = np.vstack((tf.ones([col,row]), tf.ones([col,row])))
    xi1 = xi2 = np.ones_like(lambda_LU)/10
    #xi = np.ones_like(lambda_LU)
    z = np.ones_like(lambda_LU)


    losses_LU_dx = []
    losses_LU_dy = []
    
    for i in range (1000):
         #np.hstack((b_dx, b_dy))
       
        lu_piv1 = sp.linalg.lu_factor(A1)
        lu_piv2 = sp.linalg.lu_factor(A2)
        #print('LU factorization', lu_piv1)
        aa1 = XX1_ddt -  xi1
        #print('XX1_ddt - y1', aa1)    #values same as subtra1
        aa2 = XX2_ddt -  xi2
        #print('XX2_ddt - y2', aa2)    #values same as subtra2
        b_d1 = XX1_ddt +  ρ*z - xi1
        b_d2 = XX2_ddt +  ρ*z - xi2
        #print('b_d1 is', b_d1)  #same as u_pT_t_rz1
        #print('b_d2 is', b_d2)  #same as u_pT_t_rz2
        lambda_dx = sp.linalg.lu_solve(lu_piv1, b_d1[0,:])
        lambda_dy = sp.linalg.lu_solve(lu_piv2, b_d2[1,:])
        lambda_LU = tf.transpose(np.c_[lambda_dx,lambda_dy])  #solver different from tf.matmul in lambda_T computation
        #print('lambda_LU is', lambda_LU)
    
    #____________to update z for frac2, using lambda_LU____

        frac1 = np.array([α1/rho])
        z1 = pywt.threshold(lambda_LU + xi1/ρ , frac1 , 'soft') 
        #print('Soft z(LU1) is', z1)
        z_T1 = z1
        
        frac2 = np.array([α2/rho])
        z2 = pywt.threshold(lambda_LU + xi2/ρ , frac2 , 'soft') 
        #print('Soft z(LU2) is', z2)
        z_T2 = z2

#_________to update yij, carry forward using lambda_LU _______
        xi1 = xi1 + ρ*(lambda_LU - z1)
        #print('x1 is', xi)
        
        xi2 = xi2 + ρ*(lambda_LU - z2)
        #print('x1 is', xi)
        
        #print('lambda_matrix k+1 transpose is', lambda_T)
        #print('z_k+1 is',z)
        #lambda_T = prox(lambda_T, sigma)
        #lu_piv_ten = np.array(lu_piv).reshape(6,6)
        #lam_T_LU = tf.matmul(u_pT_t_rz, lu_piv_ten)
        #print('lambda LU is',lam_T_LU)
    
    
    #--------------cal of loss
    #L = min 1/2 ||lambda_T*phi - ddt||^2 + lam ||z|| + sum y_ij(lambda_T-z_T) + rho/2 ||lambda_T-z_T||^2
    
        y_predict1 = XX1 @ lambda_dx 
        f1 = np.sum( (pinn_dxdt - y_predict1)**2 )
        g1 = np.abs(lambda_dx).sum()
        h1 = np.sum(xi1*(lambda_dx-z_T1[0,:]))
        i1 = np.sum((lambda_dx-z_T1[0,:])**2)
        loss1 = 0.5*f1 + α1*g1 + h1 + rho*0.5*i1
        
        losses_LU_dx.append(loss1)
        
        y_predict2 = XX2 @ lambda_dy  
        f2 = np.sum( (pinn_dydt - y_predict2)**2 )
        g2 = np.abs(lambda_dy).sum()
        h2 = np.sum(xi2*(lambda_dy-z_T2[1,:]))
        i2 = np.sum((lambda_dy-z_T2[1,:])**2)
        loss2 = 0.5*f2 + α2*g2 + h2 + rho*0.5*i2
        
        losses_LU_dy.append(loss2)
        
        #L = loss1 + loss2
        #print('total loss is', L)
    
    #XX = np.c_[XX1,XX2]
    y_pred1 = XX1 @ tf.transpose(lambda_LU)
    y_pred2 = XX2 @ tf.transpose(lambda_LU)
    f_de = np.sum((ddt-y_pred1)**2) + np.sum((ddt - y_pred2)**2)
    f_data = losses_LU_dx[-1] + losses_LU_dy[-1]
    print('Data loss (LU) is', f_data)
    print('Derivative loss (LU) is', f_de)
    
    fig,ax2 = plt.subplots(1,1)
    #ax2.plot(losses1, label = 'losses of dxdt')
    ax2.semilogy(losses_LU_dx, label='log losses of LU_dxdt')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("loss value")
    ax2.legend(loc='upper right')
    fig.savefig('losses_dxdt_LU.png', dpi=300)
    
    fig,ax3 = plt.subplots(1,1)
    #ax3.plot(losses2, label = 'losses of dydt')
    ax3.semilogy(losses_LU_dy, label='log losses of LUdydt')
    ax3.set_xlabel("Epoch")
    ax3.legend(loc='upper right')
    fig.savefig('losses_dydt_LU.png', dpi=300)

    print('Coeff matrix (LU) is', lambda_LU)
    
    libdata['z_T1'] = z_T1
    libdata['z_T2'] = z_T2
    libdata['lambda_LU'] = lambda_LU

    
    return lambda_LU 


def sparse(libdata):
    array1 = libdata['phi1']
    array2 = libdata['phi2']
    identity = libdata['identity']
    ddt = libdata['der_spline']
    lambda_T = libdata['lambda_T']
    y_coeff1 = libdata['y_ij1']
    y_coeff2 = libdata['y_ij2']
    z_T3 = libdata['z_T3']
    z_T4 = libdata['z_T4']
    
    pinn_dxdt = libdata['pinn_dxdt']
    pinn_dydt = libdata['pinn_dydt']
   
   

    
    # term1_tra = libdata['term1_tra']
    # term2_lamz = libdata['term2_lamz']

    # y_lam = libdata['y_lam']
    # y_lam_changes = libdata['y_lam_changes']
    # rho_lam_change = libdata['rho_lam_change']
    # L_grad = libdata['L_grad']
    
#____________to update lambda, z and y____
       
#--------------forming del_L_del_lambda to fine tune learning rate ---------
       # lam_phiphiT = tf.matmul(lambda_T, phi_phiT)
       # lam_z_diff = tf.subtract(lambda_T, z_T)
       # rho_lam_z = tf.multiply(rho, lam_z_diff)
       # del_l_del_lambda = lam_phiphiT - uarr + y_coeff + rho_lam_z
       # #print('del l del lambda', del_l_del_lambda)
       # lambda_T = lambda_T - tf.multiply(learn_rate, del_l_del_lambda)
#----------------end tuning of learning rate--------

    ut_phi1 = tf.matmul(tf.transpose(ddt), tf.transpose(array1))
    #print('u_t phi1 is', ut_phi1)
    phi_phiT1 = tf.matmul(array1, tf.transpose(array1))  #phi: (6x100)
    #print('phiphiT1',phi_phiT1)
    rhoI = tf.multiply(rho,identity)
    #print('rhoI', rhoI)
    ppT_rI1 = tf.add(phi_phiT1,rhoI)   
    #print('phi2^T1 + rhoI', ppT_rI1)
    ppTrI_inv1 = tf.linalg.pinv(ppT_rI1)
    #print('least square inverse1 is', ppTrI_inv1)
    
    ut_phi2 = tf.matmul(tf.transpose(ddt), tf.transpose(array2))
    #print('u_t phi2 is', ut_phi2)
    phi_phiT2 = tf.matmul(array2, tf.transpose(array2))  #phi: (6x100)
    #print('phiphiT2',phi_phiT2)
    rhoI = tf.multiply(rho,identity)
    #print('rhoI', rhoI)
    ppT_rI2 = tf.add(phi_phiT2,rhoI)   
    #print('phi2^T2 + rhoI', ppT_rI2)
    ppTrI_inv2 = tf.linalg.pinv(ppT_rI2)
    #print('least square inverse2 is', ppTrI_inv2)
    
    losses1 = []
    losses2 = []
   
    
    for i in range (100):
  #_______to update lambda_T__
    
        subtra1 = tf.subtract(ut_phi1, y_coeff1)
        #print('ut_phi1 - y1', subtra1)
        rhoz1 = tf.multiply(rho,z_T3)
        u_pT_t_rz1 = tf.add(subtra1, rhoz1)
        #print('ut_phi1 + rhoz1 - y1 is', u_pT_t_rz1)
        lambda_T1 = tf.matmul(u_pT_t_rz1, ppTrI_inv1)
        
        subtra2 = tf.subtract(ut_phi2, y_coeff2)
        #print('ut_phi2 - y2', subtra2)
        rhoz2 = tf.multiply(rho,z_T4)
        u_pT_t_rz2 = tf.add(subtra2, rhoz2)
        #print('ut_phi2 + rhoz2 - y2 is', u_pT_t_rz2)
        lambda_T2 = tf.matmul(u_pT_t_rz2, ppTrI_inv2)
        #print('lambda k+1 is',lambda_T2)
        
        lambda_T = np.vstack([lambda_T1[0,:], lambda_T2[1,:]])

        #____________to update z for frac2, using lambda_T____

        frac1 = np.array([α1/rho])
        frac3 = (1/rho * y_coeff1) + lambda_T
        S1 = pywt.threshold(frac3, frac1 , 'soft')  #frac2:our signal; frac1: value to threshold
        #print('Softthresholding z is', S)
                
        z_T3 = S1
        
        frac2 = np.array([α2/rho])
        frac4 = (1/rho * y_coeff2) + lambda_T
        S2 = pywt.threshold(frac4, frac2 , 'soft')  #frac2:our signal; frac1: value to threshold
        #print('Softthresholding z is', S)
        
        z_T4 = S2

#_________to update yij, carry forward using lambda_T _______
        y_coeff1 = y_coeff1 + rho*(lambda_T1 - S1)
        #print('yij_k+1 is', y_coeff)     
        
        y_coeff2 = y_coeff2 + rho*(lambda_T2 - S2)
        #print('yij_k+1 is', y_coeff)     

        #print('lambda_matrix k+1 transpose is', lambda_T)
        #print('z_k+1 is',z)
        #lambda_T = prox(lambda_T, sigma)
        #lu_piv_ten = np.array(lu_piv).reshape(6,6)
        #lam_T_LU = tf.matmul(u_pT_t_rz, lu_piv_ten)
        #print('lambda LU is',lam_T_LU)
    
    
    #--------------cal of loss
    #L = min 1/2 ||lambda_T*phi - ddt||^2 + lam ||z|| + sum y_ij(lambda_T-z_T) + rho/2 ||lambda_T-z_T||^2
    
        y_predict1 = lambda_T[0:1,:] @ array1 
        f1 = np.sum( (pinn_dxdt - y_predict1)**2 )
        g1 = np.abs(lambda_T1).sum()
        h1 = np.sum(y_coeff1*(lambda_T1-z_T3[0,:]))
        i1 = np.sum((lambda_T1-z_T3[0,:])**2)
        loss1 = 0.5*f1 + α1*g1 + h1 + rho*0.5*i1
        
        losses1.append(loss1)
        
        y_predict2 = lambda_T[1:1,:] @ array2 
        f2 = np.sum( (pinn_dydt - y_predict2)**2 )
        g2 = np.abs(lambda_T2).sum()
        h2 = np.sum(y_coeff2*(lambda_T2-z_T4[1,:]))
        i2 = np.sum((lambda_T2-z_T4[1,:])**2)
        loss2 = 0.5*f2 + α2*g2 + h2 + rho*0.5*i2
        
        losses2.append(loss2)
        
        #L = loss1 + loss2
        #print('total loss is', L)
    
    #XX = np.c_[XX1,XX2]
    y_pred3 = tf.transpose(array1) @ tf.transpose(lambda_T)
    y_pred4 = tf.transpose(array2) @ tf.transpose(lambda_T)
    f_de = np.sum((ddt-y_pred3)**2) + np.sum((ddt - y_pred4)**2)
    f_data = losses1[-1] + losses2[-1]
    print('Data loss is', f_data)
    print('Derivative loss is', f_de)
    
    fig,ax2 = plt.subplots(1,1)
    #ax2.plot(losses1, label = 'losses of dxdt')
    ax2.semilogy(losses1, label='log losses of dxdt')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("loss value")
    ax2.legend(loc='upper right')
    fig.savefig('losses_dxdt_admm.png', dpi=300)
    
    fig,ax3 = plt.subplots(1,1)
    #ax3.plot(losses2, label = 'losses of dydt')
    ax3.semilogy(losses2, label='log losses of dydt')
    ax3.set_xlabel("Epoch")
    ax3.legend(loc='upper right')
    fig.savefig('losses_dydt_admm.png', dpi=300)

    print('Coeff matrix is', lambda_T)
    
    libdata['z_T3'] = z_T3
    libdata['z_T4'] = z_T4
    libdata['y_ij1'] = y_coeff1
    libdata['y_ij2'] = y_coeff2
    libdata['lambda_T1'] = lambda_T1
    libdata['lambda_T2'] = lambda_T2
    libdata['lambda_T'] = lambda_T
    
    #y_pred3 and y_pred4 to check whether same as u_t
   
    return lambda_T

def func_opt(libdata):
     
    iv1 = libdata['iv1']
    iv2 = libdata['iv2']
    #upred_ddt = libdata['ddt']
    ddt = libdata['der_spline']
    
    dxdt = ddt[:,0]
    dydt = ddt[:,1]
    XX1 = np.c_[iv1] 
    XX2 = np.c_[iv2] 
    
    #-----------functional
    beta1 = em_mlr(XX1, dxdt)
    print(f"Estimated dxdt Coefficients: {beta1}")
 
    beta2 = em_mlr(XX2, dydt)
    print(f"Estimated dydt Coefficients: {beta2}")  
    
    libdata['beta1'] = beta1
    libdata['beta2'] = beta2
    
    return beta1, beta2


def get_func_sol(libdata):
    beta1 = libdata['beta1']
    beta2 = libdata['beta2']
    
    def func_solve(t, x):
        #dxdt = lambda_T[0,4]*x[1] + lambda_T[0,2]*(x[0]*x[0]) + lambda_T[0,3]*(x[0]**4) + lambda_T[0,7]*((x[0]**2)*(x[1]**2)) 
        #dydt = lambda_T[1,1]*x[0] + lambda_T[1,4]*x[1]+lambda_T[1,6]*((x[0]**2)*x[1]) + lambda_T[1,5]*(x[1]**3)     
        
        y1 = beta1[0] *x[0] + beta1[1]*(x[0]**2) + beta1[2]*x[1] + beta1[3]*(x[1]**2) + beta1[4]*(x[0]*x[1]) 
        #+ beta1[6]*(x[0]**3) + beta1[7]*(x[1]**3) + beta1[8]*((x[0]**2)*x[1]) + beta1[9]*((x[0])*(x[1]**2))
        y2 = beta2[0] *x[0] + beta2[1]*(x[0]**2) + beta2[2]*x[1] + beta2[3]*(x[1]**2) + beta2[4]*(x[0]*x[1])
        #+ beta2[6]*(x[0]**3) + beta2[7]*(x[1]**3) + beta2[8]*((x[0]**2)*x[1]) + beta2[9]*((x[0])*(x[1]**2))
                   
        return np.array([y1, y2])
     
    sol_func = scint.solve_ivp(fun=func_solve, t_span=(0,t_end), y0=[25,31], method="RK45", t_eval=np.linspace(0,t_end,N))
         
    libdata['sol_func'] = sol_func
     
    return sol_func

def em_algorithm_for_functional_regression(Phi, y, num_iter=100, tol=1e-6):
    """
    Estimate the coefficients of a functional regression model using the EM algorithm.
    
    :param Phi: Basis function matrix (functional predictors evaluated at discrete points)
    :param y: Dependent variable (vector)
    :param num_iter: Number of iterations
    :param tol: Convergence tolerance
    :return: Estimated coefficients (vector)
    """
    n, p = Phi.shape
    beta = np.random.randn(p)
    eps = 1e-8
    
    for _ in range(num_iter):
        # Prediction
        y_pred = Phi @ beta
        
        # Residuals
        r = y - y_pred
        
        # Estimate variance
        sigma2 = np.mean(r**2) + eps
        
        # E-step (weights from Gaussian likelihood)
        weights = np.exp(-0.5 * (r**2) / sigma2)
        weights /= (np.sum(weights) + eps)
        
        # M-step (weighted least squares)
        W = np.diag(weights)
        A = Phi.T @ W @ Phi + 1e-6 * np.eye(p)
        b = Phi.T @ W @ y
        
        beta_new = solve(A, b)
        
        if np.linalg.norm(beta_new - beta) < tol:
            break
        
        beta_est_em = beta_new
    
    return beta_est_em

def EM_func_model(libdata):
    iv1 = libdata['iv1']
    iv2 = libdata['iv2']
    
    ddt = libdata['der_spline']
    
    dxdt = ddt[:,0]
    dydt = ddt[:,1]
   
    
    Phi1 = np.c_[iv1] 
    Phi2 = np.c_[iv2] 
    
    beta3 = em_algorithm_for_functional_regression(Phi1, dxdt)
    print(f"Estimated Functional dxdt Coefficients: {beta3}")
    
    beta4 = em_algorithm_for_functional_regression(Phi2, dydt)
    print(f"Estimated Functional dydt Coefficients: {beta4}")
    
    libdata['beta3'] = beta3
    libdata['beta4'] = beta4
    
    
    return beta3, beta4

def get_func_em_sol(libdata):
    beta3 = libdata['beta3']
    beta4 = libdata['beta4']
    
    def func_em_solve(t, x):
        #dxdt = lambda_T[0,4]*x[1] + lambda_T[0,2]*(x[0]*x[0]) + lambda_T[0,3]*(x[0]**4) + lambda_T[0,7]*((x[0]**2)*(x[1]**2)) 
        #dydt = lambda_T[1,1]*x[0] + lambda_T[1,4]*x[1]+lambda_T[1,6]*((x[0]**2)*x[1]) + lambda_T[1,5]*(x[1]**3)     
        
        y1 = beta3[0] *x[0] + beta3[1]*(x[0]**2) + beta3[2]*x[1] + beta3[3]*(x[1]**2) + beta3[4]*(x[0]*x[1]) 
        #+ beta3[6]*(x[0]**3) + beta3[7]*(x[1]**3) + beta3[8]*((x[0]**2)*x[1]) + beta3[9]*((x[0])*(x[1]**2))
        y2 = beta4[0] *x[0] + beta4[1]*(x[0]**2) + beta4[2]*x[1] + beta4[3]*(x[1]**2) + beta4[4]*(x[0]*x[1])
        #+ beta4[6]*(x[0]**3) + beta4[7]*(x[1]**3) + beta4[8]*((x[0]**2)*x[1]) + beta4[9]*((x[0])*(x[1]**2))
               
        return np.array([y1, y2])
     
    sol_func_em = scint.solve_ivp(fun=func_em_solve, t_span=(0,t_end), y0=[25,31], method="RK45", t_eval=np.linspace(0,t_end,N))
     
    
    libdata['sol_func_em'] = sol_func_em
     
    return sol_func_em

    # mask3 = np.abs(beta3) > sigma
    # mask_dx3 = beta3*mask3 
    # #print('dxdt coeff is', mask_dx1)
    
    # mask4 = np.abs(beta4) > sigma
    # mask_dy4 = beta4*mask4
    # #print('dydt coeff is', mask_dy2)
    
    # lambda_func1 = tf.transpose(np.c_[mask_dx1,mask_dy2])
    
    # #------------------------------------------
    # mask3 = np.abs(lambda_LU[0,:]) > sigma
    # mask_dx3 = lambda_LU[0,:]*mask3 
    # #print('dxdt coeff is', mask_dx3)
    
    # mask4 = np.abs(lambda_LU[1,:]) > sigma
    # mask_dy4 = lambda_LU[1,:]*mask4 
    # #print('dydt coeff is', mask_dy4)
    
    # lambda_func2 = tf.transpose(np.c_[mask_dx3,mask_dy4])
    # #-------------------------------------------------------
    
    
 
    # print('Sparse Coeff Matrix (Mp) is', lambda_func1)
    # print('Sparse Coeff Matrix (SI) is', lambda_func2)
    
    # libdata['lambda_sparse1'] = lambda_func1
    # libdata['lambda_sparse2'] = lambda_func2
    
    # return lambda_sparse1, lambda_sparse2

    
def get_admm_sol(libdata):
    
    lambda_T = libdata['lambda_T']
    
    def func_check(t, x):
        #dxdt = lambda_T[0,4]*x[1] + lambda_T[0,2]*(x[0]*x[0]) + lambda_T[0,3]*(x[0]**4) + lambda_T[0,7]*((x[0]**2)*(x[1]**2)) 
        #dydt = lambda_T[1,1]*x[0] + lambda_T[1,4]*x[1]+lambda_T[1,6]*((x[0]**2)*x[1]) + lambda_T[1,5]*(x[1]**3)     
       
        y1 = lambda_T[0,0] *x[0] + lambda_T[0,1]*(x[0]**2) + lambda_T[0,2]*x[1] + lambda_T[0,3]*(x[1]**2) + lambda_T[0,4]*(x[0]*x[1]) 
        #+ lambda_T[0,6]*(x[0]**3) + lambda_T[0,7]*(x[1]**3) + lambda_T[0,8]*((x[0]**2)*x[1]) + lambda_T[0,9]*((x[0])*(x[1]**2))
        y2 = lambda_T[1,0] *x[0] + lambda_T[1,1]*(x[0]**2) + lambda_T[1,2]*x[1] + lambda_T[1,3]*(x[1]**2) + lambda_T[1,4]*(x[0]*x[1])
        #+ lambda_T[1,6]*(x[0]**3) + lambda_T[1,7]*(x[1]**3) + lambda_T[1,8]*((x[0]**2)*x[1]) + lambda_T[1,9]*((x[0])*(x[1]**2))
        
        #-------------lasso
        #y1 = -0.00426988 - 0.09552082*x[0] + 0.83130978*x[1]  
        #y2 = -0.00959996 - 0.641325*x[0] - 0.37711383*x[1]  
        
        # ------------------------- admm
        #y1 = 0.8313*x[1]  
        #y2 = 0.00118979*(x[0]*x[0]) + 0.0114018*(x[0]*x[1])  
        
        #array = tf.stack([const_one,x,x2,x4,y,y3,x2y,x2y2],axis=0)
        #dxdt = -x[1] + miu*(x[0]**2)*(1-((x[0]**2)+(x[1]**2)))      
            # = -y + miu*x^2 - miu*x^4 - miu*x^2*y^2
        #dydt = x[0] + miu*x[1]*(1-((x[0]**2)+(x[1]**2))) 
            # = x + miu*y - miu*x^2*y - miu*y^3
        #print(np.array([y1,y2]))
        
        return np.array([y1, y2])
    
    sol_lambda = scint.solve_ivp(fun=func_check, t_span=(0,t_end), y0=[25,31], method="RK45", t_eval=np.linspace(0,t_end,N))
    
    libdata['sol_lambda'] = sol_lambda
    
    return sol_lambda

def get_LU_sol(libdata):
    
    lams = libdata['lambda_LU']
    
    def func_check(t, x):
        #dxdt = lambda_T[0,4]*x[1] + lambda_T[0,2]*(x[0]*x[0]) + lambda_T[0,3]*(x[0]**4) + lambda_T[0,7]*((x[0]**2)*(x[1]**2)) 
        #dydt = lambda_T[1,1]*x[0] + lambda_T[1,4]*x[1]+lambda_T[1,6]*((x[0]**2)*x[1]) + lambda_T[1,5]*(x[1]**3)     
       
        y1 = lams[0,0] *x[0] + lams[0,1]*(x[0]**2) + lams[0,2]*x[1] + lams[0,3]*(x[1]**2) + lams[0,4]*(x[0]*x[1]) #+ lams[0,6]*(x[0]**3) + lams[0,7]*(x[1]**3) + lams[0,8]*((x[0]**2)*x[1]) + lams[0,9]*((x[0])*(x[1]**2))
        y2 = lams[1,0] *x[0] + lams[1,1]*(x[0]**2) + lams[1,2]*x[1] + lams[1,3]*(x[1]**2) + lams[1,4]*(x[0]*x[1]) #+ lams[1,6]*(x[0]**3) + lams[1,7]*(x[1]**3) + lams[1,8]*((x[0]**2)*x[1]) + lams[1,9]*((x[0])*(x[1]**2))
                
        return np.array([y1, y2])
    
    sol_LU = scint.solve_ivp(fun=func_check, t_span=(0,t_end), y0=[25,31], method="RK45", t_eval=np.linspace(0,t_end,N))
    
    libdata['sol_LU'] = sol_LU

    return sol_LU

def get_lasso_sol(libdata):
    
    w1 = libdata['w1']
    w2 = libdata['w2']
    
    w = np.c_[w1,w2]
    
    def func_check(t, x):
        
        #---------------sparse----------
        y1 = w[0,0] *x[0] + w[1,0]*(x[0]**2) + w[2,0]*x[1] + w[3,0]*(x[1]**2) + w[4,0]*(x[0]*x[1]) #+ w[6,0]*(x[0]**3) + w[7,0]*(x[1]**3) + w[8,0]*((x[0]**2)*x[1]) + w[9,0]*((x[0])*(x[1]**2))
        y2 = w[0,1] *x[0] + w[1,1]*(x[0]**2) + w[2,1]*x[1] + w[3,1]*(x[1]**2) + w[4,1]*(x[0]*x[1]) #+ w[6,1]*(x[0]**3) + w[7,1]*(x[1]**3) + w[8,1]*((x[0]**2)*x[1]) + w[9,1]*((x[0])*(x[1]**2))
        
        #---------------lasso----------
        #y1 = -3.0794e-5 + -4.9602e-1*x[0] + 9.9829e-1*x[1] + -8.1047e-9*(x[0]**2) + -5.1425*(x[1]**2) -7.5256e-9*(x[0]*x[1]) 
        #y2 = -10.9795 -26.5780*x[0] - 59.4701*x[1] + 532.9849*(x[0]**2) + 293.8160*(x[1]**2) + 265.4348*(x[0]*x[1])
       
        return np.array([y1, y2])
    
    sol_lasso = scint.solve_ivp(fun=func_check, t_span=(0,t_end), y0=[25,31], method="RK45", t_eval=np.linspace(0,t_end,N))
    
    # fig,ax2 = plt.subplots(1,1)
    # ax2.plot(output[:,0], output[:,1],'x', label = 'RK45')    #(100,2)
    # ax2.legend(loc='upper left')
    # ax2.plot(sol_sparse.y[0,:],sol_sparse.y[1,:],'x', label='ADMM')
    # ax2.set_xlabel('x')#, fontdict=font)
    # ax2.set_ylabel('y')#, fontdict=font)
    # ax2.legend(loc='upper left')
    
    libdata['sol_lasso'] = sol_lasso

    return sol_lasso

def gradcheck(libdata):
    lambda_T1 = libdata['lambda_T1']
    lambda_T2 = libdata['lambda_T2']
    phi1 = libdata['phi1']
    phi2 = libdata['phi2']
    
    end_dx = np.array(lambda_T1[0,:]).reshape(1,-1) @ phi1
    end_dy = np.array(lambda_T2[1,:]).reshape(1,-1) @ phi2
    
    libdata['end_dx'] = end_dx
    libdata['end_dy'] = end_dy

def visualize(libdata):
    t = libdata['t_given']
    output = libdata['output']
    t_pred = libdata['t_pred']
    u_pred = libdata['u_pred']
    ori_x = libdata['x_given']
    ori_y = libdata['y_given']
    
    #t_spline = libdata['t_spline']
    out_spline = libdata['out_spline']
    der_spline = libdata['der_spline']
    
   # pinn_dx = libdata['pinn_dx']
    #pinn_dy = libdata['pinn_dy']

    
    # x_vec = libdata['x_vec']
    # y_vec = libdata['y_vec']
    # output_noise = libdata['output_noise']
   
    # #losses_pinn = lib_data['losses_pinn']
    # losses_sparse = libdata['losses_sparse']
    # term1_tra = libdata['term1_tra']
    # term2_lamz = libdata['term2_lamz']
    # y_lam_changes = libdata['y_lam_changes']
    # rho_lam_change = libdata['rho_lam_change']
    # L_grad = libdata['L_grad']
    
    sol_func = libdata['sol_func']
    sol_func_em = libdata['sol_func_em']
    sol_lambda = libdata['sol_lambda']
    sol_LU = libdata['sol_LU']
    sol_lasso = libdata['sol_lasso']
    # output = libdata['output']
    
    end_dx = libdata['end_dx']
    end_dy = libdata['end_dy']
    
    
    t_spline = np.linspace(tf.reduce_min(t), tf.reduce_max(t), N)
    
    #-------------verification
    fig,ax3 = plt.subplots(1,1)
    ax3.plot(output[:,0], output[:,1],'x', label = 'Data(RK45)')
    ax3.plot(u_pred[:,0], u_pred[:,1], label = 'Data(PINNs)')
    ax3.plot(out_spline[:,0], out_spline[:,1], label = 'Data(Spline)')
    #ax3.set_title('Data vs PINNs')
    ax3.legend(loc='upper left')
    ax3.set_xlabel('x', fontsize=12)#, fontdict=font)
    ax3.set_ylabel('y', fontsize=12)#, fontdict=font)
    ax3.legend(loc='upper left')
    ax3.tick_params(axis='both', labelsize=12)
    fig.savefig('ground_inter_all3.png', dpi=300)
    
    fig,ax4 = plt.subplots(1,1)
    ax4.plot(t, output[:,0], 'x', label = 'Data(RK45)')
    ax4.plot(t_pred, u_pred[:,0], label = 'Data(PINNs)')
    ax4.plot(t_spline, out_spline[:,0], label = 'Data(Spline)')
    #ax4.set_title('trained_ori_x')
    ax4.legend(loc='upper left')
    ax4.set_xlabel('Time, t')#, fontdict=font)
    ax4.set_ylabel('x-coordinate')#, fontdict=font)
    ax4.legend(loc='upper left')
    fig.savefig('trained_x_all3.png', dpi=300)
    
    fig,ax5 = plt.subplots(1,1)
    ax5.plot(t, output[:,1],'x', label = 'Data(RK45)')
    ax5.plot(t_pred, u_pred[:,1], label = 'Data(PINNs)')
    ax5.plot(t_spline, out_spline[:,1], label = 'Data(Spline)')
    #ax5.set_title('trained_ori_y')
    ax5.legend(loc='upper left')
    ax5.set_xlabel('Time, t')#, fontdict=font)
    ax5.set_ylabel('y-coordinate')#, fontdict=font)
    ax5.legend(loc='upper left')  
    fig.savefig('trained_y_all3.png', dpi=300)
    
    #-------------------mlr
    fig,ax6 = plt.subplots(1,1)
    ax6.plot(output[:,0], output[:,1],'x', label = 'Data(RK45)')    #(100,2)
    ax6.legend(loc='upper left')
    ax6.plot(sol_func.y[0,:],sol_func.y[1,:], label='Data(MpULFR-0)')
    #ax6.set_title('Data vs IPINNs')
    ax6.set_xlabel('x', fontsize=12)#, fontdict=font)
    ax6.set_ylabel('y', fontsize=12)#, fontdict=font)
    ax6.legend(loc='upper left')
    ax6.tick_params(axis='both', labelsize=12)
    fig.savefig('ground mlr.png', dpi=300)
    
    #-------------------em
    fig,ax7 = plt.subplots(1,1)
    ax7.plot(output[:,0], output[:,1],'x', label = 'Data(RK45)')    #(100,2)
    ax7.legend(loc='upper left')
    ax7.plot(sol_func_em.y[0,:],sol_func_em.y[1,:], label='MpULFR-M')
    #ax7.set_title('Data vs IPINNs')
    ax7.set_xlabel('x', fontsize=12)#, fontdict=font)
    ax7.set_ylabel('y', fontsize=12)#, fontdict=font)
    ax7.legend(loc='upper left')
    ax7.tick_params(axis='both', labelsize=12)
    fig.savefig('ground Mp.png', dpi=300)
    
    #-------------------lasso
    fig,ax8 = plt.subplots(1,1)
    ax8.plot(output[:,0], output[:,1],'x', label = 'Data(RK45)')    #(100,2)
    ax8.legend(loc='upper left')
    ax8.plot(sol_lasso.y[0,:],sol_lasso.y[1,:], label='Data(lasso)')
    #ax8.set_title('Data vs IPINNs')
    ax8.set_xlabel('x', fontsize=12)#, fontdict=font)
    ax8.set_ylabel('y', fontsize=12)#, fontdict=font)
    ax8.legend(loc='upper left')
    ax8.tick_params(axis='both', labelsize=12)
    fig.savefig('ground lasso.png', dpi=300)
    
    #-------------------lambda_T
    fig,ax9 = plt.subplots(1,1)
    ax9.plot(output[:,0], output[:,1],'x', label = 'Data(RK45)')    #(100,2)
    ax9.legend(loc='upper left')
    ax9.plot(sol_lambda.y[0,:],sol_lambda.y[1,:], label='Data(IPINNs)')
    #ax9.set_title('Data vs IPINNs')
    ax9.set_xlabel('x', fontsize=12)#, fontdict=font)
    ax9.set_ylabel('y', fontsize=12)#, fontdict=font)
    ax9.legend(loc='upper left')
    ax9.tick_params(axis='both', labelsize=12)
    fig.savefig('ground_ip.png', dpi=300)
    
    #---------------LU
    fig,ax10 = plt.subplots(1,1)
    ax10.plot(output[:,0], output[:,1],'x', label = 'Data(RK45)')    #(100,2)
    ax10.legend(loc='upper left')
    ax10.plot(sol_LU.y[0,:],sol_LU.y[1,:], label='Data(IPINNs)')
    #ax10.set_title('Data vs IPINNs')
    ax10.set_xlabel('x', fontsize=12)#, fontdict=font)
    ax10.set_ylabel('y', fontsize=12)#, fontdict=font)
    ax10.legend(loc='upper left')
    ax10.tick_params(axis='both', labelsize=12)
    fig.savefig('ground LU.png', dpi=300)
    

    
def save_output(libdata, fname="ground_inter_admm.pkl"):
    with open(fname, 'bw') as f:
        pickle.dump(libdata,f)

def main():
    libdata = make_data()
    
    model = loading_model(libdata)
    prepare_data(libdata)
    
    lass_fit(libdata)
    
    sparse_LU(libdata)
    sparse(libdata)
    
    func_opt(libdata)
    get_func_sol(libdata)
    
    EM_func_model(libdata)
    get_func_em_sol(libdata)
    
    # various_opt(libdata)
    
    get_admm_sol(libdata)
    get_LU_sol(libdata)
    get_lasso_sol(libdata)
    
    # gradcheck(libdata)
    visualize(libdata)
    
    save_output(libdata, fname="ground_inter_admm.pkl")
    
    elapsed = time.time() - start_time 
    print('Training time: %.2f' % (elapsed))
    
    return libdata, model
    
libdata, model = main()




