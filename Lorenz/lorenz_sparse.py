# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 15:15:24 2025

"""
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy as sp
import pywt
import matplotlib.pyplot as plt
import scipy.integrate as scint
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense #, Lambda
from keras.saving import load_model
from tensorflow.keras import backend as K
from scipy.optimize import minimize
from box import Box
from scipy.linalg import solve

import pickle

from sklearn.linear_model import Lasso

epochs = 20
batch_size = 4
#lamda = 1e-2   #equal to α2 set
rho = ρ = tf.constant(5, dtype=tf.float64)
learn_rate = -2e-6
losses_sparse = []

tau = 2
N=200

miu = -0.3
row = 4
col = 1

α = 1 #np.sqrt(5) / 2

sigma = 1e-3
sigma1 = 1e-10 #proximal 
sigma2 = 5e-5

α1 = 0.001 #lasso weight search for dxdt
α2 = 0.001 #lasso weight search for dydt 
α3 = 0.001 #lasso weight search for dydt 

αf1 = 10000 #loss func cal for dxdt
αf2 = 1e-4 #loss func cal for dydt
αf3 = 1e-3 #loss func cal for dxdt

sigma, beta, rho_lo = 10.0, 8.0/3.0, 28.0 # Standard Lorenz parameters
y0=[0.96, -1.1, 0.5]

#data = pd.read_csv("ode_dataRHS_nonlinear.csv")
#train_data = tf.convert_to_tensor(data.values)



# model_pkl_file = "pinn_model.pkl" 
# with open(model_pkl_file, 'rb') as file:  
#     model = pickle.load(file)

def make_data():    
    with open ('lorenz.pkl','rb') as f:
        pdata = pickle.load(f)
       
    # global data
    libdata ={
        
           't' : pdata.t,
           'x' : pdata.u_test[:,0],
           'y' : pdata.u_test[:,1],
           'z_lo' : pdata.u_test[:,2],
           
           'u_test': pdata.u_test,
           
           'output':[],
           't_vec':[],
           'ddt' : [],
           
           
            'w1':[],
            'w2':[],
            'w_sparse1':[],
            'w_sparse2':[],
            'sol_sparse':[],
           
            #'model': pdata.model, 
            'fname':None,
           
            #'loss_':[],
            #'loss_ic':[],
            #'loss_f':[],
            #'error_vec':[],
          
           
            'sparse_u_pinn' : [],
           
            'iv' :[],
            'identity' : [],
            'phi' : [],
            'lambda_T' :[],
            'y_ij' : [],
            'z_T' : [],
            'lambda_LU' : [],
            'sol_LU' :[],
            'lambda_sparse':[],
           
           
           
            'lambda_phi': [],
            'y_lam' : [],
            'term1_tra' : [],
            'term2_lamz' :[],
            'y_lam_changes' : [],
            'rho_lam_change' : [],
            'L_grad' : [],
           
            'sol_lambda' : [],
            #'u_test': [],
            #'numerical_rhs':[],
            #'grads':[],
            'losses':[],
            'loss_sparse':[],
            'losses_sparse':[],
           
           }
    return Box(libdata)

def RHS(t, X, sigma, beta, rho_lo):
    x, y, z_lo = X
    
    dxdt = sigma * (y - x)
    dydt = x * (rho_lo - z_lo) - y
    dzdt = x * y - beta * z_lo
    
    #x, y, z = xyz
    #x_dot = s*(y - x)
    #y_dot = r*x - y - x*z
    #z_dot = x*y - b*z
    
    return np.array([dxdt, dydt, dzdt])

def prepare_ddt(libdata):
    t = libdata['t']
    x = libdata['x']
    y = libdata['y']
    z_lo = libdata['z_lo']
    
    u_test = libdata['u_test']   #(1000,3)
    
    t_vec = np.array(t).reshape(-1,1)
    u_test_vec = tf.transpose(u_test)    #(3,1000)
    
    ddt = RHS(t_vec,u_test_vec, sigma, beta, rho_lo)
    #print('ddt', ddt)

    libdata['ddt'] = ddt
    libdata['t_vec'] = t_vec
    
    return ddt

def pinn(libdata):
    t_vec = libdata['t_vec']
    ddt = libdata['ddt']
    
    dxdt_vec = ddt[0,:]
    dydt_vec = ddt[1,:]
    dzdt_vec = ddt[2,:]
    #dxdt_vec = np.array(dxdt_given).reshape(-1,1)
    #dydt_vec = np.array(dydt_given).reshape(-1,1)
    
    x = libdata['x']
    y = libdata['y']
    z_lo = libdata['z_lo']
    output = np.c_[x,y,z_lo]
    
    def custom_loss(y_true, y_pred):
        #to_take = sampling + N_f - 5
        #loss_ic = K.mean(K.square(y_true[:to_take] - y_pred[:to_take]))
        loss_ic = K.mean(K.square(y_true - y_pred))

        g = tf.convert_to_tensor(t_vec)
        #print('g shape is', g.shape)  #(100x1)
 
        with tf.GradientTape(persistent=False) as tape:
            tape.watch(g)
            
            u_eval = model(g)
            #print(u_eval)
            grads = tape.jacobian(u_eval, g) 
            #print(grads)
            grads_select = tf.einsum('bxby->bxy',grads) #only one entry, the rest are zero
            #print(grads_select)
            grads_final = grads_select[:,:]
            #print(grads_final)  #(100,2,1)
     
            dx_dt = grads_final[:,0]
            #print('dxdt is', dx_dt)
            dy_dt = grads_final[:,1]
            #print('dydt is', dy_dt)
            dz_dt = grads_final[:,2]
             
            f1 = dx_dt - dxdt_vec 
            #print('dxdt loss is', f1)
            f2 = dy_dt - dydt_vec
            #print('dydt loss is', f2)
            f3 = dz_dt - dzdt_vec
     
        #ddt_pinn = Concatenate()([dx_dt,dy_dt])
     
        loss_f = tf.reduce_mean(tf.square(f1)+tf.square(f2)+tf.square(f3))

        loss = loss_ic + loss_f
     
        #tf.print('loss on ic is:', loss_ic)
        #tf.print('loss on derivative(GTape) is:', loss_f)
        tf.print('total loss is:', loss)
     
        return loss
 
    model = load_model('lorenz_model.keras',custom_objects={'custom_loss': custom_loss})
    model.summary()

    t_to_array = tf.constant(t_vec)
    #y_size = tf.transpose(output)  #(100,2)
    y_to_fit = tf.constant(output)
    history = model.fit(t_to_array, y_to_fit, batch_size,epochs)   #should be respective t on train set
  
    #loss_pinn is decreasing?
    fig,ax1 = plt.subplots(1,1,figsize=(8, 4.5))
    ax1.plot(history.history['loss'], label = 'losses of PINNs')
    #ax3.semilogy(losses_pinn, label = 'log_losses of PINNs')  
    ax1.set_xlabel('number of iteration')#, fontdict=font)
    ax1.set_ylabel('loss value')#, fontdict=font)
    ax1.legend(loc='upper right')
    fig.savefig('losses_pinns.png', dpi=300)
  
    sparse_u_pinn = model(t_vec)
    
    libdata['sparse_u_pinn'] = sparse_u_pinn
    libdata['output'] = output
    
    return sparse_u_pinn

def prepare_data(libdata):
    
    x = libdata['x']
    y = libdata['y']
    z_lo = libdata['z_lo']
    
    const_one = tf.ones([300,])
    const_one = tf.cast(const_one,dtype=tf.float64)
    
    x2 = tf.square(x)
    y2 = tf.square(y)
    
    xy = tf.multiply(x,y)
  
    identity = tf.eye(row,dtype=tf.float64)  
           
    #array = tf.stack([x,x_sqr,y_cube,x2y2],axis=0)
    #array = tf.stack([const_one,x,y],axis=0)
    array = tf.stack([const_one,x,y,z_lo],axis=0)
    #print(array)   #(5x100)
    iv = np.c_[x,y,z_lo]
    
    #array_var_T = tf.transpose(array)
    #print(array_var_T)
    
    u_vec = tf.ones([row,col])
    v_vec = tf.ones([row,col])
    lammda = np.c_[u_vec,v_vec]
    lambda_T = lammda.T   #dtype:float32
    lambda_T = tf.cast(lambda_T,dtype=tf.float64) #convert tensor to tensor
    #print(matrix_T)
    y_coeff = tf.cast(tf.ones([2,row])/10,dtype=tf.float64)
    z_T = tf.convert_to_tensor(np.ones_like(lambda_T), dtype=tf.float64)
    #z_T = tf.convert_to_tensor(np.random.rand(2,6), dtype=tf.float64)
 
    libdata['iv'] = iv
    libdata['identity'] = identity
    libdata['phi'] = array
    libdata['lambda_T'] = lambda_T
    libdata['y_ij'] = y_coeff
    libdata['z_T'] = z_T

#--------lasso fitting to get initial coefficient matrix------
def lass_fit(libdata):
    #given_xy = las_data['output']
    ddt = libdata['ddt']
    x = libdata['x']
    y = libdata['y']
    z_lo = libdata['z_lo']
    iv = libdata['iv']
    
    #x2 = tf.square(x)
    #y2 = tf.square(y)   
    #xy = tf.multiply(x,y)
    
    #iv = np.c_[x,y, x2, y2, xy]
    
    dxdt = ddt[0,:]
    
    lasso1 = Lasso(alpha=α1, fit_intercept=True, tol=1e-4, max_iter=1000)
    lasso1.fit(iv,dxdt)
    w1 = np.array(list(lasso1.coef_) + [lasso1.intercept_])
    print('lasso for y1(dxdt) is', w1)
    #print('lasso loss for y1(dxdt) is',0.5*sum((lasso1.predict(iv)-dxdt)**2) + 1*sum(np.abs(w1)))
    
    def fun1(w1, α1=α1):
        
        #iv = np.c_[x,y, x2, y2, xy]
        
        dxdt = ddt[0,:]
        #n=100
        
        XX1 = np.c_[iv, np.ones_like(dxdt)]
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
    
    
    dydt = ddt[1,:]
    
    lasso2 = Lasso(alpha=α2, fit_intercept=True, tol=1e-4, max_iter=1000)
    lasso2.fit(iv,dydt)
    w2 = np.array(list(lasso2.coef_) + [lasso2.intercept_])
    print('lasso for y2(dydt) is',w2)
    #print('lasso loss for y2(dydt) is',0.5*sum((lasso2.predict(iv)-dydt)**2) + 1*sum(np.abs(w2)))
    
    def fun2(w2, α2=α2):
        
        dydt = ddt[1,:]
        
        XX2 = np.c_[iv, np.ones_like(dydt)]
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
    
    
    dzdt = ddt[2,:]
    
    lasso3 = Lasso(alpha=α3, fit_intercept=True, tol=1e-4, max_iter=1000)
    lasso3.fit(iv,dydt)
    w3 = np.array(list(lasso3.coef_) + [lasso3.intercept_])
    print('lasso for z(dzdt) is',w3)
    #print('lasso loss for y2(dydt) is',0.5*sum((lasso2.predict(iv)-dydt)**2) + 1*sum(np.abs(w2)))
    
    def fun3(w3, α3=α3):
        
        dzdt = ddt[2,:]
        
        XX3 = np.c_[iv, np.ones_like(dzdt)]
        y_predict3 = XX3 @ w3 #* np.random.randn(n)
        f3 = np.sum( (dzdt - y_predict3)**2 )
        g3 = np.abs(w3).sum()
        loss2 = 0.5*f3 + αf3*g3
        
        return loss2
    
    dzdt_result = minimize(fun3, w3, tol=0.0001)
    #print(dxdt_result)
    coeffdz_min = dzdt_result.x
    print('Non-sparse coeff for dzdt',coeffdz_min)
    loss_mindy = dzdt_result.fun
    #print('loss_scipy_min_dydt',loss_mindy)
    
    libdata['w1'] = w1
    libdata['w2'] = w2
    libdata['w3'] = w3
    #libdata['iv'] = iv
    
    return w1, w2 #, iv

#---------------- em_mlr as initial--------
def em_mlr(X, y, num_iter=100, tol=1e-6):
    """
    Estimate the coefficients of a multiple linear regression model using the EM algorithm.
    
    :param X: Independent variables (matrix)
    :param y: Dependent variable (vector)
    :param num_iter: Number of iterations
    :param tol: Convergence tolerance
    :return: Estimated coefficients (vector)
    """
    n, p = X.shape
    
    # Initialize parameters
    beta_est = np.random.randn(p)
    variance = np.var(y)
    
    for _ in range(num_iter):
        # E-step: Compute expected values of latent variables (errors assumed to be Gaussian)
        y_pred = X @ beta_est
        responsibilities = np.exp(-0.5 * ((y - y_pred) ** 2) / variance)
        responsibilities /= np.sum(responsibilities)
        
        # M-step: Update estimates using weighted least squares
        W = np.diag(responsibilities)
        X_weighted = X.T @ W @ X
        y_weighted = X.T @ W @ y
        
        new_beta_est = np.linalg.solve(X_weighted, y_weighted)
        
        # Check for convergence
        if np.linalg.norm(new_beta_est - beta_est) < tol:
            break
        
        beta_est = new_beta_est
    
    return beta_est

#-----------------em_MpULFR-M as initial
def em_Mp(Phi, y, num_iter=100, tol=1e-6):
    """
    Estimate the coefficients of a functional regression model using the EM algorithm.
    
    :param Phi: Basis function matrix (functional predictors evaluated at discrete points)
    :param y: Dependent variable (vector)
    :param num_iter: Number of iterations
    :param tol: Convergence tolerance
    :return: Estimated coefficients (vector)
    """
    n, p = Phi.shape
    eps = 1e-12

    # Initialize parameters
    beta_est = np.random.randn(p)
    variance = np.var(y)
    
    for _ in range(num_iter):
        # E-step: Compute expected values of latent variables (errors assumed to be Gaussian)
        noise = np.random.normal(0, tau, N)
        y_pred = Phi @ beta_est + noise
        responsibilities = np.exp(-0.5 * ((y - y_pred) ** 2) / variance)
        responsibilities /= (np.sum(responsibilities) + eps)
        
        # M-step: Update estimates using weighted least squares
        W = np.diag(responsibilities)
        Phi_weighted = Phi.T @ W @ Phi
        y_weighted = Phi.T @ W @ y
        ridge = 1e-8 * np.eye(Phi_weighted.shape[0])
        
        new_beta_est = solve(Phi_weighted +ridge, y_weighted)
        
        # Check for convergence
        if np.linalg.norm(new_beta_est - beta_est) < tol:
            break
        
        beta_est = new_beta_est
    
    return beta_est


def sparse(libdata):
    iv = libdata['iv']
    array = libdata['phi']
    identity = libdata['identity']
    ddt = libdata['ddt']
    lambda_T = libdata['lambda_T']
    y_coeff = libdata['y_ij']
    z_T = libdata['z_T']
     
    x = libdata['x']
    y = libdata['y']
   
    
    term1_tra = libdata['term1_tra']
    term2_lamz = libdata['term2_lamz']

    y_lam = libdata['y_lam']
    y_lam_changes = libdata['y_lam_changes']
    rho_lam_change = libdata['rho_lam_change']
    L_grad = libdata['L_grad']
    
#____________to update lambda, z and y____
       
#--------------forming del_L_del_lambda to fine tune learning rate ---------
       # lam_phiphiT = tf.matmul(lambda_T, phi_phiT)
       # lam_z_diff = tf.subtract(lambda_T, z_T)
       # rho_lam_z = tf.multiply(rho, lam_z_diff)
       # del_l_del_lambda = lam_phiphiT - uarr + y_coeff + rho_lam_z
       # #print('del l del lambda', del_l_del_lambda)
       # lambda_T = lambda_T - tf.multiply(learn_rate, del_l_del_lambda)
#----------------end tuning of learning rate--------

    ut_phi = tf.matmul(ddt, tf.transpose(array))
    #print('u_t phi is', ut_phi)
    phi_phiT = tf.matmul(array, tf.transpose(array))  #phi: (6x100)
    #print('phiphiT',phi_phiT)
    rhoI = tf.multiply(rho,identity)
    #print('rhoI', rhoI)
    ppT_rI = tf.add(phi_phiT,rhoI)   
    #print('phi2^T + rhoI', ppT_rI)
    ppTrI_inv = tf.linalg.pinv(ppT_rI)
    #print('least square inverse is', ppTrI_inv)
       
       #-----------------------------------
    dxdt = ddt[0,:]
    dydt = ddt[1,:]
    dzdt = ddt[2,:]
    XX = np.c_[np.ones_like(dxdt),iv]    #(100,6)
    XX_ddt = ddt @ XX
    #print('XX_ddt is', XX_ddt)   #values equal to ut_phi
    XTX = XX.T @ XX 
    #print('XTX', XTX)   #values equal to phi_phiT
    #aa = ρ* np.eye( *XTX.shape)
    #print('ρI', aa)
    A = XTX + ρ* np.eye( *XTX.shape) 
    #print('LU_method A matrix', A)   #values equal ppT_rI
    invr = np.linalg.inv(A)
    #print('inverse', invr)
   
    lambda_LU = np.vstack((tf.ones([col,row]), tf.ones([col,row])))
    xi = np.ones_like(lambda_LU)/10
    #xi = np.ones_like(lambda_LU)
    z = np.ones_like(lambda_LU)

    losses1 = []
    losses2 = []
    losses3 = []
   
    
    for i in range (50):
  #_______to update lambda_T__
    
        subtra = tf.subtract(ut_phi, y_coeff)
        #print('ut_phi - y', subtra)
        rhoz = tf.multiply(rho,z_T)
        u_pT_t_rz = tf.add(subtra, rhoz)
        #print('ut_phi + rhoz - y is', u_pT_t_rz)c
        lambda_T = tf.matmul(u_pT_t_rz, ppTrI_inv)
        #print('lambda k+1 is',lambda_T)
        
       
 #------------------------------------
         #np.hstack((b_dx, b_dy))
       
        lu_piv = sp.linalg.lu_factor(A)
        #print('LU factorization', lu_piv)
        aa = XX_ddt -  xi
        #print('XX_ddt - y', aa)    #values same as subtra
        b_d = XX_ddt +  ρ*z - xi
        #print('b_d is', b_d)  #same as u_pT_t_rz
        lambda_dx = sp.linalg.lu_solve(lu_piv, b_d[0,:])
        lambda_dy = sp.linalg.lu_solve(lu_piv, b_d[1,:])
        lambda_dz = sp.linalg.lu_solve(lu_piv, b_d[2,:])
        lambda_LU = tf.transpose(np.c_[lambda_dx,lambda_dy, lambda_dz])
        #print('lambda_LU is', lambda_LU)
        
        
        
        # w = np.array([1,1,1,1,1,1])
        # xi = np.ones_like(w)/10
        # z = w.copy()
        # b_dx = XX.T @ dxdt +  ρ*(z - xi)
        # w1 = sp.linalg.lu_solve(lu_piv, b_dx) 
        # #b_d = np.vstack((dxdt, dydt)) @ XX +  ρ*(z - xi)
        # #w = sp.linalg.lu_solve(lu_piv, b_d) 
        # print('lambda_dxdt LU is', w1)
       
        # w2 = np.array([1,1,1,1,1,1])
        # xi = np.ones_like(w2)/10
        # z = w2.copy()
        # b_dy = XX.T @ dydt +  ρ*(z - xi)
        # w2 = sp.linalg.lu_solve(lu_piv, b_dy) 
        # print('lambda_dydt LU is', w2)
        
        #lambda_LU = np.c_[w1,w2]
        
       
        #____________to update z for frac2____


        frac1 = np.array([α2/rho])
        frac2 = (1/rho * y_coeff) + lambda_T
        S = pywt.threshold(frac2, frac1 , 'soft')  #frac2:our signal; frac1: value to threshold
        #print('Softthresholding z is', S)
        
        
        z = pywt.threshold(lambda_LU + xi/ρ , frac1 , 'soft') 
        #print('Soft z(LU) is', z)
        z_T = S

#_________to update yij_______
        y_coeff = y_coeff + rho*(lambda_T - S)
        #print('yij_k+1 is', y_coeff)     

        xi = xi + ρ*(lambda_LU - z)
        #print('x1 is', xi)
        
        #print('lambda_matrix k+1 transpose is', lambda_T)
        #print('z_k+1 is',z)
        #lambda_T = prox(lambda_T, sigma)
        #lu_piv_ten = np.array(lu_piv).reshape(6,6)
        #lam_T_LU = tf.matmul(u_pT_t_rz, lu_piv_ten)
        #print('lambda LU is',lam_T_LU)
    
    #--------------cal of loss
    #L = min 1/2 ||lambda_T*phi - ddt||^2 + lam ||z|| + sum y_ij(lambda_T-z_T) + rho/2 ||lambda_T-z_T||^2
    
        y_predict1 = XX @ lambda_dx 
        f1 = np.sum( (dydt - y_predict1)**2 )
        g1 = np.abs(lambda_dx).sum()
        h1 = np.sum(y_coeff*(lambda_dx-z_T[0,:]))
        i1 = np.sum((lambda_dx-z_T[0,:])**2)
        loss1 = 0.5*f1 + α1*g1 + h1 + rho*0.5*i1
        
        losses1.append(loss1)
        
        y_predict2 = XX @ lambda_dy  
        f2 = np.sum( (dydt - y_predict2)**2 )
        g2 = np.abs(lambda_dy).sum()
        h2 = np.sum(y_coeff*(lambda_dy-z_T[1,:]))
        i2 = np.sum((lambda_dy-z_T[1,:])**2)
        loss2 = 0.5*f2 + α2*g2 + h2 + rho*0.5*i2
        
        losses2.append(loss2)
        
        y_predict1 = XX @ lambda_dx 
        f1 = np.sum( (dydt - y_predict1)**2 )
        g1 = np.abs(lambda_dx).sum()
        h1 = np.sum(y_coeff*(lambda_dx-z_T[0,:]))
        i1 = np.sum((lambda_dx-z_T[0,:])**2)
        loss1 = 0.5*f1 + α1*g1 + h1 + rho*0.5*i1
        
        losses1.append(loss1)
        
        y_predict3 = XX @ lambda_dz  
        f3 = np.sum( (dzdt - y_predict3)**2 )
        g3 = np.abs(lambda_dz).sum()
        h3 = np.sum(y_coeff*(lambda_dz-z_T[1,:]))
        i3 = np.sum((lambda_dz-z_T[1,:])**2)
        loss3 = 0.5*f3 + α3*g3 + h3 + rho*0.5*i3
        
        losses3.append(loss3)
        
        
        #L = loss1 + loss2
        #print('total loss is', L)
    
    y_pred = XX @ tf.transpose(lambda_LU)
    f_de = np.sum((tf.transpose(ddt)-y_pred)**2)
    f_data = losses1[-1] + losses2[-1] + losses3[-1]
    print('Data loss is', f_data)
    print('Derivative loss is', f_de)
    
    fig,ax2 = plt.subplots(1,1)
    ax2.plot(losses1, label = 'losses of dxdt')
    ax2.legend(loc='upper right')  
    
    fig,ax3 = plt.subplots(1,1)
    ax3.plot(losses2, label = 'losses of dydt')
    ax3.legend(loc='upper right') 
    
    fig,ax4 = plt.subplots(1,1)
    ax4.plot(losses3, label = 'losses of dzdt')
    ax4.legend(loc='upper right') 

    print('Coeff matrix is', lambda_T)
    
    libdata['z_T'] = z_T
    libdata['y_ij'] = y_coeff
    libdata['lambda_T'] = lambda_T
    libdata['lambda_LU'] = lambda_LU
    #libdata['lambda_sparse'] = lambda_sparse
      
   
    return lambda_T, lambda_LU #, lambda_sparse

def various_opt(libdata):
    
    lambda_LU = libdata['lambda_LU']    
    iv = libdata['iv']
    ddt = libdata['ddt']
    
    dxdt = ddt[0,:]
    dydt = ddt[1,:]
    dzdt = ddt[2,:]
    XX = np.c_[np.ones_like(dxdt),iv] 
    
    beta1 = em_mlr(XX, dxdt)
    #print(f"Estimated dxdt Coefficients: {beta1}")
 
    beta2 = em_mlr(XX, dydt)
    #print(f"Estimated dydt Coefficients: {beta2}")  
    
    beta3 = em_mlr(XX, dzdt)
    #print(f"Estimated dzdt Coefficients: {beta3}") 
 
    mask1 = np.abs(beta1) > sigma
    mask_dx = beta1*mask1 
    #print('dxdt coeff is', mask_dx)
    
    mask2 = np.abs(beta2) > sigma
    mask_dy = beta2*mask2 
    #print('dydt coeff is', mask_dy)
    
    mask3 = np.abs(beta3) > sigma
    mask_dz = beta3*mask3 
    #print('dydt coeff is', mask_dy)
    
    lambda_sparse = tf.transpose(np.c_[mask_dx,mask_dy, mask_dz])
    
    #------------------------------------------
    # beta1 = em_Mp(XX, dxdt)
    # #print(f"Estimated dxdt Coefficients: {beta1}")  
 
    # beta2 = em_Mp(XX, dydt)
    # #print(f"Estimated dydt Coefficients: {beta2}")  
 
    # mask1 = np.abs(beta1) > sigma
    # mask_dx = beta1*mask1 
    # #print('dxdt coeff is', mask_dx)
    
    # mask2 = np.abs(beta2) > sigma
    # mask_dy = beta2*mask2 
    # #print('dydt coeff is', mask_dy)
    
    # lambda_sparse = tf.transpose(np.c_[mask_dx,mask_dy])
    
    #----------------------------------------------------
    # mask1 = np.abs(lambda_LU[0,:]) > sigma
    # mask_dx = lambda_LU[0,:]*mask1 
    # #print('dxdt coeff is', mask_dx)
    
    # mask2 = np.abs(lambda_LU[1,:]) > sigma
    # mask_dy = lambda_LU[1,:]*mask2 
    # #print('dydt coeff is', mask_dy)
    
    # lambda_sparse = tf.transpose(np.c_[mask_dx,mask_dy])
    #-------------------------------------------------------
    
    #libdata['beta1'] = beta1
    #libdata['beta2'] = beta2
 
    print('Sparse Coeff Matrix is', lambda_sparse)
    
    array = libdata['phi']
    lambda_phis = tf.matmul(lambda_sparse, array)
    loss_sparsess = tf.reduce_mean(tf.square(ddt - lambda_phis))
    print('loss from sparsess is:', loss_sparsess)  #physics loss
    
    
    libdata['lambda_sparse'] = lambda_sparse
    
    return lambda_sparse


def get_admm_sol(libdata):
    
    lambda_T = libdata['lambda_T']
    
    def func_check(t, x):
        #dxdt = lambda_T[0,4]*x[1] + lambda_T[0,2]*(x[0]*x[0]) + lambda_T[0,3]*(x[0]**4) + lambda_T[0,7]*((x[0]**2)*(x[1]**2)) 
        #dydt = lambda_T[1,1]*x[0] + lambda_T[1,4]*x[1]+lambda_T[1,6]*((x[0]**2)*x[1]) + lambda_T[1,5]*(x[1]**3)     
       
        y1 = lambda_T[0,0] + lambda_T[0,1]*x[0] + lambda_T[0,2]*x[1] + lambda_T[0,3]*(x[2]) #+ lambda_T[0,4]*(x[1]**2) + lambda_T[0,5]*(x[0]*x[1]) 
        y2 = lambda_T[1,0] + lambda_T[1,1]*x[0] + lambda_T[1,2]*x[1] + lambda_T[1,3]*(x[2]) #+ lambda_T[1,4]*(x[1]**2) + lambda_T[1,5]*(x[0]*x[1])
        y3 = lambda_T[2,0] + lambda_T[2,1]*x[0] + lambda_T[2,2]*x[1] + lambda_T[2,3]*(x[2]) #+ lambda_T[0,4]*(x[1]**2) + lambda_T[0,5]*(x[0]*x[1]) 
        
        return np.array([y1, y2, y3])
    
    sol_lambda = scint.solve_ivp(fun=func_check, t_span=(0,100), y0=[0.96, -1.1,0.5], method="LSODA", t_eval=np.linspace(0,100,1000), args=(sigma, beta, rho_lo))
    
    libdata['sol_lambda'] = sol_lambda

    return sol_lambda

def get_LU_sol(libdata):
    
    lams = libdata['lambda_sparse']
    
    def func_check(t, x):
        #dxdt = lambda_T[0,4]*x[1] + lambda_T[0,2]*(x[0]*x[0]) + lambda_T[0,3]*(x[0]**4) + lambda_T[0,7]*((x[0]**2)*(x[1]**2)) 
        #dydt = lambda_T[1,1]*x[0] + lambda_T[1,4]*x[1]+lambda_T[1,6]*((x[0]**2)*x[1]) + lambda_T[1,5]*(x[1]**3)     
       
        y1 = lams[0,0] + lams[0,1]*x[0] + lams[0,2]*x[1] + lams[0,3]*(x[2]) #+ lams[0,4]*(x[1]**2) + lams[0,5]*(x[0]*x[1]) 
        y2 = lams[1,0] + lams[1,1]*x[0] + lams[1,2]*x[1] + lams[1,3]*(x[2]) #+ lams[1,4]*(x[1]**2) + lams[1,5]*(x[0]*x[1])
        y3 = lams[2,0] + lams[2,1]*x[0] + lams[2,2]*x[1] + lams[2,3]*(x[2])# + lams[1,4]*(x[1]**2) + lams[1,5]*(x[0]*x[1])
        
        #array = tf.stack([const_one,x,x2,x4,y,y3,x2y,x2y2],axis=0)
        #dxdt = -x[1] + miu*(x[0]**2)*(1-((x[0]**2)+(x[1]**2)))      
            # = -y + miu*x^2 - miu*x^4 - miu*x^2*y^2
        #dydt = x[0] + miu*x[1]*(1-((x[0]**2)+(x[1]**2))) 
            # = x + miu*y - miu*x^2*y - miu*y^3
        #print(np.array([y1,y2]))
        
        return np.array([y1, y2])
    
    sol_LU = scint.solve_ivp(fun=func_check, t_span=(0,100), y0=[0.96, -1.1,0.5], method="LSODA", t_eval=np.linspace(0,100,1000), args=(sigma, beta, rho))
    
    libdata['sol_LU'] = sol_LU

    return sol_LU

def get_lasso_sol(libdata):
    
    w1 = libdata['w1']
    w2 = libdata['w2']
    output = libdata['output']
    
    w = np.c_[w1,w2]
    
    def func_check(t, x):
        
        #---------------sparse----------
        y1 = w[3,0] + w[0,0]*x[0] + w[1,0]*x[1] + w[2,0]*(x[2])# + w[3,0]*(x[1]**2) + w[4,0]*(x[0]*x[1]) 
        y2 = w[3,1] + w[0,1]*x[0] + w[1,1]*x[1] + w[2,1]*(x[2])# + w[3,1]*(x[1]**2) + w[4,1]*(x[0]*x[1])
        y2 = w[3,1] + w[0,1]*x[0] + w[1,1]*x[1] + w[2,1]*(x[2])# + w[3,1]*(x[1]**2) + w[4,1]*(x[0]*x[1])
        #---------------lasso----------
        #y1 = -3.0794e-5 + -4.9602e-1*x[0] + 9.9829e-1*x[1] + -8.1047e-9*(x[0]**2) + -5.1425*(x[1]**2) -7.5256e-9*(x[0]*x[1]) 
        #y2 = -10.9795 -26.5780*x[0] - 59.4701*x[1] + 532.9849*(x[0]**2) + 293.8160*(x[1]**2) + 265.4348*(x[0]*x[1])
       
        return np.array([y1, y2])
    
    sol_sparse = scint.solve_ivp(fun=func_check, t_span=(0,100), y0=[0.96, -1.1,0.5], method="LSODA", t_eval=np.linspace(0,100,1000), args=(sigma, beta, rho))
    
    # fig,ax2 = plt.subplots(1,1)
    # ax2.plot(output[:,0], output[:,1],'x', label = 'RK45')    #(100,2)
    # ax2.legend(loc='upper left')
    # ax2.plot(sol_sparse.y[0,:],sol_sparse.y[1,:],'x', label='ADMM')
    # ax2.set_xlabel('x')#, fontdict=font)
    # ax2.set_ylabel('y')#, fontdict=font)
    # ax2.legend(loc='upper left')
    
    libdata['sol_sparse'] = sol_sparse

    return sol_sparse

def visualize(libdata):
    output = libdata['output']
    sparse_u_pinn = libdata['sparse_u_pinn']
    # x_vec = libdata['x_vec']
    # y_vec = libdata['y_vec']
    
    
    # #losses_pinn = lib_data['losses_pinn']
    # losses_sparse = libdata['losses_sparse']
    # term1_tra = libdata['term1_tra']
    # term2_lamz = libdata['term2_lamz']
    # y_lam_changes = libdata['y_lam_changes']
    # rho_lam_change = libdata['rho_lam_change']
    # L_grad = libdata['L_grad']
    
    sol_lambda = libdata['sol_lambda']
    sol_LU = libdata['sol_LU']
    sol_sparse = libdata['sol_sparse']
    output = libdata['output']
    
    t = libdata['t']
    fig,ax4 = plt.subplots(1,1)
    fig.suptitle("Trajectory of data", fontsize=12)
    #----- not suppose to plot output x,y?-----
    ax4.plot(output[:,0], output[:,1], label='data')  
    ax4.legend(loc='upper left')
    ax4.plot(sparse_u_pinn[:,0],sparse_u_pinn[:,1],'x', label='PINNs')  
    ax4.set_xlabel('x')#, fontdict=font)
    ax4.set_ylabel('y')#, fontdict=font)
    ax4.legend(loc='upper left')
    fig.savefig('solutions trajectory.png', dpi=300)
    
    fig,ax5 = plt.subplots(1,1)
    fig.suptitle("RK45 vs IPINNs", fontsize=12)
    ax5.plot(output[:,0], output[:,1],'x', label = 'RK45')    #(100,2)
    ax5.legend(loc='upper left')
    ax5.plot(sol_lambda.y[0,:],sol_lambda.y[1,:],'x', label='IPINNs')
    ax5.set_xlabel('x(t)')#, fontdict=font)
    ax5.set_ylabel('y(t)')#, fontdict=font)
    ax5.legend(loc='upper left')
    fig.savefig('ADMM trajectory.png', dpi=300)
    
    fig,ax6 = plt.subplots(1,1)
    fig.suptitle("RK45 vs SIPINNs", fontsize=12)
    ax6.plot(output[:,0], output[:,1],'x', label = 'RK45')    #(100,2)
    ax6.legend(loc='upper right')
    ax6.plot(sol_LU.y[0,:],sol_LU.y[1,:],'x', label='SIPINNs')
    ax6.set_xlabel('x')#, fontdict=font)
    ax6.set_ylabel('y')#, fontdict=font)
    ax6.legend(loc='upper right')
    fig.savefig('LU trajectory.png', dpi=300)
    
    
    #differences between u_train vs lambda_T (PINNs-Sparse) leads to loss_sparse
    fig,ax7 = plt.subplots(1,1)
    ax7.plot(output[:,0], output[:,1],'x', label = 'data')    #(100,2)
    ax7.legend(loc='upper right')
    ax7.plot(sol_sparse.y[0,:],sol_sparse.y[1,:],'x', label='lasso')
    ax7.set_xlabel('x')#, fontdict=font)
    ax7.set_ylabel('y')#, fontdict=font)
    ax7.legend(loc='upper right')
     
    fig,ax8 = plt.subplots(1,1)
    ax8.plot(t, output[:,0], label = 'exact_x')
    ax8.plot(t, sol_lambda.y[0,:], 'x', label = 'x')
    #ax8.plot(t[0:-1], sol_lambda.y[0,:], 'x', label = 'x')
    ax8.set_title('sparse_x')
    ax8.legend(loc='upper right')
    ax8.set_xlabel('t')#, fontdict=font)
    ax8.set_ylabel('x')#, fontdict=font)
    ax8.legend(loc='upper right')
     
    fig,ax9 = plt.subplots(1,1)
    ax9.plot(t, output[:,1], label = 'exact_y')
    ax9.plot(t, sol_lambda.y[1,:], 'x', label = 'y')
    #ax9.plot(t[0:-1], sol_lambda.y[1,:], 'x', label = 'y')
    ax9.set_title('sparse_y')
    ax9.legend(loc='upper right')
    ax9.set_xlabel('t')#, fontdict=font)
    ax9.set_ylabel('y')#, fontdict=font)
    ax9.legend(loc='upper left')
    
    fig,ax10 = plt.subplots(1,1)
    ax10.plot(t, output[:,2], label = 'exact_z')
    ax10.plot(t, sol_lambda.y[2,:], 'x', label = 'z')
    #ax9.plot(t[0:-1], sol_lambda.y[1,:], 'x', label = 'y')
    ax10.set_title('sparse_z')
    ax10.legend(loc='upper right')
    ax10.set_xlabel('t')#, fontdict=font)
    ax10.set_ylabel('z')#, fontdict=font)
    ax10.legend(loc='upper left')
    
    # #loss_sparse is decreasing?
    # fig,ax2 = plt.subplots(1,1)
    # ax2.plot(losses_sparse, label = 'losses of Sparse')
    # ax2.legend(loc='upper right')   
    # #plot trajectory of coupled first order ODE (especially on other time t)
    
    
    # fig,ax5 = plt.subplots(1,1)
    # ax5.plot(term1_tra, label = 'Smooth (1st)')
    # ax5.set_xlabel('number of iteration')#, fontdict=font)
    # ax5.set_ylabel('changes of lambda_T phi^T - u_t^T')#, fontdict=font)
    # ax5.legend(loc='upper right')   
    # fig.savefig('trajectory of smooth (1st).png')
  
    # fig,ax6 = plt.subplots(1,1)
    # ax6.plot(term2_lamz, label = 'lam_z (2nd)')
    # ax6.set_xlabel('number of iteration')#, fontdict=font)
    # ax6.set_ylabel('changes of lambda z')#, fontdict=font)
    # ax6.legend(loc='upper right')   
    # fig.savefig('trajectory of lambda norm1 z.png')
  
    # fig,ax7 = plt.subplots(1,1)
    # ax7.plot(y_lam_changes, label = 'y_lam (3rd)')
    # ax7.set_xlabel('number of iteration')#, fontdict=font)
    # ax7.set_ylabel('changes of y_lam')#, fontdict=font)
    # ax7.legend(loc='upper right')   
    # fig.savefig('trajectory of y_lambda z.png')
  
    # fig,ax8 = plt.subplots(1,1)
    # ax8.plot(rho_lam_change, label = 'rho_lam (4th)')
    # ax8.set_xlabel('number of iteration')#, fontdict=font)
    # ax8.set_ylabel('changes of rho_lam')#, fontdict=font)
    # ax8.legend(loc='upper right')   
    # fig.savefig('trajectory of rho_lambda z.png')
  
    # fig,ax9 = plt.subplots(1,1)
    # ax9.plot(L_grad, label = 'Grad_L')
    # ax9.set_xlabel('number of iteration')#, fontdict=font)
    # ax9.set_ylabel('L norm')#, fontdict=font)
    # ax9.legend(loc='upper right')   
    # fig.savefig('L magnitude.png')
    

def main():
    libdata = make_data()
    prepare_ddt(libdata)
    pinn(libdata)
    #prepare_data(libdata)    
    
    #lass_fit(libdata)
    
    #sparse(libdata)
    #various_opt(libdata)
    
    #get_admm_sol(libdata)
    #get_LU_sol(libdata)
    #get_lasso_sol(libdata)
    #visualize(libdata)
    
    return libdata
    
libdata = main()


