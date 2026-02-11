# -*- coding: utf-8 -*-
"""
Created on Jul 9  2025

"""

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy as sp
import pywt
import time
import matplotlib.pyplot as plt
import scipy.integrate as scint
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense #, Lambda
from keras.saving import load_model
from tensorflow.keras import backend as K
from tensorflow import keras
from scipy.optimize import minimize
from box import Box
import pickle

from sklearn.linear_model import Lasso

start_time = time.time()

epochs = 30
batch_size = 16
lamda = 1e-2
rho = ρ = tf.constant(2, dtype=tf.float64)
learn_rate = -2e-6
losses_sparse = []
N = 43
fore_N = 37

α = 1 #np.sqrt(5) / 2

row = 4
col = 1

sigma = 8e-3  #>sigma-->keep
sigma1 = 1e-10 #proximal 
sigma2 = 5e-5

α1 = 0.001 #lasso weight search for dxdt
α2 = 0.001 #lasso weight search for dydt 
αf1 = 10000 #loss func cal for dxdt
αf2 = 1e-4 #loss func cal for dydt

data = pd.read_csv("population_norm.csv")
train_data = tf.convert_to_tensor(data.values)

    
dosm = pd.read_csv("dosm.csv")
forecast_data = tf.convert_to_tensor(dosm.values)

# model_pkl_file = "pinn_model.pkl" 
# with open(model_pkl_file, 'rb') as file:  
#     model = pickle.load(file)

def make_data():    
    #with open ('p_repeated_noise.pkl','rb') as f:
        #pdata = pickle.load(f)
       
    # global data
    libdata ={
        
           't_full' : train_data[:,0],
           'x_full' : train_data[:,1],
           'y_full' : train_data[:,2],
           'dxdt_full' : train_data[:,3],
           'dydt_full' : train_data[:,4],
           'ori_x_full': train_data[:,5],
           'ori_y_full' : train_data[:,6],
           'ori_dxdt_full':train_data[:,7],
           'ori_dydt_full':train_data[:,8],
           
           't' : train_data[:N,0],
           'x' : train_data[:N,1],
           'y' : train_data[:N,2],
           
           'dxdt_given' : train_data[:N,3],
           'dydt_given' : train_data[:N,4],
           'ori_x': train_data[:N,5],
           'ori_y' : train_data[:N,6],
           'ori_dxdt':train_data[:N,7],
           'ori_dydt':train_data[:N,8],
           
           'va_t' : train_data[N:0],
           'va_ori_x' : train_data[N:,5],
           'va_ori_y' : train_data[N:,6],
           
           'ddt':[],
           'pinn_dxdt' :[],
           'pinn_dydt':[],
           
           'model': None,
           
           'ori_ddt' :[],
           'w1':[],
           'w2':[],
           'iv1':[],
           'iv2':[],
           'w_sparse1':[],
           'w_sparse2':[],
           'sol_sparse':[],
           
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
           'lambda_sparse':[],
           'sol_future':[],
           
           'lam_dxdt':[],
           'lam_dydt':[],
           
           'u_pinn':[],
           'u_pred':[],
           'upred_dx':[],
           'upred_dy':[],
           'upred_ddt' : [],
           
           'end_dx' :[],
           'end_dy' :[],
           
           #'u_pinn' : pdata.u_pinn,
           #'ddt_noise':pdata.ddt_noise,
           #'output_noise':pdata.output_noise,
           #'u_pinn_noise': pdata.u_pinn_noise,
           
           #'model': pdata.model, 
           'fname':None,
           
           #'loss_':[],
           #'loss_ic':[],
           #'loss_f':[],
           #'error_vec':[],
           
           't_vec' : [],
           #'x_vec' : pdata.x_vec,
           #'y_vec' : pdata.y_vec,
           'output': [],
           
           'lambda_phi': [],
           'y_lam' : [],
           'term1_tra' : [],
           'term2_lamz' :[],
           'y_lam_changes' : [],
           'rho_lam_change' : [],
           'L_grad' : [],
           
           'sol_lambda' : [],
           'sol_ml' : [],
        
           #'u_test': [],
           #'numerical_rhs':[],
           #'grads':[],
           'losses':[],
           'loss_sparse':[],
           'losses_sparse':[],
           
           'fore_t' : forecast_data[:-3,0],
           'fore_f' : forecast_data[:-3,1],
           'fore_m' : forecast_data[:-3,2],
           
           }
    return Box(libdata)

def setting_model(libdata):   
    t = libdata['t']
    x = libdata['x']
    y = libdata['y']
    dxdt_given = libdata['dxdt_given']
    dydt_given = libdata['dydt_given']
    
    t_vec = np.array(t).reshape(-1,1)
   
    dxdt_vec = np.array(dxdt_given).reshape(-1,1)
    dydt_vec = np.array(dydt_given).reshape(-1,1)
     
    #print(output.shape)
    ddt = np.c_[dxdt_vec,dydt_vec]
   #print(ddt)
    
    model = Sequential()
    model.add(InputLayer(input_shape=(1,),dtype=tf.float64))
    #normalize contraint add here - batch normalize layer
    model.add(Dense(8, activation=tf.keras.activations.gelu,kernel_initializer="glorot_uniform", dtype=tf.float64))
    model.add(Dense(16, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(32, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(64, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(128, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(64, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(32, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(16, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(8, activation=tf.keras.activations.gelu, kernel_initializer="glorot_uniform", dtype=tf.float64))
    model.add(Dense(2, activation=None, kernel_initializer="glorot_normal",dtype=tf.float64))

    model.summary()
    
    #plot_model(model,show_shapes=True)
    #model.trainable_variables
    #libdata['model'] = model
      
#-------------    
    @keras.saving.register_keras_serializable()
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
                
            f1 = dx_dt - dxdt_vec 
            #print('dxdt loss is', f1)
            f2 = dy_dt - dydt_vec
            #print('dydt loss is', f2)
        
        #ddt_pinn = Concatenate()([dx_dt,dy_dt])
        
        loss_f = tf.reduce_mean(tf.square(f1)+tf.square(f2))
 
        loss = loss_ic + loss_f
        
        tf.print('loss on ic is:', loss_ic)
        tf.print('loss on derivative(GTape) is:', loss_f)
        tf.print('total loss is:', loss)
        
        return loss
        
    """3 steps: compile, fit, predict.       """
    #model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
    #model.compile(loss=CustomLoss(), optimizer='adam', metrics=['accuracy'])
    
    #years = np.arange(1971, 2024).reshape(-1, 1)
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
    
    libdata['pinn_dx'] = dx_dt
    libdata['pinn_dy'] = dy_dt
    libdata['ddt'] = ddt
    
    model.save('pinn_model.keras')
    
    return model
    
    
def pinn(libdata, model):
    t = libdata['t']
    t_vec = np.array(t).reshape(-1,1)
    x = libdata['x']
    y = libdata['y']
    ori_x = libdata['ori_x']
    ori_y = libdata['ori_y']
    ori_dxdt = libdata['ori_dxdt']
    ori_dydt = libdata['ori_dydt']
    pinn_dx = libdata['pinn_dx']
    pinn_dy = libdata['pinn_dy']
    
    
    x_max = np.max(ori_x)
    x_min = np.min(ori_x)
    y_min = np.min(ori_y)
    y_max = np.max(ori_y) 
    
    x_vec = np.array(x).reshape(-1,1)
    y_vec = np.array(y).reshape(-1,1)
    output = np.c_[x,y]
    
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
    
    u_pinn = model(t_vec)
    #print(u_pinn)
    u_pred_x = u_pinn[:,0] * (x_max - x_min) + x_min
    u_pred_y = u_pinn[:,1] * (y_max - y_min) + y_min
    u_pred = np.c_[u_pred_x,u_pred_y]
    #print(u_pred)
    loss_x = tf.reduce_mean(tf.square(x_vec - u_pinn[:,0]))
    loss_y = tf.reduce_mean(tf.square(y_vec - u_pinn[:,1]))
    loss_data = loss_x + loss_y
    
    print('Data loss on x', loss_x)
    print('Data loss on y',loss_y)
    print('Total_Data_loss is', loss_data)
    
    dx_max = np.max(ori_dxdt)
    dx_min = np.min(ori_dxdt)
    dy_min = np.min(ori_dydt)
    dy_max = np.max(ori_dydt) 
    
    upred_dx = pinn_dx * (dx_max - dx_min) + dx_min # np.array(ori_dxdt).reshape(-1,1)
    upred_dy = pinn_dy * (dy_max - dy_min) + dy_min #np.array(ori_dydt).reshape(-1,1)
    #upred_ddt = np.c_[upred_dx,upred_dy]
    
    libdata['t_vec'] = t_vec
    libdata['x_vec'] = x_vec
    libdata['y_vec'] = y_vec
    libdata['output'] = output
    libdata['u_pinn'] = u_pinn
    libdata['u_pred'] = u_pred
    libdata['upred_dx'] = upred_dx
    libdata['upred_dy'] = upred_dy
    
    return u_pred, model

def prepare_data(libdata):
    
    ori_x = libdata['ori_x']
    ori_y = libdata['ori_y']
    
    const_one = tf.ones([N,])
    const_one = tf.cast(const_one,dtype=tf.float64)
    
    x2 = tf.square(ori_x)
    y2 = tf.square(ori_y)
    
    xy = tf.multiply(ori_x,ori_y)
  
    identity = tf.eye(row,dtype=tf.float64)  
           
    #array = tf.stack([x,x_sqr,y_cube,x2y2],axis=0)
    #array = tf.stack([const_one,x,y],axis=0)
    array1 = tf.stack([ori_x,x2, ori_y, xy],axis=0)
    array2 = tf.stack([ori_y,y2, ori_x, xy],axis=0)
    #print(array)   #(5x100)
    iv1 = np.c_[ori_x,x2, ori_y, xy]  #+ xy
    iv2 = np.c_[ori_y,y2, ori_x, xy]  # + xy 
    
    #array_var_T = tf.transpose(array)
    #print(array_var_T)
    
    u_vec = tf.ones([row,col])
    v_vec = tf.ones([row,col])
    lammda = np.c_[u_vec,v_vec]
    lambda_T = lammda.T   #dtype:float32
    lambda_T = tf.cast(lambda_T,dtype=tf.float64) #convert tensor to tensor
    #print(matrix_T)
    y_coeff1 = y_coeff2 = tf.cast(tf.ones([2,row])/10,dtype=tf.float64)
    z_T1 = z_T2 = tf.convert_to_tensor(np.ones_like(lambda_T), dtype=tf.float64)
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



#--------lasso fitting to get initial coefficient matrix------
def lass_fit(libdata):
    #given_xy = las_data['output']
    ddt = libdata['upred_ddt']
    x = libdata['x']
    y = libdata['y']
    iv1 = libdata['iv1']
    iv2 = libdata['iv2']
    
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

#----------------end of sparse_noise ------------------


def sparse(libdata):
    iv1 = libdata['iv1']
    iv2 = libdata['iv2']
    array1 = libdata['phi1']
    array2 = libdata['phi2']
    identity = libdata['identity']
    ddt = libdata['ddt']
    lambda_T = libdata['lambda_T']
    y_coeff1 = libdata['y_ij1']
    y_coeff2 = libdata['y_ij2']
    z_T1 = libdata['z_T1']
    z_T2 = libdata['z_T2']
    
    ori_dxdt = libdata['ori_dxdt']
    ori_dydt = libdata['ori_dydt']
    u_pred = libdata['u_pred']
    
    upred_dx = libdata['upred_dx']
    upred_dy = libdata['upred_dy']
     
    #ori_x = libdata['ori_x']
    #ori_y = libdata['ori_y']
   
    ori_ddt = np.c_[ori_dxdt,ori_dydt]
    upred_ddt = np.c_[upred_dx,upred_dy]
    
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

    ut_phi1 = tf.matmul(tf.transpose(ori_ddt), tf.transpose(array1))
    #print('u_t phi1 is', ut_phi1)
    phi_phiT1 = tf.matmul(array1, tf.transpose(array1))  #phi: (6x100)
    #print('phiphiT1',phi_phiT1)
    rhoI = tf.multiply(rho,identity)
    #print('rhoI', rhoI)
    ppT_rI1 = tf.add(phi_phiT1,rhoI)   
    #print('phi2^T1 + rhoI', ppT_rI1)
    ppTrI_inv1 = tf.linalg.pinv(ppT_rI1)
    #print('least square inverse1 is', ppTrI_inv1)
    
    ut_phi2 = tf.matmul(tf.transpose(ori_ddt), tf.transpose(array2))
    #print('u_t phi2 is', ut_phi2)
    phi_phiT2 = tf.matmul(array2, tf.transpose(array2))  #phi: (6x100)
    #print('phiphiT2',phi_phiT2)
    rhoI = tf.multiply(rho,identity)
    #print('rhoI', rhoI)
    ppT_rI2 = tf.add(phi_phiT2,rhoI)   
    #print('phi2^T2 + rhoI', ppT_rI2)
    ppTrI_inv2 = tf.linalg.pinv(ppT_rI2)
    #print('least square inverse2 is', ppTrI_inv2)
    
       #-----------------------------------
    #dxdt = ori_ddt[:,0]
    #dydt = ori_ddt[:,1]
    #XX1 = np.c_[np.ones_like(dxdt),iv1]    #(100,6)
    XX1 = iv1
    XX1_ddt = ori_ddt.T @ XX1
    #print('XX1_ddt is', XX1_ddt)   #values equal to ut_phi1
    XTX1 = XX1.T @ XX1 
    #print('XTX1', XTX1)   #values equal to phi_phiT
    #aa = ρ* np.eye( *XTX.shape)
    #print('ρI', aa)
    A1 = XTX1 + ρ* np.eye( *XTX1.shape) 
    #print('LU_method A1 matrix', A1)   #values equal ppT_rI
    invr1 = np.linalg.inv(A1)
    #print('inverse1', invr1)
   
    #XX2 = np.c_[np.ones_like(dxdt),iv2]    #(100,6)
    XX2 = iv2
    XX2_ddt = ori_ddt.T @ XX2
    #print('XX2_ddt is', XX2_ddt)   #values equal to ut_phi2
    XTX2 = XX2.T @ XX2 
    #print('XTX2', XTX2)   #values equal to phi_phiT
    #aa = ρ* np.eye( *XTX.shape)
    #print('ρI', aa)
    A2 = XTX2 + ρ* np.eye( *XTX2.shape) 
    #print('LU_method A2 matrix', A2)   #values equal ppT_rI
    invr2 = np.linalg.inv(A2)
    #print('inverse2', invr2) 
   
    lambda_LU = np.vstack((tf.ones([col,row]), tf.ones([col,row])))
    xi1 = xi2 = np.ones_like(lambda_LU)/10
    #xi = np.ones_like(lambda_LU)
    z = np.ones_like(lambda_LU)

    losses1 = []
    losses2 = []
   
    
    for i in range (300):
  #_______to update lambda_T__
    
        subtra1 = tf.subtract(ut_phi1, y_coeff1)
        #print('ut_phi1 - y1', subtra1)
        rhoz1 = tf.multiply(rho,z_T1)
        u_pT_t_rz1 = tf.add(subtra1, rhoz1)
        #print('ut_phi1 + rhoz1 - y1 is', u_pT_t_rz1)
        lambda_T1 = tf.matmul(u_pT_t_rz1, ppTrI_inv1)
        
        subtra2 = tf.subtract(ut_phi2, y_coeff2)
        #print('ut_phi2 - y2', subtra2)
        rhoz2 = tf.multiply(rho,z_T2)
        u_pT_t_rz2 = tf.add(subtra2, rhoz2)
        #print('ut_phi2 + rhoz2 - y2 is', u_pT_t_rz2)
        lambda_T2 = tf.matmul(u_pT_t_rz2, ppTrI_inv2)
        #print('lambda k+1 is',lambda_T2)
        
        lambda_T = np.vstack([lambda_T1[0,:], lambda_T2[1,:]])
 #------------------------------------
         #np.hstack((b_dx, b_dy))
       
        lu_piv1 = sp.linalg.lu_factor(A1)
        lu_piv2 = sp.linalg.lu_factor(A2)
        #print('LU factorization', lu_piv)
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
        lambda_LU = tf.transpose(np.c_[lambda_dx,lambda_dy])
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
        frac2 = (1/rho * y_coeff1) + lambda_T1
        S1 = pywt.threshold(frac2, frac1 , 'soft')  #frac2:our signal; frac1: value to threshold
        #print('Softthresholding z is', S)
                
        z1 = pywt.threshold(lambda_LU + xi1/ρ , frac1 , 'soft') 
        #print('Soft z(LU1) is', z1)
        z_T1 = S1
        
        frac3 = np.array([α2/rho])
        frac4 = (1/rho * y_coeff2) + lambda_T2
        S2 = pywt.threshold(frac4, frac3 , 'soft')  #frac2:our signal; frac1: value to threshold
        #print('Softthresholding z is', S)
        
        
        z2 = pywt.threshold(lambda_LU + xi2/ρ , frac3 , 'soft') 
        #print('Soft z(LU2) is', z2)
        z_T2 = S2

#_________to update yij_______
        y_coeff1 = y_coeff1 + rho*(lambda_T1 - S1)
        #print('yij_k+1 is', y_coeff)     

        xi1 = xi1 + ρ*(lambda_LU - z1)
        #print('x1 is', xi)
        
        y_coeff2 = y_coeff2 + rho*(lambda_T2 - S2)
        #print('yij_k+1 is', y_coeff)     

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
        f1 = np.sum( (upred_dx - y_predict1)**2 )
        g1 = np.abs(lambda_dx).sum()
        h1 = np.sum(y_coeff1*(lambda_dx-z_T1[0,:]))
        i1 = np.sum((lambda_dx-z_T1[0,:])**2)
        loss1 = 0.5*f1 + α1*g1 + h1 + rho*0.5*i1
        
        losses1.append(loss1)
        
        y_predict2 = XX2 @ lambda_dy  
        f2 = np.sum( (upred_dy - y_predict2)**2 )
        g2 = np.abs(lambda_dy).sum()
        h2 = np.sum(y_coeff2*(lambda_dy-z_T2[1,:]))
        i2 = np.sum((lambda_dy-z_T2[1,:])**2)
        loss2 = 0.5*f2 + α2*g2 + h2 + rho*0.5*i2
        
        losses2.append(loss2)
        
        #L = loss1 + loss2
        #print('total loss is', L)
    
    #XX = np.c_[XX1,XX2]
    y_pred1 = XX1 @ tf.transpose(lambda_LU)
    y_pred2 = XX2 @ tf.transpose(lambda_LU)
    f_de = np.sum((upred_ddt-y_pred1)**2) + np.sum((upred_ddt - y_pred2)**2)
    f_data = losses1[-1] + losses2[-1]
    print('Data loss is', f_data)
    print('Derivative loss is', f_de)
    
    fig,ax2 = plt.subplots(1,1)
    ax2.plot(losses1, label = 'losses of dxdt')
    ax2.legend(loc='upper right')  
    
    fig,ax3 = plt.subplots(1,1)
    ax3.plot(losses1, label = 'losses of dydt')
    ax3.legend(loc='upper right') 

    print('Coeff matrix is', lambda_T)
    
    libdata['ori_ddt'] = ori_ddt
    libdata['upred_ddt'] = upred_ddt
    libdata['z_T1'] = z_T1
    libdata['z_T2'] = z_T2
    libdata['y_ij1'] = y_coeff1
    libdata['y_ij2'] = y_coeff2
    libdata['lambda_T1'] = lambda_T1
    libdata['lambda_T2'] = lambda_T2
    libdata['lambda_LU'] = lambda_LU
    libdata['lambda_T'] = lambda_T
    libdata['lam_dxdt'] = y_pred1
    libdata['lam_dydt'] = y_pred2
    #libdata['lambda_sparse'] = lambda_sparse
      
   
    return lambda_T1, lambda_T2, lambda_LU #, lambda_sparse

def various_opt(libdata):
    
    lambda_LU = libdata['lambda_LU']    
    iv1 = libdata['iv1']
    iv2 = libdata['iv2']
    upred_ddt = libdata['upred_ddt']
    ori_ddt = libdata['ori_ddt']
    
    dxdt = ori_ddt[:,0]
    dydt = upred_ddt[:,1]
    XX1 = np.c_[np.ones_like(dxdt),iv1] 
    
    # beta1 = em_mlr(XX, dxdt)
    # #print(f"Estimated dxdt Coefficients: {beta1}")
 
    # beta2 = em_mlr(XX, dydt)
    # #print(f"Estimated dydt Coefficients: {beta2}")  
 
    # mask1 = np.abs(beta1) > sigma
    # mask_dx = beta1*mask1 
    # #print('dxdt coeff is', mask_dx)
    
    # mask2 = np.abs(beta2) > sigma
    # mask_dy = beta2*mask2 
    # #print('dydt coeff is', mask_dy)
    
    # lambda_sparse = tf.transpose(np.c_[mask_dx,mask_dy])
    
    #------------------------------------------
    mask1 = np.abs(lambda_LU[0,:]) > sigma
    mask_dx = lambda_LU[0,:]*mask1 
    #print('dxdt coeff is', mask_dx)
    
    mask2 = np.abs(lambda_LU[1,:]) > sigma
    mask_dy = lambda_LU[1,:]*mask2 
    #print('dydt coeff is', mask_dy)
    
    lambda_sparse = tf.transpose(np.c_[mask_dx,mask_dy])
    #-------------------------------------------------------
    
    #libdata['beta1'] = beta1
    #libdata['beta2'] = beta2
 
    print('Sparse Coeff Matrix is', lambda_sparse)
    
    libdata['lambda_sparse'] = lambda_sparse
    
    return lambda_sparse

def sing_eigen_values(libdata):
    lambda_T = libdata['lambda_T']
    lambda_sparse = libdata['lambda_sparse']
    
    #singular_values = np.linalg.svdvals(lambda_T)
    U, s, V = np.linalg.svd(lambda_T)
    print("Singular values (using svd):", s)
        
    #original_mat = -0.1*np.array([[7,1],[-4,3]])  #[-0.7,-0.1], [-0.4,0.3]
    # Calculate eigenvalues and eigenvectors
    #eigenvalues, eigenvectors = np.linalg.eig(original_mat)
    # Print the eigenvalues
    #print("Eigenvalues:", eigenvalues)
    # Print the eigenvectors
    #print("Eigenvectors:", eigenvectors)
    
    U2, s2, V2 = np.linalg.svd(lambda_sparse)
    print("Singular sparse_lambda (using svd):", s2)
    
    cond_1 = np.linalg.cond(lambda_T)
    print("Condition number (coeff_mat):", cond_1)
    
    cond_2 = np.linalg.cond(lambda_sparse)
    print("Condition number (coeff_sparse_mat):", cond_2)
    
    #cond_3 = np.linalg.cond(original_mat)
    #print("Condition number (ori_mat):", cond_3)
    
def get_admm_sol(libdata):
    
    lambda_T = libdata['lambda_T']
    
    def func_check(t, x):
        #dxdt = lambda_T[0,4]*x[1] + lambda_T[0,2]*(x[0]*x[0]) + lambda_T[0,3]*(x[0]**4) + lambda_T[0,7]*((x[0]**2)*(x[1]**2)) 
        #dydt = lambda_T[1,1]*x[0] + lambda_T[1,4]*x[1]+lambda_T[1,6]*((x[0]**2)*x[1]) + lambda_T[1,5]*(x[1]**3)     
       
        y1 = lambda_T[0,0]*x[0] + lambda_T[0,1]*(x[0]**2) + lambda_T[0,2]*(x[1]) + lambda_T[0,3]*(x[0]*x[1]) #+ lambda_T[0,2]*x[1] + lambda_T[0,4]*(x[1]**2) + lambda_T[0,5]*(x[0]*x[1]) 
        y2 = lambda_T[1,0]*x[1] + lambda_T[1,1]*(x[1]**2) + lambda_T[1,2]*(x[0])+ lambda_T[1,3]*(x[0]*x[1]) #lambda_T[1,0] + lambda_T[1,3]*(x[0]**2)+ lambda_T[1,1]*x[0] + lambda_T[1,5]*(x[0]*x[1])
        
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
    
    sol_lambda = scint.solve_ivp(fun=func_check, t_span=(0,N-1), y0=[5500,5600], method="RK45", t_eval=np.linspace(0,N-1,N-1))
    
    ml_start = sol_lambda.y[:, -1] 
    sol_ml = scint.solve_ivp(fun=func_check, t_span=(0,N+10), y0=[5500,5600], method="RK45", t_eval=np.linspace(0,N+10,N+10))

    #for_start = sol_lambda.y[:,-1]
    forecast_start=[16150,17880]
    t_future = np.linspace(0, fore_N, fore_N)
    sol_future = scint.solve_ivp(func_check, (t_future[0], t_future[-1]), forecast_start, method='RK45',t_eval=t_future)

    libdata['sol_lambda'] = sol_lambda
    libdata['sol_ml'] = sol_ml
    libdata['sol_future'] = sol_future
    
    return sol_lambda

def get_LU_sol(libdata):
    
    lams = libdata['lambda_sparse']
    
    def func_check(t, x):
        #dxdt = lambda_T[0,4]*x[1] + lambda_T[0,2]*(x[0]*x[0]) + lambda_T[0,3]*(x[0]**4) + lambda_T[0,7]*((x[0]**2)*(x[1]**2)) 
        #dydt = lambda_T[1,1]*x[0] + lambda_T[1,4]*x[1]+lambda_T[1,6]*((x[0]**2)*x[1]) + lambda_T[1,5]*(x[1]**3)     
       
        y1 = lams[0,0]*x[0] + lams[0,1]*(x[0]**2) + lams[0,2]*(x[1]) + lams[0,3]*(x[0]*x[1]) #lams[0,0]  +lams[0,4]*(x[1]**2) + lams[0,2]*x[1] + lams[0,5]*(x[0]*x[1]) 
        y2 = lams[1,0]*x[1] + lams[1,1]*(x[1]**2) + lams[1,2]*(x[0])+ lams[1,3]*(x[0]*x[1])#lams[1,0] + lams[1,1]*x[0] + lams[1,3]*(x[0]**2) + lams[1,5]*(x[0]*x[1])
        
        
        #array = tf.stack([const_one,x,x2,x4,y,y3,x2y,x2y2],axis=0)
        #dxdt = -x[1] + miu*(x[0]**2)*(1-((x[0]**2)+(x[1]**2)))      
            # = -y + miu*x^2 - miu*x^4 - miu*x^2*y^2
        #dydt = x[0] + miu*x[1]*(1-((x[0]**2)+(x[1]**2))) 
            # = x + miu*y - miu*x^2*y - miu*y^3
        #print(np.array([y1,y2]))
        
        return np.array([y1, y2])
    
    sol_LU = scint.solve_ivp(fun=func_check, t_span=(0,N-1), y0=[5500,5600], method="RK45", t_eval=np.linspace(0,N-1,N-1))
    
    libdata['sol_LU'] = sol_LU

    return sol_LU

def get_lasso_sol(libdata):
    
    w1 = libdata['w1']
    w2 = libdata['w2']
    
    w = np.c_[w1,w2]
    
    def func_check(t, x):
        
        #---------------sparse----------
        y1 = w[0,0]*x[0] + w[1,0]*(x[0]**2) + w[2,0]*x[1] + w[3,0]*(x[0]*x[1])#w[2,0] + + w[1,0]*x[1]+w[3,0]*(x[1]**2) + w[4,0]*(x[0]*x[1]) 
        y2 = w[0,1]*x[1] + w[1,1]*(x[1]**2) + w[2,1]*x[1] + w[3,1]*(x[0]*x[1]) #w[2,1] + w[0,1]*x[0]+ w[2,1]*(x[0]**2) + w[4,1]*(x[0]*x[1])
        
        #---------------lasso----------
        #y1 = -3.0794e-5 + -4.9602e-1*x[0] + 9.9829e-1*x[1] + -8.1047e-9*(x[0]**2) + -5.1425*(x[1]**2) -7.5256e-9*(x[0]*x[1]) 
        #y2 = -10.9795 -26.5780*x[0] - 59.4701*x[1] + 532.9849*(x[0]**2) + 293.8160*(x[1]**2) + 265.4348*(x[0]*x[1])
       
        return np.array([y1, y2])
    
    sol_sparse = scint.solve_ivp(fun=func_check, t_span=(0,N-1), y0=[5500,5600], method="RK45", t_eval=np.linspace(0,N-1,N-1))
    
    # fig,ax2 = plt.subplots(1,1)
    # ax2.plot(output[:,0], output[:,1],'x', label = 'RK45')    #(100,2)
    # ax2.legend(loc='upper left')
    # ax2.plot(sol_sparse.y[0,:],sol_sparse.y[1,:],'x', label='ADMM')
    # ax2.set_xlabel('x')#, fontdict=font)
    # ax2.set_ylabel('y')#, fontdict=font)
    # ax2.legend(loc='upper left')
    
    libdata['sol_sparse'] = sol_sparse

    return sol_sparse

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
    t = libdata['t']
    output = libdata['output']
    u_pinn = libdata['u_pinn']
    u_pred = libdata['u_pred']
    ori_x = libdata['ori_x']
    ori_y = libdata['ori_y']
    
    t_full = libdata['t_full']
    ori_x_full = libdata['ori_x_full']
    ori_y_full = libdata['ori_y_full']
    
    dxdt_given = libdata['dxdt_given']
    dydt_given = libdata['dydt_given']
    
    ori_dxdt = libdata['ori_dxdt']
    ori_dydt = libdata['ori_dydt']
    pinn_dx = libdata['pinn_dx']
    pinn_dy = libdata['pinn_dy']
    upred_dx = libdata['upred_dx']
    upred_dy = libdata['upred_dy']
    
    va_t = libdata['va_t']
    va_ori_x = libdata['va_ori_x']
    va_ori_y = libdata['va_ori_y']
    sol_ml = libdata['sol_ml']
    
    sol_future = libdata['sol_future']
    fore_t = libdata['fore_t']
    fore_f = libdata['fore_f']
    fore_m = libdata['fore_m']
    
    # x_vec = libdata['x_vec']
    # y_vec = libdata['y_vec']
    # output_noise = libdata['output_noise']
    
    output_ori = np.c_[ori_x,ori_y]
    out_ori_full = np.c_[ori_x_full, ori_y_full]
    va_output = np.c_[va_ori_x, va_ori_y]
    forecast_fm = np.c_[fore_f, fore_m]
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
    # output = libdata['output']
    
    end_dx = libdata['end_dx']
    end_dy = libdata['end_dy']
    
    fig,ax4 = plt.subplots(1,1)
    fig.suptitle("Trajectory of data", fontsize=12)
    ax4.plot(output[:,0], output[:,1],'x', label='data_norm')  
    ax4.legend(loc='upper left')
    ax4.plot(u_pinn[:,0],u_pinn[:,1],'x', label='PINNs_norm')  
    ax4.set_xlabel('x')#, fontdict=font)
    ax4.set_ylabel('y')#, fontdict=font)
    ax4.legend(loc='upper left')
    fig.savefig('sol trajectory_xy.png', dpi=300)
    
    fig,ax5 = plt.subplots(1,1)
    ax5.plot(t, output_ori[:,0], label = 'data_x')
    ax5.plot(t, u_pred[:,0], 'x', label = 'PINNs_x')
    ax5.set_title('trained_ori_x')
    ax5.legend(loc='upper left')
    ax5.set_xlabel('t')#, fontdict=font)
    ax5.set_ylabel('x')#, fontdict=font)
    ax5.legend(loc='upper left')
    
    fig,ax6 = plt.subplots(1,1)
    ax6.plot(t, output_ori[:,1], label = 'data_y')
    ax6.plot(t, u_pred[:,1], 'x', label = 'PINNs_y')
    ax6.set_title('trained_ori_y')
    ax6.legend(loc='upper left')
    ax6.set_xlabel('t')#, fontdict=font)
    ax6.set_ylabel('y')#, fontdict=font)
    ax6.legend(loc='upper left')
    
    fig,ax6 = plt.subplots(1,1)
    ax6.plot(t, dxdt_given, label = 'data_n_dxdt')
    ax6.plot(t, pinn_dx, 'x', label = 'PINNs_n_dxdt')
    ax6.set_title('pinn_n_dxdt')
    ax6.legend(loc='upper left')
    ax6.set_xlabel('t')#, fontdict=font)
    ax6.set_ylabel('dx')#, fontdict=font)
    ax6.legend(loc='upper left')
    
    fig,ax7 = plt.subplots(1,1)
    ax7.plot(t, dydt_given, label = 'data_n_dydt')
    ax7.plot(t, pinn_dy, 'x', label = 'PINNs_n_dydt')
    ax7.set_title('pinn_n_dydt')
    ax7.legend(loc='upper left')
    ax7.set_xlabel('t')#, fontdict=font)
    ax7.set_ylabel('dy')#, fontdict=font)
    ax7.legend(loc='upper left')
    
    fig,ax8 = plt.subplots(1,1)
    ax8.plot(t, ori_dxdt, label = 'data_ori_dxdt')
    ax8.plot(t, upred_dx, 'x', label = 'PINNs_ori_dxdt')
    ax8.set_title('pinn_ori_dxdt')
    ax8.legend(loc='upper left')
    ax8.set_xlabel('t')#, fontdict=font)
    ax8.set_ylabel('dx')#, fontdict=font)
    ax8.legend(loc='upper left')
    
    fig,ax9 = plt.subplots(1,1)
    ax9.plot(t, ori_dydt, label = 'data_ori_dydt')
    ax9.plot(t, upred_dy, 'x', label = 'PINNs_ori_dydt')
    ax9.set_title('pinn_ori_dydt')
    ax9.legend(loc='upper left')
    ax9.set_xlabel('t')#, fontdict=font)
    ax9.set_ylabel('dy')#, fontdict=font)
    ax9.legend(loc='upper left')
    
    fig,ax10 = plt.subplots(1,1)
    ax10.plot(output_ori[:,0], output_ori[:,1],'x', label = 'dosm')    #(100,2)
    ax10.legend(loc='upper left')
    ax10.plot(sol_lambda.y[0,:],sol_lambda.y[1,:],'x', label='IPINNs')
    ax10.set_title('DOSM vs IPINNs_Total Population')
    ax10.set_xlabel('female, x(t) (thousands)')#, fontdict=font)
    ax10.set_ylabel('male, y(t) (thousands)')#, fontdict=font)
    ax10.legend(loc='upper left')
    fig.savefig('ADMM non_sparse 43 trajectory_xy.png', dpi=300)
    
    fig,ax11 = plt.subplots(1,1)
    ax11.plot(output_ori[:,0], output_ori[:,1],'x', label = 'dosm')    #(100,2)
    ax11.legend(loc='upper left')
    ax11.plot(sol_LU.y[0,:],sol_LU.y[1,:],'x', label='SIPINNs')
    ax11.set_title('DOSM vs SIPINNs_Total Population')
    ax11.set_xlabel('female, x(t) (thousands)')#, fontdict=font)
    ax11.set_ylabel('male, y(t) (thousands)')#, fontdict=font)
    ax11.legend(loc='upper left')
    fig.savefig('LU trajectory_xy.png', dpi=300)
    
    fig,ax12 = plt.subplots(1,1)
    ax12.plot(t, ori_dxdt, label = 'data_ori_dxdt')
    ax12.plot(t, tf.transpose(end_dx), 'x', label = 'admm_ori_dxdt')
    ax12.set_title('admm_ori_dydt')
    ax12.legend(loc='upper left')
    ax12.set_xlabel('t')#, fontdict=font)
    ax12.set_ylabel('dx')#, fontdict=font)
    ax12.legend(loc='upper left')
    
    fig,ax13 = plt.subplots(1,1)
    ax13.plot(t, ori_dydt, label = 'data_ori_dydt')
    ax13.plot(t, tf.transpose(end_dy), 'x', label = 'admm_ori_dydt')
    ax13.set_title('admm_ori_dydt')
    ax13.legend(loc='upper left')
    ax13.set_xlabel('t')#, fontdict=font)
    ax13.set_ylabel('dy')#, fontdict=font)
    ax13.legend(loc='upper left')
    
    #differences between u_train vs lambda_T (PINNs-Sparse) leads to loss_sparse
    fig,ax14 = plt.subplots(1,1)
    ax14.plot(output_ori[:,0], output_ori[:,1],'x', label = 'data')    #(100,2)
    ax14.legend(loc='upper left')
    ax14.plot(sol_sparse.y[0,:],sol_sparse.y[1,:],'x', label='lasso')
    ax14.set_xlabel('female,x (thousands)')#, fontdict=font)
    ax14.set_ylabel('male,y (thousands)')#, fontdict=font)
    ax14.legend(loc='upper left')
    
    #------------------ML prediction validation------------
    fig,ax15 = plt.subplots(1,1)
    ax15.plot(out_ori_full[:,0], out_ori_full[:,1],'x', label = 'data')    #(100,2)
    ax15.legend(loc='upper left')
    ax15.plot(sol_ml.y[0,:],sol_ml.y[1,:],'x', label='IPINNs_fore')
    ax15.set_title('IPINNs forecast')
    ax15.set_xlabel('female, x (thousands)')#, fontdict=font)
    ax15.set_ylabel('male, y (thousands)')#, fontdict=font)
    ax15.legend(loc='upper left')
    fig.savefig('ML validation_xy.png', dpi=300)
    
    #last 10 error
    #sol_lambda --> 43 years
    #sol_ml 10 yrs compare with output_full
    
    fig,ax16 = plt.subplots(1,1)
    ax16.plot(t_full, out_ori_full[:,0], 'x', label = 'full_x')
    ax16.plot(t_full, sol_ml.y[0], 'x', label = 'ml_x')
    ax16.set_title('forecast_x')
    ax16.legend(loc='upper left')
    ax16.set_xlabel('t')#, fontdict=font)
    ax16.set_ylabel('x')#, fontdict=font)
    ax16.legend(loc='upper left')
  
    fig,ax17 = plt.subplots(1,1)
    ax17.plot(t_full, out_ori_full[:,1], 'x', label = 'full_y')
    ax17.plot(t_full, sol_ml.y[1], 'x', label = 'ml_y')
    ax17.set_title('forecast_y')
    ax17.legend(loc='upper left')
    ax17.set_xlabel('t')#, fontdict=font)
    ax17.set_ylabel('y')#, fontdict=font)
    ax17.legend(loc='upper left')
    
    fig,ax18 = plt.subplots(1,1)
    ax18.plot(out_ori_full[-9:,0], out_ori_full[-9:,1],'x', label = 'data')    #(100,2)
    ax18.legend(loc='upper left')
    ax18.plot(sol_ml.y[0,-9:],sol_ml.y[1,-9:],'x', label='ml')
    ax18.set_title('last ten forecast')
    ax18.set_xlabel('last ten x')#, fontdict=font)
    ax18.set_ylabel('last ten y')#, fontdict=font)
    ax18.legend(loc='upper left')
    fig.savefig('ML validation_last ten.png', dpi=300)
    
    forex_loss = ori_x_full[-10:] - sol_ml.y[0,-10:]
    forey_loss = ori_y_full[-10:] - sol_ml.y[1,-10:]
    
    f_l = tf.reduce_mean(tf.square(forex_loss) + tf.square(forey_loss))
    #print('x loss', forex_loss)
    #print('y loss', forey_loss)
    #print('square_x', tf.square(forex_loss))
    #print('square_y', tf.square(forey_loss))
    print('MSE for forecasting 10 years is', f_l)
    
    #----------calculation below not correct because MSE was divided by 20
    #aa = tf.transpose(out_ori_full [-10:,:]) - sol_ml.y[:,-10:]
    #fore_loss = tf.reduce_mean(tf.square(aa))  
    #print('diff', aa)
    #print('square_all', tf.square(aa))
    #print('Total forecast loss is:', fore_loss)
    #--------------------
    
    #----------future 37 years
    fig,ax17 = plt.subplots(1,1)
    fig.suptitle("IPINNs_Forecast Total Population", fontsize=12)
    ax17.plot(forecast_fm[:,0], forecast_fm[:,1],'x', label = 'dosm')    #(100,2)
    ax17.legend(loc='upper left')
    ax17.plot(sol_future.y[0,:],sol_future.y[1,:],'x', label='IPINNs')
    ax17.set_xlabel('female, x(t) (thousands)')#, fontdict=font)
    ax17.set_ylabel('male, y(t) (thousands)')#, fontdict=font)
    ax17.legend(loc='upper left')
    fig.savefig('forecast xy.png', dpi=300)
    
    fig,ax18 = plt.subplots(1,1)
    fig.suptitle("IPINNs_Forecast Female Population", fontsize=12)
    ax18.plot(fore_t, forecast_fm[:,0], 'x', label = 'dosm_x')
    ax18.plot(fore_t, sol_future.y[0], 'x', label = 'IPINNs_x')
    #ax18.set_title('forecast_female, x')
    ax18.legend(loc='upper left')
    ax18.set_xlabel('year, t')#, fontdict=font)
    ax18.set_ylabel('female, x(t) (thousands)')#, fontdict=font)
    ax18.legend(loc='upper left')
    fig.savefig('forecast x.png', dpi=300)
    
    fig,ax19 = plt.subplots(1,1)
    fig.suptitle("IPINNs_Forecast Male Population", fontsize=12)
    ax19.plot(fore_t, forecast_fm[:,1], 'x', label = 'dosm_y')
    ax19.plot(fore_t, sol_future.y[1], 'x', label = 'IPINNs_y')
    #ax19.set_title('forecast_male, y')
    ax19.legend(loc='upper left')
    ax19.set_xlabel('year, t')#, fontdict=font)
    ax19.set_ylabel('male, y(t) (thousands)')#, fontdict=font)
    ax19.legend(loc='upper left')
    fig.savefig('forecast y.png', dpi=300)
   

    
def save_output(libdata, fname="ps_pop_xy_4var.pkl"):
    with open(fname, 'bw') as f:
        pickle.dump(libdata,f)

def main():
    libdata = make_data()
    
    model = setting_model(libdata)
    tmp, model = pinn(libdata, model)
    prepare_data(libdata)
    
    sparse(libdata)
    
    various_opt(libdata)
    sing_eigen_values(libdata)
    get_admm_sol(libdata)
    get_LU_sol(libdata)
    
    lass_fit(libdata)
    get_lasso_sol(libdata)
    
    gradcheck(libdata)
    visualize(libdata)
    
    save_output(libdata, fname="ps_pop_xy_4var.pkl")
    
    elapsed = time.time() - start_time 
    print('Training time: %.2f' % (elapsed))
    
    return libdata, model
    
libdata, model = main()



