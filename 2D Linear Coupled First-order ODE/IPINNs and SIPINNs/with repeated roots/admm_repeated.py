# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 16:27:41 2025

admm_repeated_roots
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import scipy as sp
import scipy.integrate as scint
import matplotlib.pyplot as plt
import time
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense #, Lambda
from tensorflow.keras import backend as K
from sklearn.metrics import explained_variance_score as evs
import pickle
import pywt #pywavelets thresholding

from sklearn.linear_model import Lasso
from box import Box

start_time = time.time() 
t0, t1 = 0, 10
#gap = 100
y0=[-0.1,0.3]
lamda = 115e-2
rho = tf.constant(0.985, dtype=tf.float64)
#miu = -0.3

batch_size = 4
epochs = 15
itera = 10

losses_sparse = []

α = 1e-3 #lasso weight search for dxdt
α2 = 1e-3 #lasso weight search for dydt 
α3 = 0.01 #weight for train sparse dydt

def make_data():    
    with open ('p_repeated.pkl','rb') as f:
        data = pickle.load(f)

    # global data
    p_data ={
           #'x': data.u_pred[0:,0],
           #'y': data.u_pred[0:,1],
           'u_pinn': data.u_pinn,     
           't': data.t,
           'x': data.x,
           'y': data.y,
           't_vec': data.t_vec,          
           'x_vec': data.x_vec,          
           'y_vec': data.y_vec,          
           'output': data.output,  
           'ddt': data.ddt,
           'model_pinn': data.model,
           
           'model': None,           
           'u_pinn_admm':[],
           
           'w1':[],
           'w2':[],
           'iv':[],
           'w_sparse1':[],
           'w_sparse2':[],
           'sol_sparse':[],
           
           'phi_array':[],
           'lambda_T' : [],
           'z_k' : [],
           'y_coeff' : [],
           'lambda_phi':[],
           'lam_sparse':[],
           
           'sol_lambda':[],
           
           'term1_tra' :[],
           'term2_lamz' :[],

           'y_lam' : [],
           'y_lam_changes' : [],
           'rho_lam_change' : [],
           'L_grad' : [],
           
           #'loss_pinn':[],
           'losses_pinn':[],
           #'loss_sparse':[],
           'losses_sparse':[],
           #'dx_dt':[],
           #'dy_dt':[],
           
           }
    return Box(p_data)

def custom_loss(y_true, y_pred):
    t_vec = p_data['t_vec']
    dxdt_given = p_data['dxdt_given']
    dydt_given = p_data['dydt_given']
    
    dxdt_vec = np.array(dxdt_given).reshape(-1,1)
    dydt_vec = np.array(dydt_given).reshape(-1,1)
    
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

    loss = tf.cast(loss_ic, tf.float32) + tf.cast(loss_f, tf.float32)
     
    tf.print('loss on ic is:', loss_ic)
    tf.print('loss on derivative(GTape) is:', loss_f)
    tf.print('total loss is:', loss)
     
    return loss
 
model = load_model('pinn_model.keras',custom_objects={'custom_loss': custom_loss})
model.summary()

def prepare_data(p_data):    
    #t_vec = p_data['t_vec']    #the sampling input data, total 100
       
    u_vec = tf.ones([6,1])
    v_vec = tf.ones([6,1])
    lammda = np.c_[u_vec,v_vec]
    lambda_T = lammda.T   #dtype:float32
    lambda_T = tf.cast(lambda_T,dtype=tf.float64) #convert tensor to tensor
    #print(matrix_T)
    y_coeff = tf.cast(tf.ones([2,6]),dtype=tf.float64)
    z_T = tf.convert_to_tensor(np.random.rand(2,6), dtype=tf.float64)
          
    #df.to_excel('output.xlsx', index=False)
    
    p_data['lambda_T'] = lambda_T  #lambda transpose 
    p_data['z_T'] = z_T
    p_data['y_coeff'] = y_coeff
    
def compute_data(p_data):
    #t_shape = var_data['t_shape']    #the only input data, total 100
    #t_train = var_data['t_train']   #total 60
    
    #t_vec = p_data['t_vec']
    x = p_data['x']
    y = p_data['y']
    x2 = tf.square(x)
    y2 = tf.square(y)   
    xy = tf.multiply(x,y)
    
    #print(x2y2)
    const_one = tf.ones([100,])
    const_one = tf.cast(const_one,dtype=tf.float64)
    #array = tf.stack([x,x_sqr,y_cube,x2y2],axis=0)
    array = tf.stack([x,y,x2,y2,xy,const_one],axis=0)
    #print(array)
             
    #array_var_T = tf.transpose(array)
    #print(array_var_T)
    
    p_data['phi_array'] = array      #shape:(3,100)

#---------------PINN-----------------#
def setting_model(p_data):
        
    model = Sequential()
    model.add(InputLayer(input_shape=(1,),dtype=tf.float64))
    model.add(Dense(8, activation=tf.keras.activations.gelu,kernel_initializer="glorot_uniform", dtype=tf.float64))
    model.add(Dense(16, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(32, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(16, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(8, activation=tf.keras.activations.gelu, kernel_initializer="glorot_uniform", dtype=tf.float64))
    model.add(Dense(2, activation=None, kernel_initializer="glorot_normal",dtype=tf.float64))

    model.summary()

    p_data['model'] = model
    
    return model

def train_PINN(p_data):
    t = p_data['t']    #the only input data, total 100
    t_shape = p_data['t_vec']
    
    sol_t = p_data['u_pinn']          #u_pinn
    x_vec = p_data['x_vec']
    y_vec = p_data['y_vec']
    #output = p_data['output']
    ddt = p_data['ddt']
    
    w_sparse1 = p_data['w_sparse1']
    w_sparse2 = p_data['w_sparse2']
    
    #model_pinn = p_data['model_pinn']
    #model = p_data['model']
    
    lambda_T = p_data['lambda_T']
    array = p_data['phi_array']
    
    losses_pinn = p_data['losses_pinn']
    
    #ddt = RHS2(t_shape,sol_t)   #numerical substitution
    #print('ddt is',ddt)
    #print('dxdt_rhs', dxdt_rhs)
    #print('dydt_rhs',dydt_rhs)
    
    lamda = np.c_[w_sparse1,w_sparse2]
    lambda_T = lamda.T
    dudt_est = tf.matmul(lambda_T, array)
    
    lam_phi = tf.matmul(lambda_T, array) #(2x100)
    print(np.shape(lam_phi))
    sub = lam_phi - tf.transpose(ddt)
    
    loss_sparse = tf.reduce_mean(tf.square(sub))
    print('loss from sparse is:', loss_sparse)  #physics loss
    losses_sparse.append(loss_sparse)
    
    fig,ax1 = plt.subplots(1,1)
    ax1.plot(losses_sparse, label = 'losses of Sparse')
    ax1.legend(loc='upper right')   
    
    #g = tf.Variable(t_f, dtype = 'float64', trainable = False)
    g = tf.convert_to_tensor(t_shape)
    
    def custom_loss(y_true, y_pred):
        
        loss_ic = K.mean(K.square(y_true - y_pred))
               
        #dx_dt = tf.reduce_mean(grads_final[:,0])
        #dy_dt = tf.reduce_mean(grads_final[:,1])
        #dx_dt = allgrads[:,0:1]
        #dy_dt = allgrads[:,1:2]
        
        #loss_ODE from training data
        #f1 = dx_dt - dxdt_rhs 
        #f2 = dy_dt - dydt_rhs

        with tf.GradientTape(persistent=False) as tape:
            tape.watch(g)
              
            #model1 = load_model('recode_spiral_model.h5')
            u_eval = model(g)
            #print(u_eval)
            grads = tape.jacobian(u_eval, g)  
            #print(grads.values)
            grads_select = tf.einsum('bxby->bxy',grads) #only one entry, the rest are zero
            grads_final = grads_select[:,:]
            #print(grads_final)  #(100,2,1)
              
            dx_dt = grads_final[:,0]
            #print('dxdt is', dx_dt)
            dy_dt = grads_final[:,1]
            #print('dydt is', dy_dt)
                      
            #f1 = dx_dt - ddt[:,0]     #from numerical
            #print('dxdt loss is', f1)
            #f2 = dy_dt - ddt[:,1]
            #print('dydt loss is', f2) #from numerical
              
            f1 = dx_dt - dudt_est[0:1,:]   #lambda*phi not trained
            f2 = dy_dt - dudt_est[1:2,:]   #lambda*phi not trained
              
            #allgrads = grads[:,:,0,0]
            #print(allgrads)
            #dx_dt = allgrads[:,0:1]
            #dy_dt = allgrads[:,1:2]
            #[dl_dw, dl_db] =tape.gradient(loss, [w, b])
              
    
        loss_f = tf.reduce_mean(tf.square(f1)+tf.square(f2))
 
        loss = tf.cast(loss_ic, tf.float32) + tf.cast(loss_f, tf.float32)
        
        tf.print('loss on ic is:', loss_ic)
        tf.print('loss on derivative (G_tape) is:', loss_f)
        tf.print('total loss is:', loss)
        
        return loss  
    
   # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
 
   #model compilation and fitting
    t_to_array = tf.constant(t_shape)  #(100,1)
    #y_size = tf.transpose(sol_t)  #(2,100)
    y_to_fit = tf.constant(sol_t)  #(100,2)
    
    history = model.fit(t_to_array, y_to_fit, batch_size,epochs)   #should be respective t on train set
   
    #loss_pinn is decreasing?
    fig,ax2 = plt.subplots(1,1)
    ax2.plot(history.history['loss'], label = 'losses of PINNs')
    #ax3.semilogy(losses_pinn, label = 'log_losses of PINNs')  
    ax2.set_xlabel('number of iteration')#, fontdict=font)
    ax2.set_ylabel('loss value')#, fontdict=font)
    ax2.legend(loc='upper right')
    
    u_pinn_admm = model(t_shape)  
    #print(new_u[0])
    
    #ic_x = tf.reduce_mean(tf.square(sol_t[:,0] - u_pinn_admm[:,0]))
    #ic_y = tf.reduce_mean(tf.square(sol_t[:,1] - u_pinn_admm[:,1]))
    
    ic_x = tf.reduce_mean(tf.square(x_vec - u_pinn_admm[:,0]))
    ic_y = tf.reduce_mean(tf.square(y_vec - u_pinn_admm[:,1]))
    
    ic_all = ic_x + ic_y
    
    tf.print('loss on data is:', ic_all)
    
   
    
    #differences between u_train (observed solution) vs u_pinn (predicted solution) leads to loss_pinn
    # fig,ax2 = plt.subplots(1,1)
    # fig.suptitle("admm vs sol trajectory", fontsize=12)
    # #----- not suppose to plot output x,y?-----
    # ax2.plot(output[:,0], output[:,1], label='ODE_repeated_roots')  
    # ax2.legend(loc='upper left')
    # ax2.plot(u_pinn_admm[:,0],u_pinn_admm[:,1],'x', label='PINNs')  
    # ax2.set_xlabel('x')#, fontdict=font)
    # ax2.set_ylabel('y')#, fontdict=font)
    # ax2.legend(loc='upper left')
    
    #p_data['u_t'] = u_t   
    p_data['lambda_T'] = lambda_T
    p_data['u_pinn_admm'] = u_pinn_admm
    p_data['losses_sparse'] = losses_sparse
    #p_data['loss_pinn'] = loss_pinn
    #p_data['losses_pinn'] = losses_pinn
    
    return u_pinn_admm #,model

#--------lasso fitting to get initial coefficient matrix------
def lass_fit(p_data):
    #given_xy = las_data['output']
    ddt = p_data['ddt']
    x = p_data['x']
    y = p_data['y']
    
    x2 = tf.square(x)
    y2 = tf.square(y)   
    xy = tf.multiply(x,y)
    
    iv = np.c_[x,y, x2, y2, xy]
    
    dxdt = ddt[:,0]
    
    lasso1 = Lasso(alpha=α, fit_intercept=True, tol=1e-4, max_iter=1000)
    lasso1.fit(iv,dxdt)
    w1 = np.array(list(lasso1.coef_) + [lasso1.intercept_])
    print('lasso for y1(dxdt) is', w1)
    print('lasso loss for y1(dxdt) is',0.5*sum((lasso1.predict(iv)-dxdt)**2) + 1*sum(np.abs(w1)))
    
    dydt = ddt[:,1]
    
    lasso2 = Lasso(alpha=α2, fit_intercept=True, tol=1e-4, max_iter=1000)
    lasso2.fit(iv,dydt)
    w2 = np.array(list(lasso2.coef_) + [lasso2.intercept_])
    print('lasso for y2(dydt) is',w2)
    print('lasso loss for y2(dydt) is',0.5*sum((lasso2.predict(iv)-dydt)**2) + 1*sum(np.abs(w2)))
    
    p_data['w1'] = w1
    p_data['w2'] = w2
    p_data['iv'] = iv
    
    return w1, w2, iv

def prox1(w1, p_data, sigma=0.0001):
    w1 = p_data['w1']
    #print('coeff of dxdt from lasso', w1)
    mask = w1 > sigma
    return w1*mask


def prox2(w2, p_data, sigma=0.0001):
    w2 = p_data['w2']
    #print('coeff of dydt from lasso', w2)
    
    mask = np.abs(w2) > sigma
    return w2*mask

# def prox2(lambda_T, p_data, sigma=0.001):
#     lambda_T = p_data['lambda_T']
#     #print('coeff of dydt from lasso', w2)
    
#     mask = np.abs(lambda_T) > sigma
#     #mask = lambda_T > sigma
#     #print(mask)
#     return lambda_T*mask

def fun1(w1, p_data, α=1):
    iv = p_data['iv']
    ddt = p_data['ddt']
    x = p_data['x']
    y = p_data['y']
    dxdt = ddt[:,0]
    
    XX1 = np.c_[iv, np.ones_like(dxdt)]
    y_predict1 = XX1 @ w1 - np.log(np.abs(tf.multiply(x,y)))
    f1 = np.sum( (dxdt - y_predict1)**2 )
    g1 = np.abs(w1).sum()
    loss1 = 0.5*f1 + α*g1
  
    return loss1

def fun2(w2, p_data, α=1):
    iv = p_data['iv']
    ddt = p_data['ddt']
    
    dydt = ddt[:,1]
    
    XX2 = np.c_[iv, np.ones_like(dydt)]
    y_predict2 = XX2 @ w2
    f2 = np.sum( (dydt - y_predict2)**2 )
    g2 = np.abs(w2).sum()
    loss2 = 0.5*f2 + α*g2
    
    return loss2

def w1_admm_proxy(p_data):

    w1 = p_data['w1']
    iv = p_data['iv']
    ddt = p_data['ddt']
    
    ρ = 2
    xi = np.ones_like(w1)/5
    z = w1.copy()

    dxdt = ddt[:,0]
    
    XX = np.c_[iv, np.ones_like(dxdt)]
    XTX = XX.T @ XX
    A = XTX + ρ* np.eye( *XTX.shape)
    lu_piv = sp.linalg.lu_factor(A)

    losses1 = []
    for i in range(100):
        b = XX.T @ dxdt +  ρ*z - xi
        w = sp.linalg.lu_solve(lu_piv, b)
        w1_sparse = prox1(w, p_data)
        #z = soft_threshold(w1+xi/ρ, α/ρ)
        z = pywt.threshold(w1_sparse+xi/ρ, α/ρ , 'soft')
        xi = xi + ρ*(w1_sparse-z)
        losses1.append(fun1(w1_sparse, p_data))
        #print('w1_sparse here', w1_sparse)
    #plt.plot('losses on dxdt',losses1)
    
    b = XX.T @ dxdt +  ρ*z - xi
    w = sp.linalg.lu_solve(lu_piv, b)
    w1_sparse = prox1(w, p_data)
    z = pywt.threshold(w1_sparse+xi/ρ, α/ρ , 'soft')
    xi = xi + ρ*(w1_sparse-z)
    print('coeff of dxdt (admm proxy) FINAL', w1_sparse)
    print('admm (proxy): loss value from dxdt',losses1[-1])

    p_data['w_sparse1'] = w1_sparse

def w2_admm_proxy(p_data):

    w2 = p_data['w2']
    iv = p_data['iv']
    ddt = p_data['ddt']

    ρ = 2
    xi = np.ones_like(w2)/5
    z = w2.copy()

    dydt = ddt[:,1]
    
    XX = np.c_[iv, np.ones_like(dydt)]
    XTX = XX.T @ XX
    A = XTX + ρ* np.eye( *XTX.shape)
    lu_piv = sp.linalg.lu_factor(A)

    losses2 = []
    for i in range(100):
        b = XX.T @ dydt +  ρ*z - xi
        w2 = sp.linalg.lu_solve(lu_piv, b)
        w2_sparse = prox2(w2, p_data)
        #z = soft_threshold(w1+xi/ρ, α/ρ)
        z = pywt.threshold(w2_sparse+xi/ρ, α3/ρ , 'soft')
        xi = xi + ρ*(w2_sparse-z)
        losses2.append(fun2(w2_sparse, p_data))

    #plt.plot('losses on dydt',losses2)
    
    b = XX.T @ dydt +  ρ*z - xi
    w2 = sp.linalg.lu_solve(lu_piv, b)
    w2_sparse = prox2(w2, p_data)
    z = pywt.threshold(w2_sparse+xi/ρ, α3/ρ , 'soft')
    xi = xi + ρ*(w2_sparse-z)
    print('coeff of dydt (admm proxy)', w2_sparse)
    print('admm (proxy): loss value from dydt',losses2[-1])
    
    p_data['w_sparse2'] = w2_sparse

def sparse_update(p_data):
    #update on lambda_T, z, y also need to be in a loop
   
    lambda_T = p_data['lambda_T']
    ddt = p_data['ddt']
    
    array = p_data['phi_array']
    z_T = p_data['z_T']
    y_coeff = p_data['y_coeff']
    
    losses_sparse = p_data['losses_sparse']
    ddt = p_data['ddt']
    
    term1_tra = p_data['term1_tra']
    term2_lamz = p_data['term2_lamz']

    y_lam = p_data['y_lam']
    y_lam_changes = p_data['y_lam_changes']
    rho_lam_change = p_data['rho_lam_change']
    L_grad = p_data['L_grad']

    identity_matrix = tf.ones([3, 3],dtype=tf.float64)
    rhoI = tf.multiply(rho,identity_matrix)
    rhoz = tf.multiply(rho,z_T)
     
    uarr = tf.matmul(tf.transpose(ddt), tf.transpose(array))
    
    phi_phiT = tf.matmul(array, tf.transpose(array))
    ppT_rI = tf.add(phi_phiT,rhoI)
    ppTrI_inv = tf.linalg.pinv(ppT_rI)
    
    for i in range (100):
       
        subtra = tf.subtract(uarr, y_coeff)
        u_pT_t_rz = tf.add(subtra, rhoz)
        
        lambda_T = tf.matmul(u_pT_t_rz, ppTrI_inv)
        lam_sparse = prox2(lambda_T,p_data)
        #print('lams is', lam_sparse)
        
        lam_phi = tf.matmul(lam_sparse, array) #(2x100)
        sub = lam_phi - tf.transpose(ddt)

        term1 = 0.5 * ((tf.linalg.norm(sub))**2)
        term1_tra.append(term1)
        
        frac1 = np.array([lamda/rho])
        frac2 = (1/rho * y_coeff) + lam_sparse
        S = pywt.threshold(frac2, frac1 , 'soft')  #frac2:our signal; frac1: value to threshold
        
        z_k = S
        
        y_coeff = y_coeff + rho*(lam_sparse - z_k)
        
    print('lamT is',lambda_T)
    print('lamda sparse is',lam_sparse)
    print('z_k+1 is',z_k)
    print('yij_k+1 is', y_coeff)
    
    lambda_phi = tf.matmul(lam_sparse, array)
    loss_sparse = tf.reduce_mean(tf.square(tf.transpose(ddt) - lambda_phi))
    #loss_sparse = tf.reduce_mean(tf.square(ddt-ddt_pinn))
    print('loss from sparse is:', loss_sparse)  #physics loss
    losses_sparse.append(loss_sparse)
    
    y_coeff_T = tf.transpose(y_coeff) #(3x2)
    y_lam = tf.reduce_sum((tf.matmul(y_coeff_T,lambda_T))-tf.matmul(y_coeff_T,z_k))
    y_lam_changes.append(y_lam) 

    #lse_norm2 = 0.5 * (np.linalg.norm((lam_phi - tf.transpose(u_t)),ord=2))
    rho_lam = 0.5*rho*np.linalg.norm((lambda_T - z_k), ord=2)
    rho_lam_change.append(rho_lam)

    #loss_manual = term1 + lam_z + y_lam + rho_lam 
    #print('Manual Calculated Loss', loss_manual)

    array_T = tf.transpose(array)
    ut_T = tf.transpose(ddt)  #(2,100)

    dL_dlam = tf.matmul(lam_phi, array_T) - tf.matmul(ut_T, array_T) + y_coeff + rho*(lambda_T - z_k)
    dL_norm = np.linalg.norm((dL_dlam), ord=2)
    L_grad.append(dL_norm)
    #print(dL_norm)

    p_data['z_T'] = z_k
    p_data['y_coeff'] = y_coeff
    p_data['lambda_T'] = lambda_T
    p_data['lam_sparse'] = lam_sparse
    p_data['lambda_phi'] = lambda_phi
    p_data['loss_sparse'] = loss_sparse
    p_data['losses_sparse'] = losses_sparse
    p_data['term1_tra'] = term1_tra
    p_data['term2_lamz'] = term2_lamz
    p_data['y_lam_changes'] = y_lam_changes
    p_data['rho_lam_change'] = rho_lam_change
    p_data['L_grad'] = L_grad

    
def admm_sol(p_data):
    lambda_T = p_data['lam_sparse']
    t = p_data['t']
    
    def func_check(t, x):
        dxdt = lambda_T[0,0]*x[0] + lambda_T[0,1]*x[1] + lambda_T[0,2]*(x[0]**2) + lambda_T[0,3]*(x[1]**2) + lambda_T[0,4]*(x[0]*x[1]) + lambda_T[0,5]  
        dydt = lambda_T[1,0]*x[0] + lambda_T[1,1]*x[1] + lambda_T[1,2]*(x[0]**2) + lambda_T[1,3]*(x[1]**2) + lambda_T[1,4]*(x[0]*x[1]) + lambda_T[1,5] 
         
        return np.array([dxdt, dydt])
    
    sol_lambda = scint.solve_ivp(fun=func_check, t_span=(t0,t1),y0=y0, method = 'RK45',t_eval=t)
    
    p_data['sol_lambda'] = sol_lambda
    
    return sol_lambda

#dxdt = 1 + 0.2*x[0] - 0.3*x[1]      
#dydt = 2 - 0.4*x[0] + 0.5*x[1]

def get_sparse_sol(p_data):
    
    w1 = p_data['w_sparse1']
    w2 = p_data['w_sparse2']
    output = p_data['output']
    
    w = np.c_[w1,w2]
    
    def func_check(t, x):
        
        #---------------lasso----------
        y1 = w[5,0] + w[0,0]*x[0] + w[1,0]*x[1] + w[2,0]*(x[0]**2) + w[3,0]*(x[1]**2) + w[4,0]*(x[0]*x[1]) 
        y2 = w[5,1] + w[0,1]*x[0] + w[1,1]*x[1] + w[2,1]*(x[0]**2) + w[3,1]*(x[1]**2) + w[4,1]*(x[0]*x[1])
        
        return np.array([y1, y2])
    
    sol_sparse = scint.solve_ivp(fun=func_check, t_span=(0,10), y0=[-0.1,0.3], method="RK45", t_eval=np.linspace(0,10,99))
    
    # fig,ax2 = plt.subplots(1,1)
    # ax2.plot(output[:,0], output[:,1],'x', label = 'RK45')    #(100,2)
    # ax2.legend(loc='upper left')
    # ax2.plot(sol_sparse.y[0,:],sol_sparse.y[1,:],'x', label='ADMM')
    # ax2.set_xlabel('x')#, fontdict=font)
    # ax2.set_ylabel('y')#, fontdict=font)
    # ax2.legend(loc='upper left')
    
    p_data['sol_sparse'] = sol_sparse

    return sol_sparse

def visualize(p_data):
    output = p_data['output']
    u_pinn = p_data['u_pinn']
    u_pinn_admm = p_data['u_pinn_admm']
    #u_t = var_data['u_t']
    #lambda_phi = var_data['lambda_phi']
    losses_pinn = p_data['losses_pinn']
    losses_sparse = p_data['losses_sparse']
    
    sol_lambda = p_data['sol_lambda']
    sol_sparse = p_data['sol_sparse']
    term1_tra = p_data['term1_tra']
    term2_lamz = p_data['term2_lamz']
    y_lam_changes = p_data['y_lam_changes']
    rho_lam_change = p_data['rho_lam_change']
    L_grad = p_data['L_grad']
    
    t = p_data['t']
    
    #differences between u_train (observed solution) vs u_pinn (predicted solution) leads to loss_pinn
    fig,ax3 = plt.subplots(1,1)
    fig.suptitle("u_pinn vs sol trajectory ", fontsize=12)
    ax3.plot(output[:,0], output[:,1], label='ODE_repeated_roots') 
    ax3.legend(loc='upper left')
    ax3.plot(u_pinn[:,0],u_pinn[:,1],'x', label='PINNs')  
    ax3.set_xlabel('x')#, fontdict=font)
    ax3.set_ylabel('y')#, fontdict=font)
    ax3.legend(loc='upper left')
    
    #difference between exact vs u_pinn_admm
    fig,ax4 = plt.subplots(1,1)
    fig.suptitle("u_pinn_admm vs sol trajectory ", fontsize=12)
    ax4.plot(output[:,0], output[:,1], label='ODE_repeated_roots') 
    ax4.legend(loc='upper left')
    ax4.plot(u_pinn_admm[:,0],u_pinn_admm[:,1],'x', label='trained sparse')  
    ax4.set_xlabel('x')#, fontdict=font)
    ax4.set_ylabel('y')#, fontdict=font)
    ax4.legend(loc='upper left')
    
    # #differences between u_train vs lambda_T (PINNs-Sparse) leads to loss_sparse
    fig,ax5 = plt.subplots(1,1)
    fig.suptitle("admm vs sol trajectory", fontsize=12)
    ax5.plot(output[:,0], output[:,1], label='ODE_repeated_roots')     #(100,2)
    ax5.legend(loc='upper left')
    ax5.plot(sol_sparse.y[0,:],sol_sparse.y[1,:],'x', label='PINNs-Sparse') 
    ax5.set_xlabel('x')#, fontdict=font)
    ax5.set_ylabel('y')#, fontdict=font)
    ax5.legend(loc='upper left')
    
    
    
    fig,ax8 = plt.subplots(1,1)
    ax8.plot(t[0:-1], sol_sparse.y[0,:], label = 'x')
    ax8.plot(t[0:-1], sol_sparse.y[1,:], color = 'red', label = 'y')
    ax8.set_title('sparse_numerical sol')
    ax8.legend(loc='upper left')
    ax8.set_xlabel('t')#, fontdict=font)
    ax8.set_ylabel('x & y')#, fontdict=font)
    ax8.legend(loc='upper left')
    
    fig,ax9 = plt.subplots(1,1)
    ax9.plot(t, u_pinn_admm[:,0], label = 'x')
    ax9.plot(t, u_pinn_admm[:,1], color = 'red', label = 'y')
    ax9.set_title('pinn_numerical sol')
    ax9.legend(loc='upper left')
    ax9.set_xlabel('t')#, fontdict=font)
    ax9.set_ylabel('x & y')#, fontdict=font)
    ax9.legend(loc='upper left')
    
    fig,ax10 = plt.subplots(1,1)
    ax10.plot(t, output[:,0], label = 'exact_x')
    ax10.plot(t[0:-1], sol_sparse.y[0,:], 'x', label = 'x')
    ax10.set_title('sparse_x vs num')
    ax10.legend(loc='upper left')
    ax10.set_xlabel('t')#, fontdict=font)
    ax10.set_ylabel('x')#, fontdict=font)
    ax10.legend(loc='upper left')
    
    fig,ax11 = plt.subplots(1,1)
    ax11.plot(t, output[:,1], label = 'exact_y')
    ax11.plot(t[0:-1], sol_sparse.y[1,:], 'x', label = 'y')
    ax11.set_title('sparse_y_num')
    ax11.legend(loc='upper left')
    ax11.set_xlabel('t')#, fontdict=font)
    ax11.set_ylabel('y')#, fontdict=font)
    ax11.legend(loc='upper left')
    
   #loss_sparse is decreasing?
    #fig,ax6 = plt.subplots(1,1)
    #ax6.plot(losses_sparse, label = 'losses of Sparse')
    #ax6.legend(loc='upper right')   
    #plot trajectory of coupled first order ODE (especially on other time t)

    # fig,ax6 = plt.subplots(1,1)
    # ax6.plot(term1_tra, label = 'Smooth (1st)')
    # ax6.set_xlabel('number of iteration')#, fontdict=font)
    # ax6.set_ylabel('changes of lambda_T phi^T - u_t^T')#, fontdict=font)
    # ax6.legend(loc='upper right')   
    # fig.savefig('trajectory of smooth (1st).png')
    
    # fig,ax7 = plt.subplots(1,1)
    # ax7.plot(term2_lamz, label = 'lam_z (2nd)')
    # ax7.set_xlabel('number of iteration')#, fontdict=font)
    # ax7.set_ylabel('changes of lambda z')#, fontdict=font)
    # ax7.legend(loc='upper right')   
    # fig.savefig('trajectory of lambda norm1 z.png')
    
    # fig,ax8 = plt.subplots(1,1)
    # ax8.plot(y_lam_changes, label = 'y_lam (3rd)')
    # ax8.set_xlabel('number of iteration')#, fontdict=font)
    # ax8.set_ylabel('changes of y_lam')#, fontdict=font)
    # ax8.legend(loc='upper right')   
    # fig.savefig('trajectory of y_lambda z.png')
    
    # fig,ax9 = plt.subplots(1,1)
    # ax9.plot(rho_lam_change, label = 'rho_lam (4th)')
    # ax9.set_xlabel('number of iteration')#, fontdict=font)
    # ax9.set_ylabel('changes of rho_lam')#, fontdict=font)
    # ax9.legend(loc='upper right')   
    # fig.savefig('trajectory of rho_lambda z.png')
    
    # fig,ax10 = plt.subplots(1,1)
    # ax10.plot(L_grad, label = 'Grad_L')
    # ax10.set_xlabel('number of iteration')#, fontdict=font)
    # ax10.set_ylabel('L norm')#, fontdict=font)
    # ax10.legend(loc='upper right')   
    # fig.savefig('L magnitude.png')
    

def save_output(p_data, fname="admm_repeated.pkl"):
    with open(fname, 'bw') as f:
        pickle.dump(p_data, f)


    
def main():
    
    p_data = make_data() 
    prepare_data(p_data)
    compute_data(p_data)
    
    #model = setting_model(p_data)
    
    lass_fit(p_data)
    for i in range(itera):
        w1_admm_proxy(p_data)
        w2_admm_proxy(p_data)
    
        train_PINN(p_data)
        print(i)
    #sparse_update(p_data)
    
    #put a loop PINN update, sparse update
    #for i in range (itera):
        #model=setting_model(var_data)
    #     train_PINN(p_data)
    #     sparse_update(p_data)
    #     print(i)
    
    #admm_sol(p_data)
    get_sparse_sol(p_data)
    
    visualize(p_data)
    
    
    #get_numerical_sol(data)
    #prepare_input_data(data, N_f=N_f)
    #neighbour_data(data)
        
    #tmp, tmp, model = setting_model(data)
    #loss_ic(data)
    #loss_ODE(data)
        
    save_output(p_data, fname="admm_repeated.pkl")
    
    elapsed = time.time() - start_time 
    print('Training time: %.2f' % (elapsed))
    
    return p_data#, model

p_data = main()
#p_data, model = main()
#if __name__ == "__main__":
#    main(layers, fname='recode_spiral.pkl')

#data,model = main()
    
   
    #_______to update lambda_T__
    #(ut_phi-y_coeff)    size:2x3
    ##  identity_matrix = tf.ones([3, 3],dtype=tf.float64)
    ##  rhoI = tf.multiply(rho,identity_matrix)
    #rhoI1 = tf.convert_to_tensor(rhoI)
    #rhoI2 = tf.cast(rhoI1,dtype=tf.float64)
    ##  rhoz = tf.multiply(rho,z_T)
    #print(u_t)
    
    #ddt_matrix = ddt_pinn
    #u_t_matrix = u_t[:,:,0]
    #u_t_matrix = u_t[:,:]
    
    ##  uarr = tf.matmul(tf.transpose(ddt_pinn), tf.transpose(array))
    ##  subtra = tf.subtract(uarr, y_coeff)
    ##  u_pT_t_rz = tf.add(subtra, rhoz)
    
    #print(u_pT_t_rz)
    
    ##  phi_phiT = tf.matmul(array, tf.transpose(array))
    ##  ppT_rI = tf.add(phi_phiT,rhoI)
    ##  ppTrI_inv = tf.linalg.pinv(ppT_rI)
    
    ##-->   lambda_T = tf.matmul(u_pT_t_rz, ppTrI_inv)
    #print('lambda k+1 is',lambda_T)
    
    #------------1/2 ||lambda^T *phi_array - u_t^T|| (norm2 sqr)
    ##lam_phi = tf.matmul(lambda_T, array) #(2x100)
    ##sub = lam_phi - tf.transpose(ddt_pinn)

    ##term1 = 0.5 * ((tf.linalg.norm(sub))**2)
    ##term1_tra.append(term1) 
    
    #____________to update z for frac2____
    ##frac1 = np.array([lamda/rho])
    ##frac2 = (1/rho * y_coeff) + lambda_T
    ##S = pywt.threshold(frac2, frac1 , 'soft')  #frac2:our signal; frac1: value to threshold
    
    ##z_k = S
    
    ##lam_z = lamda * np.linalg.norm((z_k), ord=1)
    ##term2_lamz.append(lam_z)

    # #_________to update yij_______
    ##y_coeff = y_coeff + rho*(lambda_T - z_k)
    