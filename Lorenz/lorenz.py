# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 17:29:27 2026

3D-Lorenz
"""

import numpy as np
import tensorflow as tf
import scipy.integrate as scint
import matplotlib.pyplot as plt
import time
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense #, Lambda
from tensorflow.keras import backend as K
from sklearn.metrics import explained_variance_score as evs
import pickle
#import pandas as pd
#import time
#import scipy.optimize
#import keras
#from keras.utils.vis_utils import plot_model
#from pinn import Sequentialmodel

from box import Box

#pip install python-box
start_time = time.time() 
# Configuration
t0, t1, gap= 0, 100, 800  #step size 0.05
sampling = 600

sigma, beta, rho = 10.0, 8.0/3.0, 28.0 # Standard Lorenz parameters
y0=[0.96, -1.1, 0.5]
y0_shift=[0.98, -1.11, 0.6]

#layers = np.array([1,3,2]) 
N_f = 100
epochs = 50
batch_size = 8

a=0.25
b=4
F=8.0
h=1.0

def make_data():    
    # global data
    data ={
           'y0': y0,   
           'numerical_sol': [],
           
           't':np.linspace(t0, t1, gap),
           't_train':[],
           'N_f': N_f,
           't_f':[], 
           
           'model': None, 
           'fname':None,
           
           'loss_':[],
           'loss_ic':[],
           'loss_f':[],
           'error_vec':[],
                      
           'u_fit': [],
           'u_pred':[],
           'u_test': [],
           'numerical_rhs':[],
           'grads':[],
           
           'y_numerical_idx':[],        #checking
           'numerical_neighbour': [],   #checking
           }
    return Box(data)

# x' = RHS(x,t)
def RHS(t, X, sigma, beta, rho):
    x, y, z = X
    
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    
    #x, y, z = xyz
    #x_dot = s*(y - x)
    #y_dot = r*x - y - x*z
    #z_dot = x*y - b*z
    
    return np.array([dxdt, dydt, dzdt])

# def lorenz(xyz, *, s=10, r=28, b=2.667):
#     """
#     Parameters
#     ----------
#     xyz : array-like, shape (3,)
#        Point of interest in three-dimensional space.
#     s, r, b : float
#        Parameters defining the Lorenz attractor.

#     Returns
#     -------
#     xyz_dot : array, shape (3,)
#        Values of the Lorenz attractor's partial derivatives at *xyz*.
#     """
#     x, y, z = xyz
#     x_dot = s*(y - x)
#     y_dot = r*x - y - x*z
#     z_dot = x*y - b*z
#     return np.array([x_dot, y_dot, z_dot])

def RHS2(t, x):
    dxdt = -x[1]**2 - x[2]**2 - a*x[0] + a*F 
    #print('rhs dxdt is', dxdt)     
    dydt = x[0]*x[1] - b*x[0]*x[2] -x[1] + h
    #print('rhs dydt is', dydt)
    #print('rhs dx_dy_dt is', [dxdt, dydt])
    dzdt = b*x[0]*x[1] + x[0]*x[2] - x[2]
    return np.c_[dxdt, dydt, dzdt]

# Numerical solution
def get_numerical_sol(data,
        fun=RHS,
        t_span=(t0, t1),
        y0=y0,
        method="LSODA",  # "BDF", "LSODA", "DOP853"
        t_eval=np.linspace(t0, t1, gap)
        ):
    sol = scint.solve_ivp(fun, t_span, y0, method, t_eval, args=(sigma, beta, rho))
    
    data['numerical_sol'] = sol   #(3,100)
    return sol

def prepare_input_data(data, N_f):
    t = data['t']
    idx = np.random.choice(t.shape[0], sampling-1, replace=False)
    
    t_random = t[idx]
    t_train = np.array(t_random)
    t_train = np.hstack((0,t_random))
    t_train = np.array(t_train).reshape(-1,1)
    
    numerical_sol = data['numerical_sol']
    y_numerical_idx = numerical_sol.y.T[idx,:]
    y_numerical_idx = np.vstack((y0, y_numerical_idx))    
       
    data['t_train'] = t_train
    data['y_numerical_idx'] = y_numerical_idx   #(60,3)
    
    return t_train, y_numerical_idx   

def neighbour_data(data):
    
    t_train = data['t_train']
    
    '''Additional Random neighbouring Points'''
    #T_randpts = np.random.uniform(0,1,(N_f,2))  #based on t             #(N_f,2)
    t_randpts = tf.random.uniform(shape=[N_f], minval=t0, maxval=t1)    
      
    t_f = np.array(t_randpts).reshape(-1,1)    # append random points to extend training points   #(N_f,2)
    t_f = np.vstack((t_f, t_train))                   #(N_f+1,2)
    
    sort_t_f=sorted(set(t_f.ravel()))  #set to eliminate duplicate t selected
    t_eval=sort_t_f
    
    sol2 = scint.solve_ivp(fun=RHS, t_span=(t0,t1),y0=y0, method = 'LSODA',t_eval=t_eval, args=(sigma, beta, rho))

    data['t_f'] = t_f
    data['numerical_neighbour'] = sol2      #(3,79)
    
    return t_f, sol2

def setting_model(data):
    
    t = data['t']
    sol = data['numerical_sol']
    t_train = data['t_train']
    sol_idx = data['y_numerical_idx']
    t_f = data['t_f']
    numerical_neighbour = data['numerical_neighbour']
    
    model = Sequential()
    model.add(InputLayer(input_shape=(1,),dtype=tf.float64))
    model.add(Dense(8, activation=tf.keras.activations.gelu,kernel_initializer="glorot_uniform", dtype=tf.float64))
    model.add(Dense(16, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(32, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(64, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(32, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(16, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(8, activation=tf.keras.activations.gelu, kernel_initializer="glorot_uniform", dtype=tf.float64))
    model.add(Dense(3, activation=None, kernel_initializer="glorot_normal",dtype=tf.float64))

    model.summary()
    #plot_model(model,show_shapes=True)
    #model.trainable_variables
    #data['model'] = model
    
    ddt = RHS2(t_f,numerical_neighbour.y)
    print('ddt is',ddt)
    #print('dxdt_rhs', dxdt_rhs)
    #print('dydt_rhs',dydt_rhs)
    
    #g = tf.Variable(t_f, dtype = 'float64', trainable = False)
    g = tf.convert_to_tensor(t_f)
    
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(g)
        
        #model1 = load_model('recode_spiral_model.h5')
        u_eval = model(g)
        #print(u_eval)
        grads = tape.jacobian(u_eval, g)  
        #print(grads.values)
        grads_select = tf.einsum('bxby->bxy',grads) #only one entry, the rest are zero
        grads_final = grads_select[:,:]
        print(grads_final)  #(100,2,1)
        
        dx_dt = grads_final[:,0]
        #print('dxdt is', dx_dt)
        dy_dt = grads_final[:,1]
        #print('dydt is', dy_dt)
        dz_dt = grads_final[:,2]
        #print('dydt is', dy_dt)
                
        f1 = dx_dt - ddt[:,0] 
        #print('dxdt loss is', f1)
        f2 = dy_dt - ddt[:,1]
        #print('dydt loss is', f2)
        f3 = dz_dt - ddt[:,2]
        #print('dydt loss is', f2)
        
        allgrads = grads[:,:,0,0]
        print(allgrads)
        #dx_dt = allgrads[:,0:1]
        #dy_dt = allgrads[:,1:2]
      
        #[dl_dw, dl_db] =tape.gradient(loss, [w, b])
    
    def custom_loss(y_true, y_pred):
        
        to_take = sampling + N_f - 5
        loss_ic = K.mean(K.square(y_true[:to_take] - y_pred[:to_take]))
               
        #dx_dt = tf.reduce_mean(grads_final[:,0])
        #dy_dt = tf.reduce_mean(grads_final[:,1])
        #dx_dt = allgrads[:,0:1]
        #dy_dt = allgrads[:,1:2]
        
        #loss_ODE from training data
        #f1 = dx_dt - dxdt_rhs 
        #f2 = dy_dt - dydt_rhs

        loss_f = tf.reduce_mean(tf.square(f1)+tf.square(f2))
 
        loss = loss_ic + loss_f
        
        tf.print('loss on ic is:', loss_ic)
        tf.print('loss on ODE is:', loss_f)
        tf.print('total loss is:', loss)
        
        return loss
    
    """3 steps: compile, fit, predict.       """
    #model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
          
    """model fitting"""
    t_to_array = tf.constant(t_train)
    y_to_fit = tf.constant(sol_idx)
    #print(y_to_fit)
    
    x_test = sol.y.T[0]
    y_test = sol.y.T[1]
    z_test = sol.y.T[2]
    model.fit(t_to_array, y_to_fit, batch_size,epochs, validation_data = (x_test, y_test, z_test))
    
    #model.save('recode_spiral_model.h5')
    
    """model prediction      """
    u_pred = model(t_f)
   
    #Checking performance of model with all t as testing data
    t=np.array(t).reshape(-1,1)
    u_test = model(t)
    
    ic_x = tf.reduce_mean(tf.square(sol.y.T[:,0] - u_test[:,0]))
    ic_y = tf.reduce_mean(tf.square(sol.y.T[:,1] - u_test[:,1]))
    ic_z = tf.reduce_mean(tf.square(sol.y.T[:,2] - u_test[:,2]))
    ic_all = ic_x + ic_y + ic_z
    
    ddt_all = RHS2(t,sol.y.T)
    
    h = tf.convert_to_tensor(t)
        
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(h)
                 
        #model1 = load_model('recode_spiral_model.h5')
        u_fit = model(h)
        #print(u_test)
        grads = tape.jacobian(u_fit, h)  
        grads_select = tf.einsum('bxby->bxy',grads) #only one entry, the rest are zero
        grads_final = grads_select[:,:]
            #print(grads.values)
        #allgrads = grads[:,:,0,0]
        dx_dt = grads_final[:,0]
        dy_dt = grads_final[:,1]
        dz_dt = grads_final[:,2]
    
        f_all1 = tf.reduce_mean(tf.square(dx_dt - ddt_all[:,0]))
        f_all2 = tf.reduce_mean(tf.square(dy_dt - ddt_all[:,0]))
        f_all3 = tf.reduce_mean(tf.square(dz_dt - ddt_all[:,0]))
        
        loss_fall = f_all1 + f_all2 + f_all3

        print('Derivatives loss on testing data dxdt', f_all1)
        print('Derivatives loss on testing data dydt',f_all2)
        print('Derivatives loss on testing data dzdt',f_all3)
        print('Data loss', ic_all)
        print('Total testing data loss', loss_fall + ic_all)
     
    model.save('lorenz_model.keras')
    
    data['u_pred'] = u_pred
    data['u_test'] = u_test
    
    return u_pred, u_test, model
    
def loss_ic(data):
    sol_neighbour = data['numerical_neighbour']
    u_pred = data['u_pred']
    
    to_take = sampling + N_f - 10
    loss_ic = tf.reduce_mean(tf.square(sol_neighbour.y.T[:to_take] - u_pred[:to_take]))
        
    data['loss_ic'] = loss_ic
    return loss_ic

def loss_ODE(data):
    
    t_f = data['t_f']
    numerical_neighbour = data['numerical_neighbour']
    model = data['model']
    #u_pred = data['u_pred']
        
    g = tf.Variable(t_f, dtype = 'float64', trainable = False)
    #h = tf.Variable(u_pred)
    #x = tf.Variable(u_pred)
     
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(g)
       # tape.watch(x)
         
        #model1 = load_model('recode_spiral_model.h5')
        u_eval = model(g)
        
        grads = tape.jacobian(u_eval, g)  
        
        allgrads = grads[:,:,:,0,0,0]
        dx_dt = allgrads[:,0:1]
        dy_dt = allgrads[:,1:2]
        dz_dt = allgrads[:,2:3]
        #x = tf.Variable(u_eval[:,0])
       # y = tf.Variable(u_eval[:,1])
        #dx_dt = tape.gradient(u_eval,g)        
        #dx_dt, dy_dt = tape.gradient((x,y), g)
        
   #     del(tape)
        #[dl_dw, dl_db] =tape.gradient(loss, [w, b])
    
    dxdt_rhs, dydt_rhs, dzdt_rhs = RHS(t_f,numerical_neighbour.y.T)
    
    f1 = dx_dt - dxdt_rhs 
    f2 = dy_dt - dydt_rhs
    f3 = dz_dt - dzdt_rhs
    loss_f = tf.reduce_mean(tf.square(f1)+tf.square(f2)+tf.square(f3))
 
    data['grads'] = grads
    data['loss_f'] = loss_f
    return loss_f

def visualize(data):
    numerical_sol = data['numerical_sol']
    y_numerical_idx = data['y_numerical_idx']
    numerical_neighbour = data['numerical_neighbour']
    u_pred = data['u_pred']
    u_test = data['u_test']
    t_f = data['t_f']
    
    #plt.subplot(131)
    #plt.plot(numerical_sol.y[0,:], numerical_sol.y[1,:])
    #plt.subplot(132)
    #plt.plot(y_numerical_idx[:,0], y_numerical_idx[:,1],'x')
    #plt.subplot(133)
    #plt.plot(numerical_neighbour.y[0,:], numerical_neighbour.y[1,:])
    #plt.show()
    
    # fig,ax1 = plt.subplots(1,1)
    # fig.suptitle("Results comparison between RK45 and PINNs with training data", fontsize=12)
    # ax1.plot(numerical_neighbour.y[0,:], numerical_neighbour.y[1,:], numerical_neighbour.y[2,:],label='RK45')
    # ax1.legend(loc='upper left')
    # ax1.plot(u_pred[:,0],u_pred[:,1], u_pred[:,2],'x',label='PINNs')
    # ax1.legend(loc='upper left')
    # plt.show()
    
    # fig,ax2 = plt.subplots(1,1)
    # fig.suptitle("RK45 vs PINNs_x", fontsize=12)
    # ax2.plot(numerical_neighbour.y[0,:],label='RK45')
    # ax2.legend(loc='lower left')
    # ax2.plot(u_pred[:,0],'x',label='PINNs_x')
    # ax2.legend(loc='lower left')
    # ax2.set_xlabel('epoch')#, fontdict=font)
    # ax2.set_ylabel('x(t)')#, fontdict=font)
    # fig.savefig('x.png', dpi = 300)
    # plt.show()
    
    # fig,ax3 = plt.subplots(1,1)
    # fig.suptitle("RK45 vs PINNs_y", fontsize=12)
    # ax3.plot(numerical_neighbour.y[1,:],label='RK45')
    # ax3.legend(loc='lower left')
    # ax3.plot(u_pred[:,1],'x',label='PINNs_y')
    # ax3.legend(loc='lower left')
    # ax3.set_xlabel('epoch')#, fontdict=font)
    # ax3.set_ylabel('y(t)')#, fontdict=font)
    # fig.savefig('y.png', dpi = 300)
    # plt.show()
    
    # fig,ax4 = plt.subplots(1,1)
    # fig.suptitle("RK45 vs PINNs_z", fontsize=12)
    # ax4.plot(numerical_neighbour.y[2,:],label='RK45')
    # ax4.legend(loc='lower left')
    # ax4.plot(u_pred[:,2],'x',label='PINNs_z')
    # ax4.legend(loc='lower left')
    # ax4.set_xlabel('epoch')#, fontdict=font)
    # ax4.set_ylabel('z(t)')#, fontdict=font)
    # fig.savefig('z.png', dpi = 300)
    # plt.show()


    fig,ax5 = plt.subplots(figsize=(10, 8))
    # Add a 3D subplot
    ax5 = fig.add_subplot(111, projection='3d')
    x, y, z = numerical_sol.y
    # Plot the trajectory
    ax5.plot(x, y, z, lw=0.5, color='blue') #
    # Set labels and title
    ax5.set_xlabel("X Axis")
    ax5.set_ylabel("Y Axis")
    ax5.set_zlabel("Z Axis")
    ax5.set_title("The Lorenz Attractor")
    # Display the plot
    plt.show()
    fig.savefig('lorenz_num.png', dpi=300)
    
    fig,ax6 = plt.subplots(figsize=(10, 8))
    # Add a 3D subplot
    ax6 = fig.add_subplot(111, projection='3d')
    x, y, z = numerical_neighbour.y
    # Plot the trajectory
    ax6.plot(x, y, z, lw=0.5, color='blue') #
    # Set labels and title
    ax6.set_xlabel("X Axis")
    ax6.set_ylabel("Y Axis")
    ax6.set_zlabel("Z Axis")
    ax6.set_title("The Lorenz")
    # Display the plot
    plt.show()
    fig.savefig('lorenz_pinn.png', dpi=300)
    
    #fig,ax5 = plt.subplots(1,1)
    #fig.suptitle("comparison", fontsize=12)
    #ax5.plot(t_f, numerical_neighbour.y[0,:],label='RK45_x')
    #ax5.plot(t_f, numerical_neighbour.y[1,:],label='RK45_y')
    #ax5.plot(t_f, numerical_neighbour.y[2,:],label='RK45_z')
    #ax5.plot(t_f, u_pred[:,0],label='RK45_x')
    #ax5.plot(t_f, u_pred[:,1],label='RK45_y')
    #ax5.plot(t_f, u_pred[:,2],label='RK45_z')
    
    #ax5.legend(loc='upper left')
    #ax5.plot(u_pred[:,2],'x',label='PINNs_z')
    #ax5.legend(loc='upper left')
    #plt.show()

    #ax5.plot(t, output_ori[:,0], label = 'data_x')
    #ax5.plot(t, u_pred[:,0], 'x', label = 'PINNs_x')
    
    #fig,ax2 = plt.subplots(1,1)
    #fig.suptitle("Results comparison between RK45 and PINNs with testing data", fontsize=12)
    #ax2.plot(numerical_sol.y[0,:], numerical_sol.y[1,:], numerical_sol.y[2,:], label='RK45')
    #ax2.legend(loc='upper left')
    #ax2.plot(u_test[:,0],u_test[:,1], u_test[:,2],'x',label='PINNs')
    #ax2.legend(loc='upper left')
    #plt.show()

def save_output(data, fname="lorenz.pkl"):
    with open(fname, 'bw') as f:
        pickle.dump(data,f)
        
def main():
    data = make_data()   
    get_numerical_sol(data)
    prepare_input_data(data, N_f=N_f)
    neighbour_data(data)
        
    tmp, tmp, model = setting_model(data)
    #loss_ic(data)
    #loss_ODE(data)
    
    visualize(data)
    save_output(data, fname="lorenz.pkl")
    
    elapsed = time.time() - start_time 
    print('Training time: %.2f' % (elapsed))
    
    return data, model

#if __name__ == "__main__":
#    main(layers, fname='recode_spiral.pkl')

data,model = main()
#data=main()

#save model

