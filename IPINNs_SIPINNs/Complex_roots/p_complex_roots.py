#Created on 3rd May 2025

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pickle
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense, Concatenate #, Lambda
from tensorflow.keras import backend as K
from tensorflow import keras
from box import Box

start_time = time.time() 
epochs = 10
batch_size = 4
losses_pinn = []

data = pd.read_csv("ode_dataRHS_complex_roots.csv")
sim_data = tf.convert_to_tensor(data.values)

def make_data():    
    # global data
    libdata ={
        
           't' : sim_data[:,0],
           'x' : sim_data[:,1],
           'y' : sim_data[:,2],
           
           'dxdt_given' : sim_data[:,3],
           'dydt_given' : sim_data[:,4],
           'ddt':[],
           
           'model': None, 
           'fname':None,
           
           #'loss_':[],
           #'loss_ic':[],
           #'loss_f':[],
           #'error_vec':[],
           
           't_vec' : [],
           'x_vec' : [],
           'y_vec' : [],
           'output': [],
           
           #'u_fit': [],
           'u_pinn':[],
           
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
    model.add(Dense(8, activation=tf.keras.activations.gelu,kernel_initializer="glorot_uniform", dtype=tf.float64))
    model.add(Dense(16, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(32, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
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
    
    libdata['t_vec'] = t_vec
    libdata['ddt'] = ddt
    
    model.save('pinn_model.keras')
    
    return model

def train_model(libdata, model):
    t_vec = libdata['t_vec']
    x = libdata['x']
    y = libdata['y']
    output = np.c_[x,y]
    
    x_vec = np.array(x).reshape(-1,1)
    y_vec = np.array(y).reshape(-1,1)
    
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
    
    loss_x = tf.reduce_mean(tf.square(x_vec - u_pinn[:,0]))
    loss_y = tf.reduce_mean(tf.square(y_vec - u_pinn[:,1]))
    loss_data = loss_x + loss_y
    
    print('Data loss on x', loss_x)
    print('Data loss on y',loss_y)
    print('Total_Data_loss is', loss_data)
    
    
    libdata['x_vec'] = x_vec
    libdata['y_vec'] = y_vec
    libdata['output'] = output
    libdata['u_pinn'] = u_pinn
    
    return u_pinn, model  

def visualize(data):
    output = data['output']
    u_pinn = data['u_pinn']
    x_vec = data['x_vec']
    y_vec = data['y_vec']
    
    #differences between u_train (observed solution) vs u_pinn (predicted solution) leads to loss_pinn
    fig,ax2 = plt.subplots(1,1)
    fig.suptitle("Trajectory of data", fontsize=12)
    #----- not suppose to plot output x,y?-----
    ax2.plot(output[:,0], output[:,1], label='data')  
    ax2.legend(loc='upper left')
    ax2.plot(u_pinn[:,0],u_pinn[:,1],'x', label='PINNs')  
    ax2.set_xlabel('x')#, fontdict=font)
    ax2.set_ylabel('y')#, fontdict=font)
    ax2.legend(loc='upper left')
    fig.savefig('solutions trajectory.png', dpi=300)
     
    fig,ax3 = plt.subplots(1,1)
    fig.suptitle("x(t) and y(t) of data", fontsize=12)
    ax3.plot(x_vec, label='x(t)')
    ax3.plot(y_vec,label='y(t)')
    ax3.legend(loc='upper left')
    fig.savefig('data trajectory.png', dpi=300)
    
    fig,ax4 = plt.subplots(1,1)
    fig.suptitle("x(t) and y(t) of u_pinn", fontsize=12)
    ax4.plot(u_pinn[:,0], label='x(t)')
    ax4.plot(u_pinn[:,1],label='y(t)')
    ax4.legend(loc='upper left')
    fig.savefig('u_pinn trained data trajectory.png', dpi=300)
    
    fig,ax5 = plt.subplots(1,1)
    fig.suptitle("Trajectory of u_pinn", fontsize=12)
    ax5.plot(u_pinn[:,0],u_pinn[:,1],'x', label='PINNs')  
    ax5.set_xlabel('x')#, fontdict=font)
    ax5.set_ylabel('y')#, fontdict=font)
    ax5.legend(loc='upper left')
    fig.savefig('pinns_sol.png', dpi=300)

def save_output(libdata, fname="p_complex.pkl"):
    with open(fname, 'bw') as f:
        pickle.dump(libdata,f)
        
def main():
    libdata = make_data()     
    
    model = setting_model(libdata)
    tmp, model = train_model(libdata, model)
    model_pkl_file = "pinn_model.pkl"

    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(model, file)
    
    #loss_ic(data)
    #loss_ODE(data)
    
    visualize(libdata)
    save_output(libdata, fname="p_complex.pkl")
    
    elapsed = time.time() - start_time 
    print('Training time: %.2f' % (elapsed))
    
    #libdata['model'] = pmodel
    
    return libdata, model

#if __name__ == "__main__":
#    main(layers, fname='recode_spiral.pkl')

libdata,model = main()
#libdata = main()
