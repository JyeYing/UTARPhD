import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.integrate as scint
import matplotlib.pyplot as plt
import time
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense #, Lambda
from tensorflow.keras import backend as K
from sklearn.metrics import explained_variance_score as evs
import pickle
import pywt #pywavelets thresholding

from box import Box

start_time = time.time() 
t0, t1, gap= 0, 10, 300
y0=[0,1]
lamda = 0.5
rho = tf.constant(1, dtype=tf.float64)
miu = -0.3

batch_size = 4
epochs = 6
itera = 6

def make_data():    
    with open ('recode_spiral.pkl','rb') as f:
        data = pickle.load(f)

    # global data
    var_data ={
           #'x': data.u_pred[0:,0],
           #'y': data.u_pred[0:,1],
           'u': data.u_pred,     #(80,2)
           't': data.t,          #(100,1)
           't_f': data.t_f,      #(80,1)
           't_train':data.t_train,   #(60,1)
           'numerical_sol': data.numerical_sol.y,   #(2,100)
           'sol_neighbour': data.numerical_neighbour.y,  #(2,79)
           'sol_idx': data.y_numerical_idx,    #(60,2)
           
           'sol_train':[],         
           't_shape':[],
           'u_t':[],
           
           'phi_array':[],
                     
           'model':None,
                   
           'lambda_T' : [],
           'z_k' : [],
           'y_coeff' : [],
           'y_lam' : [],
           'y_lam_changes' : [],
           'lambda_phi':[],
           
           'sol_lambda':[],
           
           'u_pinn':[],
           'loss_pinn':[],
           'losses_pinn':[],
           'loss_sparse':[],
           'losses_sparse':[],
           'dx_dt':[],
           'dy_dt':[],
           
           }
    return Box(var_data)

#x' = RHS(x,t)
def RHS(t, x):
    dxdt = -x[1] + miu*(x[0]**2)*(1-((x[0]**2)+(x[1]**2)))      
    dydt = x[0] + miu*x[1]*(1-((x[0]**2)+(x[1]**2))) 
    
    return np.array([dxdt, dydt])

# def RHS(t, x):
#     dxdt = -x[1] + miu*(x[0]**2)*(1-((x[0]**2)+(x[1]**2)))      
#     dydt = x[0] + miu*x[1]*(1-((x[0]**2)+(x[1]**2))) 
    
#     return np.array([dxdt, dydt])


# def RHS2(t, x):
#     dxdt = 1 + 0.2*x[0,:] - 0.3*x[1,:] 
#     #print('rhs dxdt is', dxdt)     
#     dydt = 2 - 0.4*x[0,:] + 0.5*x[1,:]
#     #print('rhs dydt is', dydt)
#     #print('rhs dx_dy_dt is', [dxdt, dydt])
#     return np.c_[dxdt, dydt]

def prepare_data(var_data):
    #sort_t=sorted(set(t.flatten()))  #data frame method to reshape
    #t_eval=sort_t
    #sol_train = scint.solve_ivp(fun=RHS, t_span=(t0,t1),y0=y0, method = 'RK45',t_eval=t_eval)
    
    t = var_data['t']    #the sampling input data, total 100
    t_shape = np.array(t).reshape(-1,1)
    
    u_vec = tf.ones([3,1])
    v_vec = tf.ones([3,1])
    lammda = np.c_[u_vec,v_vec]
    lambda_T = lammda.T   #dtype:float32
    lambda_T = tf.cast(lambda_T,dtype=tf.float64) #convert tensor to tensor
    #print(matrix_T)
    y_coeff = tf.cast(tf.ones([2,3]),dtype=tf.float64)
    z_T = tf.convert_to_tensor(np.random.rand(2,3), dtype=tf.float64)
          
    #df.to_excel('output.xlsx', index=False)
    
    var_data['t_shape'] = t_shape
    var_data['lambda_T'] = lambda_T  #lambda transpose 
    var_data['z_T'] = z_T
    var_data['y_coeff'] = y_coeff
    
def compute_data(var_data):
    #t_shape = var_data['t_shape']    #the only input data, total 100
    #t_train = var_data['t_train']   #total 60
    
    sol_t = var_data['numerical_sol']   #(2,100)
    #sol_t_train = var_data['sol_idx']   #(2,60)
     
    x = sol_t[0]
    y = sol_t[1]
    
    #print(x2y2)
    const_one = tf.ones([300,])
    const_one = tf.cast(const_one,dtype=tf.float64)
    #array = tf.stack([x,x_sqr,y_cube,x2y2],axis=0)
    array = tf.stack([const_one,x,y],axis=0)
    #print(array)
             
    #array_var_T = tf.transpose(array)
    #print(array_var_T)
    
    var_data['phi_array'] = array      #shape:(3,100)

#---------------PINN-----------------#
def setting_model(var_data):
        
    model = Sequential()
    model.add(InputLayer(input_shape=(1,),dtype=tf.float64))
    model.add(Dense(8, activation=tf.keras.activations.gelu,kernel_initializer="glorot_uniform", dtype=tf.float64))
    model.add(Dense(16, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(32, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(16, activation=tf.keras.activations.gelu, kernel_initializer="glorot_normal", dtype=tf.float64))
    model.add(Dense(8, activation=tf.keras.activations.gelu, kernel_initializer="glorot_uniform", dtype=tf.float64))
    model.add(Dense(2, activation=None, kernel_initializer="glorot_normal",dtype=tf.float64))

    model.summary()

    var_data['model'] = model
    
    return model

def train_PINN(var_data):
    t_shape = var_data['t_shape']    #the only input data, total 100
    sol_t = var_data['numerical_sol']
    
    model = var_data['model']
    
    lambda_T = var_data['lambda_T']
    array = var_data['phi_array']
    
    losses_pinn = var_data['losses_pinn']
    
    #ddt = RHS2(t_shape,sol_t)   #numerical substitution
    #print('ddt is',ddt)
    #print('dxdt_rhs', dxdt_rhs)
    #print('dydt_rhs',dydt_rhs)
    
    dudt_value = tf.matmul(lambda_T, array)
    
    #g = tf.Variable(t_f, dtype = 'float64', trainable = False)
    g = tf.convert_to_tensor(t_shape)
    
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
                
        #f1 = dx_dt - ddt[:,0]     #from numerical
        #print('dxdt loss is', f1)
        #f2 = dy_dt - ddt[:,1]
        #print('dydt loss is', f2) #from numerical
        
        f1 = dx_dt - dudt_value[0:1,:] 
        f2 = dy_dt - dudt_value[1:2,:]
        
        allgrads = grads[:,:,0,0]
        #print(allgrads)
        #dx_dt = allgrads[:,0:1]
        #dy_dt = allgrads[:,1:2]
      
        #[dl_dw, dl_db] =tape.gradient(loss, [w, b])
    
    def custom_loss(y_true, y_pred):
        
        loss_ic = K.mean(K.square(y_true - y_pred))
               
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
    
   # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])
 
   #model compilation and fitting
    t_to_array = tf.constant(t_shape)
    y_size = tf.transpose(sol_t)  #(100,2)
    y_to_fit = tf.constant(y_size)
    
    model.fit(t_to_array, y_to_fit, batch_size,epochs)   #should be respective t on train set
   
    u_pinn = model(t_shape)  
    #print(new_u[0])
    
    ic_x = tf.reduce_mean(tf.square(sol_t.T[:,0] - u_pinn[:,0]))
    ic_y = tf.reduce_mean(tf.square(sol_t.T[:,1] - u_pinn[:,1]))
    ic_all = ic_x + ic_y
    
    # ddt_all = RHS2(t_shape,sol_t.T)
    
    # h = tf.convert_to_tensor(t_shape)
        
    # with tf.GradientTape(persistent=False) as tape:
    #     tape.watch(h)
                 
    #     #model1 = load_model('recode_spiral_model.h5')
    #     u_fit = model(h)
    #     #print(u_test)
    #     grads = tape.jacobian(u_fit, h)  
    #     grads_select = tf.einsum('bxby->bxy',grads) #only one entry, the rest are zero
    #     grads_final = grads_select[:,:]
    #         #print(grads.values)
    #     #allgrads = grads[:,:,0,0]
    #     dx_dt = grads_final[:,0]
    #     dy_dt = grads_final[:,1]
    
        # f_all1 = tf.reduce_mean(tf.square(dx_dt - ddt_all[:,0]))
        # f_all2 = tf.reduce_mean(tf.square(dy_dt - ddt_all[:,0]))
        # loss_fall = f_all1 + f_all2

        # print('Derivatives loss on testing data dxdt', f_all1)
        # print('Derivatives loss on testing data dydt',f_all2)
        # print('Data loss', ic_all)
             
        # loss_pinn = loss_fall + ic_all
        # print('loss from PINNs is:', loss_pinn)  #data loss
        # losses_pinn.append(loss_pinn)
        
    u_t = np.c_[dx_dt, dy_dt] #grads_final 
    #u_t = allgrads  
    
    f_all1 = tf.reduce_mean(tf.square(f1))
    f_all2 = tf.reduce_mean(tf.square(f2))
    loss_fall = f_all1 + f_all2
    
    print('Derivatives loss on testing data dxdt', f_all1)
    print('Derivatives loss on testing data dydt',f_all2)
    print('Data loss', ic_all)
         
    loss_pinn = loss_fall + ic_all
    print('loss from PINNs is:', loss_pinn)  #data loss
    losses_pinn.append(loss_pinn)
    
    var_data['u_t'] = u_t   
    var_data['u_pinn'] = u_pinn
    var_data['loss_pinn'] = loss_pinn
    var_data['losses_pinn'] = losses_pinn
    
    return u_t #,model
      
def sparse_update(var_data):
   
    lambda_T = var_data['lambda_T']
    u_t = var_data['u_t']
    
    array = var_data['phi_array']
    z_T = var_data['z_T']
    y_coeff = var_data['y_coeff']
    
    y_lam = var_data['y_lam']
    y_lam_changes = var_data['y_lam_changes']
    
    losses_sparse = var_data['losses_sparse']
    
    #_______to update lambda_T__
    #(ut_phi-y_coeff)    size:2x3
    identity_matrix = tf.ones([3, 3],dtype=tf.float64)
    rhoI = tf.multiply(rho,identity_matrix)
    #rhoI1 = tf.convert_to_tensor(rhoI)
    #rhoI2 = tf.cast(rhoI1,dtype=tf.float64)
    rhoz = tf.multiply(rho,z_T)
    #print(u_t)
    
    u_t_matrix = u_t
    #u_t_matrix = u_t[:,:,0]
    #u_t_matrix = u_t[:,:]
    uarr = tf.matmul(tf.transpose(u_t_matrix), tf.transpose(array))
    subtra = tf.subtract(uarr, y_coeff)
    u_pT_t_rz = tf.add(subtra, rhoz)
    #print(u_pT_t_rz)
    
    phi_phiT = tf.matmul(array, tf.transpose(array))
    ppT_rI = tf.add(phi_phiT,rhoI)
    ppTrI_inv = tf.linalg.pinv(ppT_rI)
    
    lambda_T = tf.matmul(u_pT_t_rz, ppTrI_inv)
    #print(lambda_T)
    
    #lam_pphiT = tf.matmul(lambda_T, phi_phiT)
    #uarray = tf.cast(uarr, dtype=tf.float64)
    #subtra = tf.subtract(lam_pphiT, uarray)
       
    #____________to update z for frac2____
    frac1 = np.array([lamda/rho])
    frac2 = (1/rho * y_coeff) + lambda_T
    S = pywt.threshold(frac2, frac1 , 'soft')  #frac2:our signal; frac1: value to threshold
    
    z_k = S
    
    #_________to update yij_______
    y_coeff = y_coeff + rho*(lambda_T - z_k)
    
    print('lambda_matrix k+1 transpose is', lambda_T)
    print('z_k+1 is',z_k)
    print('yij_k+1 is', y_coeff)
    
    lambda_phi = tf.matmul(lambda_T, array)
    loss_sparse = tf.reduce_mean(tf.square(tf.transpose(u_t) - lambda_phi))
    print('loss from sparse is:', loss_sparse)  #physics loss
    losses_sparse.append(loss_sparse)
    
    lse_norm2 = 0.5 * (np.linalg.norm((lambda_phi - tf.transpose(u_t)),ord=2))
    lam_z = lamda * np.linalg.norm((lambda_T - z_k), ord=1)
    y_coeff_T = tf.transpose(y_coeff) #(3x2)
    y_lam = tf.reduce_sum((tf.matmul(y_coeff_T,lambda_T))-tf.matmul(y_coeff_T,z_k))
    rho_lam = 0.5*rho*np.linalg.norm((lambda_T - z_k), ord=2)
    
    y_lam_changes.append(y_lam) 
    
    loss_manual = lse_norm2 + lam_z + y_lam + rho_lam 
    print('Manual Calculated Loss', loss_manual)
    
    var_data['z_T'] = z_k
    var_data['y_coeff'] = y_coeff
    var_data['lambda_T'] = lambda_T
    var_data['lambda_phi'] = lambda_phi
    var_data['loss_sparse'] = loss_sparse
    var_data['losses_sparse'] = losses_sparse
    var_data['y_lam_changes'] = y_lam_changes
  
    
def admm_sol(var_data):
    lambda_T = var_data['lambda_T']
    t = var_data['t']
    
    def func_check(t, x):
        dxdt = lambda_T[0,0] + lambda_T[0,1]*x[0] + lambda_T[0,2]*x[1]      
        dydt = lambda_T[1,0] + lambda_T[1,1]*x[0] + lambda_T[1,2]*x[1]
        
        #dxdt = 0.9993 - 1.1657*x[0] - 0.1711*x[1]      
        #dydt = 1.9588 - 3.1502*x[0] + 0.7653*x[1]
        
        return np.array([dxdt, dydt])
    
    sol_lambda = scint.solve_ivp(fun=func_check, t_span=(t0,t1),y0=y0, method = 'RK45',t_eval=t)
    #np.linspace(0,10,99)
    var_data['sol_lambda'] = sol_lambda
    
    return sol_lambda

#dxdt = 1 + 0.2*x[0] - 0.3*x[1]      
#dydt = 2 - 0.4*x[0] + 0.5*x[1]


def visualize(var_data):
    sol_t = var_data['numerical_sol']
    u_pinn = var_data['u_pinn']
    u_t = var_data['u_t']
    lambda_phi = var_data['lambda_phi']
    losses_pinn = var_data['losses_pinn']
    losses_sparse = var_data['losses_sparse']
    
    sol_lambda = var_data['sol_lambda']
    
    #differences between u_train (observed solution) vs u_pinn (predicted solution) leads to loss_pinn
    fig,ax1 = plt.subplots(1,1)
    fig.suptitle("RK45 vs PINNs", fontsize=12)
    ax1.plot(sol_t[0,:], sol_t[1,:], label='RK45')
    ax1.legend(loc='upper left')
    ax1.plot(u_pinn[:,0],u_pinn[:,1],'x', label='PINNs')  
    ax1.set_xlabel('x(t)')#, fontdict=font)
    ax1.set_ylabel('y(t)')#, fontdict=font)
    ax1.legend(loc='upper left')
    fig.savefig('RK45 vs PINNs.png', dpi = 300)
    
    #differences between u_train vs lambda_T (PINNs-Sparse) leads to loss_sparse
    fig,ax2 = plt.subplots(1,1)
    fig.suptitle("RK45 vs IPINNs", fontsize=12)
    ax2.plot(sol_t[0,:], sol_t[1,:], label = 'RK45')    #(100,2)
    ax2.legend(loc='upper left')
    ax2.plot(sol_lambda.y[0,:],sol_lambda.y[1,:],'x', label='IPINNs') 
    ax2.set_xlabel('x(t)')#, fontdict=font)
    ax2.set_ylabel('y(t)')#, fontdict=font)
    ax2.legend(loc='upper left')
    fig.savefig('RK45 vs PINNs-Sparse.png', dpi = 300)
    
    #loss_pinn is decreasing?
    fig,ax3 = plt.subplots(1,1)
    #ax3.plot(losses_pinn, label = 'losses of PINNs')
    ax3.semilogy(losses_pinn, label = 'log_losses of PINNs')  
    ax3.set_xlabel('number of iteration')#, fontdict=font)
    ax3.set_ylabel('loss value')#, fontdict=font)
    ax3.legend(loc='upper right')
    fig.savefig('PINNs log_loss.png')
    
        #ax.semilogx(t, np.exp(-t / 5.0))
    #loss_sparse is decreasing?
    fig,ax4 = plt.subplots(1,1)
    ax4.plot(losses_sparse, label = 'losses of Sparse')
    ax4.set_xlabel('number of iteration')#, fontdict=font)
    ax4.set_ylabel('loss value')#, fontdict=font)
    ax4.legend(loc='upper right')   
    fig.savefig('Sparse_loss.png')
    #plot trajectory of coupled first order ODE (especially on other time t)
    

def save_output(var_data, fname="trial_sparse_simple.pkl"):
    with open(fname, 'bw') as f:
        pickle.dump(var_data,f)


def hand_compute(var_data):
    lambda_T = var_data['lambda_T']
    
    mul2_lambda_T = tf.multiply(2,lambda_T)
    
    zeromatrix = np.zeros([7, 6])
    
    a11 = a21 = zeromatrix
    a11=np.c_[lambda_T[0],a11] #add first column
   # a11 = np.vstack([lambda_T[0], a11])
    a11[0,:]=lambda_T[0]    #replace values of first row
    a11[0,0]=mul2_lambda_T[0,0]      #replace value at position [0,0]
    
    a21=np.c_[lambda_T[1],a21] #add first column
    a21[0,:]=lambda_T[1]    #replace values of first row
    a21[0,0]=mul2_lambda_T[0,0]      #replace value at position [0,0]
    
    
    zeromat = np.zeros([6, 7])
    a12 = a13 = a14 = a15 = a16 = a17 = zeromat
    
    a12=np.insert(a12, 1, lambda_T[0], axis = 0)
    a12[:,1]=lambda_T[0]
    a12[1,1]=mul2_lambda_T[0,1]
    a12=a12
    
def to_excel(var_data):
    t_f = var_data['t_f']
    u_pinn = var_data['u_pinn']
    u_t = var_data['u_t']
    lambda_T = var_data['lambda_T']
    lambda_phi = var_data['lambda_phi']

    #try_on = {'key': t_f}
    t_f = str(t_f) 
    u_t = str(u_t)
    u_pinn = str(u_pinn)
    
    to_compile = {t_f, u_t, u_pinn}#, lambda_T, lambda_phi}
    df1 = pd.DataFrame(to_compile)
    df1.to_excel('for_computation.xlsx', index = False)
    
    #df2 = pd.DataFrame(try_on)
    #df2.to_excel('try_on.xlsx', index = False)

    
def main():
    
    var_data = make_data() 
    prepare_data(var_data)
    compute_data(var_data)
    
    model=setting_model(var_data)
    
    #hand_compute(var_data)
    
    #train_PINN(var_data)
    #sparse_update(var_data)
    
    #put a loop PINN update, sparse update
    for i in range (itera):
       #model=setting_model(var_data)
        train_PINN(var_data)
        sparse_update(var_data)
        print(i)
    
    admm_sol(var_data)
    
    visualize(var_data)
    
    
    #get_numerical_sol(data)
    #prepare_input_data(data, N_f=N_f)
    #neighbour_data(data)
        
    #tmp, tmp, model = setting_model(data)
    #loss_ic(data)
    #loss_ODE(data)
        
    save_output(var_data, fname="ipinns_sparse_simple.pkl")
    
    elapsed = time.time() - start_time 
    print('Training time: %.2f' % (elapsed))
    
    return var_data, model

#var_data = main()
var_data, model = main()
#if __name__ == "__main__":
#    main(layers, fname='recode_spiral.pkl')

#data,model = main()
    

# with tf.GradientTape(persistent=False) as tape:
#     tape.watch(a)
#     u_trial = var_data.model(a)
#     grads_trial = tape.jacobian(u_trial,a)
#     print(grads_trial)
#     grads_choose = ('bxby->bxy',grads_trial)
#     print(grads_choose)
#     grads_sum = tf.einsum('bxby->bxy',grads_trial)
#     print(grads_sum)
#     dxt = grads_sum[:,0]
#     dyt = grads_sum[:,1]
#     grads_combine = np.c_[dxt,dyt]
#     print(grads_combine)
