# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 19:41:20 2025

EM for multiple linear regression
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize
import scipy.stats as st

import matplotlib.pyplot as plt
import scipy.integrate as scint
from sklearn.metrics import mean_squared_error
import pickle

from box import Box

t0, t1, gap= 0, 2*np.pi, 200  #step size 0.01
x0=[0.0, 1.0]

def em_algorithm_for_multiple_linear_regression(X, y, num_iter=100, tol=1e-6):
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
    
        mse = ((y - y_pred) ** 2).mean()
        print("MSE:", mse)

    return beta_est

# Example usage----------------------------
#X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])  # Including bias term (intercept)
#y = np.array([2, 4, 5, 4, 5])

#print(X)
#print(y)

#beta = em_algorithm_for_multiple_linear_regression(X, y)
#print(f"Estimated Coefficients: {beta}")
#-------------------------------

# data = pd.read_csv("ode_dataRHS_complex_roots.csv")
# #train_data = tf.convert_to_tensor(data.values)
# train_data = data.values

def make_data():    
    with open ('harmonic.pkl','rb') as f:
        data = pickle.load(f)
    
        
    # global data
    libdata ={
        
           't' : data.t,
           #'output' : data.numerical_sol.y,
           'x' : data.numerical_sol.y[0,:],
           'y' : data.numerical_sol.y[1,:],
           
           'x_train' : data.u_pred[:,0],
           'y_train' : data.u_pred[:,1],
           
           'x_sample' : data.numerical_neighbour.y[0,:],
           'y_sample' : data.numerical_neighbour.y[1,:],
           
           't_f' : data.t_f,
           #'dxdt_given' : data.grads,
           #'dydt_given' : s_data.u_t[:,1],
           'ddt':data.ddt,
           'dxdt':data.ddt[0,:],
           'dydt':data.ddt[1,:],
           
           
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
           'output_noise':[],
           
           #'u_fit': [],
           'u_pinn':[],
           'u_pinn_noise':[],
           
           'beta1':[],
           'beta2':[],
           'sol_func':[],
           'sol_emmlr':[],
           'beta3' :[],
           'beta4':[],
           'sigma1':[],
           'sigma2':[],
           'array':[],
           
           }
    return Box(libdata)

def EM_mlr(libdata):
    t = libdata['t']
    #t_vec = np.array(t).reshape(-1,1)
    x = libdata['x_sample']
    y = libdata['y_sample']
    t_f = libdata['t_f']
    #dydt = libdata['dydt_given']
      
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)
        
    x2 = tf.square(x)
    y2 = tf.square(y)   
    xy = tf.multiply(x,y)
    
    const_one = tf.ones([170,])
    
    iv1 = np.c_[const_one,x, y]
    iv2 = np.c_[const_one, x, y ]
    
    
    dxdt = libdata['dxdt'] 
    dydt = libdata['dydt'] 
    
    
    #print(output)
    #print(dxdt)
    
    beta1 = em_algorithm_for_multiple_linear_regression(iv1, dxdt)
    print(f"Estimated em_dxdt Coefficients: {beta1}")
    
    beta2 = em_algorithm_for_multiple_linear_regression(iv2, dydt)
    print(f"Estimated em_dydt Coefficients: {beta2}")   
    

    
    libdata['beta1'] = beta1
    libdata['beta2'] = beta2
    
    return beta1, beta2

def mle_mlr(X, y, alpha=0.05):
    """
    Fit beta using MLE for Multiple Linear Regression.
    Args:
        X: 2D array (n_samples x n_features), predictors including intercept term if desired
        y: 1D array (n_samples,), response
    Returns:
        beta_hat: estimated coefficients
        sigma2_hat: estimated variance of errors
    """
    n, p = X.shape

    # Negative log-likelihood function
    def neg_log_likelihood(beta_sigma2):
        beta = beta_sigma2[:p]
        sigma2 = beta_sigma2[p]
        if sigma2 <= 0:
            return np.inf
        residuals = y - X @ beta
        ll = -0.5 * n * np.log(2 * np.pi * sigma2) - (residuals @ residuals) / (2 * sigma2)
        return -ll  # negative for minimization

    # Initial guesses: OLS estimates and variance
    beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ beta_init
    sigma2_init = np.var(residuals, ddof=p)
    init_params = np.concatenate([beta_init, [sigma2_init]])

    result = minimize(neg_log_likelihood, init_params, method='L-BFGS-B')
    beta_hat = result.x[:p]
    sigma2_hat = result.x[p]
    
    # Compute standard errors and confidence intervals
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2_hat * XtX_inv))
    t_val = st.t.ppf(1 - alpha/2, df=n - p)
    ci = [(beta_hat[i] - t_val*se[i], beta_hat[i] + t_val*se[i]) for i in range(p)]

    return beta_hat, sigma2_hat, ci

def mlr(libdata):
    t = libdata['t']
    #t_vec = np.array(t).reshape(-1,1)
    x = libdata['x_sample']
    y = libdata['y_sample']
    dxdt = libdata['dxdt']
    dydt = libdata['dydt']
      
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)
        
    x2 = tf.square(x)
    y2 = tf.square(y)   
    xy = tf.multiply(x,y)
    
    n=170
    
    const_one = tf.ones([n,])
    
    iv = np.c_[const_one, x, y, x2, y2, xy]
    
    #df = pd.read_csv('your_dataset.csv')
    #y = df['y'].values
    #X = df.drop(columns=['y']).values
    #X = np.c_[np.ones(len(X)), X] # add intercept


    beta_hat1, sigma2_hat1, ci1 = mle_mlr(iv, dxdt)
    print("Estimated coefficients (dxdt):", beta_hat1)
    print("Estimated variance1 (sigma^2):", sigma2_hat1)
    
    beta_hat2, sigma2_hat2, ci2 = mle_mlr(iv, dydt)
    print("Estimated coefficients (dydt):", beta_hat2)
    print("Estimated variance2 (sigma^2):", sigma2_hat2)

    for i, interval in enumerate(ci1):
        print(f"Beta_{i} 95% CI: ({interval[0]:.4f}, {interval[1]:.4f})")
    
    for i, interval in enumerate(ci2):
        print(f"Beta_{i} 95% CI: ({interval[0]:.4f}, {interval[1]:.4f})")

    libdata['beta3'] = beta_hat1
    libdata['beta4'] = beta_hat2
    libdata['sigma1'] = sigma2_hat1
    libdata['sigma2'] = sigma2_hat2
    libdata['array'] = iv
    
    return beta_hat1, beta_hat2, sigma2_hat1, sigma2_hat2

def get_em_mlr_sol(libdata):
    
    beta1 = libdata['beta1']
    beta2 = libdata['beta2']
    iv = libdata['array']
   
    #print(np.shape(beta1))
    #print(beta2)
    
    w = np.c_[beta1,beta2]
    print('w is', w)
    print('w(0,0) is', w[0,0])
    print('w(0,1) is', w[0,1])
    print('w(1,0) is', w[1,0])
    print('w(1,1) is', w[1,1])
    
    def func_check(t, x):
        
        #---------------sparse----------
        y1 = w[0,0] + w[1,0]*x[0] + w[2,0]*x[1] #+ w[3,0]*(x[0]**2) + w[4,0]*(x[1]**2) + w[5,0]*(x[0]*x[1]) 
        y2 = w[0,1] + w[1,1]*x[0] + w[2,1]*x[1] #+ w[3,1]*(x[0]**2) + w[4,1]*(x[1]**2) + w[5,1]*(x[0]*x[1])
        
        #---------------lasso----------
        #y1 = -3.0794e-5 + -4.9602e-1*x[0] + 9.9829e-1*x[1] + -8.1047e-9*(x[0]**2) + -5.1425*(x[1]**2) -7.5256e-9*(x[0]*x[1]) 
        #y2 = -10.9795 -26.5780*x[0] - 59.4701*x[1] + 532.9849*(x[0]**2) + 293.8160*(x[1]**2) + 265.4348*(x[0]*x[1])
       
        return np.array([y1])#, y2])
    
    sol_func = scint.solve_ivp(fun=func_check, t_span=(t0,t1), y0=x0, method="RK45", t_eval=np.linspace(t0,t1,gap))

    
    
    libdata['sol_func'] = sol_func


    return sol_func

def get_mlr_sol(libdata):
    
    beta3 = libdata['beta3']
    beta4 = libdata['beta4']
   
    #print(np.shape(beta1))
    #print(beta2)
    
    w = np.c_[beta3,beta4]
    
    def func_check(t, x):
        
        #---------------sparse----------
        y1 = w[0,0] + w[1,0]*x[0] + w[2,0]*x[1] #+ w[3,0]*(x[0]**2) + w[4,0]*(x[1]**2) + w[5,0]*(x[0]*x[1]) 
        y2 = w[0,1] + w[1,1]*x[0] + w[2,1]*x[1] #+ w[3,1]*(x[0]**2) + w[4,1]*(x[1]**2) + w[5,1]*(x[0]*x[1])
        
        #---------------lasso----------
        #y1 = -3.0794e-5 + -4.9602e-1*x[0] + 9.9829e-1*x[1] + -8.1047e-9*(x[0]**2) + -5.1425*(x[1]**2) -7.5256e-9*(x[0]*x[1]) 
        #y2 = -10.9795 -26.5780*x[0] - 59.4701*x[1] + 532.9849*(x[0]**2) + 293.8160*(x[1]**2) + 265.4348*(x[0]*x[1])
       
        return np.array([y1, y2])
    
    sol_em_mlr = scint.solve_ivp(fun=func_check, t_span=(t0,t1), y0=x0, method="RK45", t_eval=np.linspace(t0,t1,gap))
    
    

    
    libdata['sol_emmlr'] = sol_em_mlr

    return sol_em_mlr

def visualize(libdata):
    
    sol_func = libdata['sol_func']
    sol_em_mlr = libdata['sol_emmlr']
   
    t = libdata['t']
    x = libdata['x']
    y = libdata['y']
    #dxdt = libdata['dxdt_given']
    #dydt = libdata['dydt_given']
      
    t_f = libdata['t_f']
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)
    
    output = np.c_[x,y]
    
    #differences between u_train vs lambda_T (PINNs-Sparse) leads to loss_sparse
    fig,ax1 = plt.subplots(1,1)
    #fig.suptitle("MpULFR-M_complex", fontsize=12)
    ax1.plot(t,output[:,0],'x', label = 'Data(RK45)')    #(100,2)
    ax1.legend(loc='upper right')
    ax1.plot(t, sol_func.y[0,:], label='Data(MpULFR-0)')
    ax1.set_xlabel('Time, t')#, fontdict=font)
    ax1.set_ylabel('Solution, y(t)')#, fontdict=font)
    ax1.legend(loc='upper right')
    fig.savefig('functional trajectory comparison.png', dpi=300)

    # fig,ax2 = plt.subplots(1,1)
    # fig.suptitle("MpULFR-M_complex", fontsize=12)
    # ax2.plot(output[:,0], output[:,1],'x', label = 'data')    #(100,2)
    # ax2.legend(loc='upper left')
    # ax2.plot(sol_em_mlr.y[0,:],sol_em_mlr.y[1,:],'x', label='MpULFR_0')
    # ax2.set_xlabel('x(t)')#, fontdict=font)
    # ax2.set_ylabel('y(t)')#, fontdict=font)
    # ax2.legend(loc='upper left')
    # fig.savefig('em_mlr.png', dpi=300)

def main():
    libdata = make_data()     
    
    EM_mlr(libdata)
    get_em_mlr_sol(libdata)
    
    #mlr(libdata)
    #get_mlr_sol(libdata)
    visualize(libdata)
    
    return libdata

libdata = main()

    


