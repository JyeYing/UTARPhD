# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 19:41:20 2025

EM for multiple linear regression
"""
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.integrate as scint
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
import seaborn as sns
from box import Box

σ = 1
tau = 1
N = 53

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
    eps = 1e-12
    
    # Initialize parameters
    beta_est = np.random.randn(p)
    variance = np.var(y)
    
    for _ in range(num_iter):
        # E-step: Compute expected values of latent variables (errors assumed to be Gaussian)
        y_pred = X @ beta_est
        responsibilities = np.exp(-0.5 * ((y - y_pred) ** 2) / variance)
        responsibilities /= (np.sum(responsibilities) + eps)
        
        # M-step: Update estimates using weighted least squares
        W = np.diag(responsibilities)
        X_weighted = X.T @ W @ X
        y_weighted = X.T @ W @ y
        
        #print(X_weighted.shape)
        #print(y_weighted.shape)
        
        new_beta_est = np.linalg.pinv(X_weighted)@ y_weighted
        
        # Check for convergence
        if np.linalg.norm(new_beta_est - beta_est) < tol:
            break
        
        beta_est = new_beta_est
    
    return beta_est

# Example usage----------------------------
#X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])  # Including bias term (intercept)
#y = np.array([2, 4, 5, 4, 5])

#print(X)
#print(y)

#beta = em_algorithm_for_multiple_linear_regression(X, y)
#print(f"Estimated Coefficients: {beta}")
#-------------------------------

data = pd.read_csv("population.csv")
#data = pd.read_csv("population.csv")
#train_data = tf.convert_to_tensor(data.values)
train_data = data.values

N = 53

def make_data():    
    # global data
    libdata ={
        
           't' : train_data[:,0],
           'x' : train_data[:,1],
           'y' : train_data[:,2],
           
           'dxdt_given' : train_data[:,3],
           'dydt_given' : train_data[:,4],
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
           'output_noise':[],
           
           #'u_fit': [],
           'u_pinn':[],
           'u_pinn_noise':[],
           
           'beta1':[],
           'beta2':[],
           'sol_func':[],
           'beta3':[],
           'beta4':[],
           'sol_noise':[],
           
           }
    return Box(libdata)

def EM_model(libdata):
    t = libdata['t']
    #t_vec = np.array(t).reshape(-1,1)
    x = libdata['x']
    y = libdata['y']
    dxdt = libdata['dxdt_given']
    dydt = libdata['dydt_given']
      
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)
        
    x2 = tf.square(x)
    y2 = tf.square(y)   
    xy = tf.multiply(x,y)
    output = np.c_[x,y]
    
    const_one = tf.ones([N,])
    
    iv1 = np.c_[x]#, x2, y2, xy]
    iv2 = np.c_[y]
    
    #print(output)
    #print(dxdt)
    
    beta1 = em_algorithm_for_multiple_linear_regression(iv1, dxdt)
    print(f"Estimated em_dxdt Coefficients: {beta1}")
    
    beta2 = em_algorithm_for_multiple_linear_regression(iv2, dydt)
    print(f"Estimated em_dydt Coefficients: {beta2}")
    
    libdata['beta1'] = beta1
    libdata['beta2'] = beta2
    libdata['output'] = output
    
    return beta1, beta2



def EM_noise_model(libdata):
    t = libdata['t']
    #t_vec = np.array(t).reshape(-1,1)
    x = libdata['x']
    y = libdata['y']
    dxdt = libdata['dxdt_given']
    dydt = libdata['dydt_given']
      
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)
        
 
    
    const_one = tf.ones([N,])
    dxdt_noise = dxdt + np.random.normal(loc=0, scale=tau**2, size=N)
    dydt_noise = dydt + np.random.normal(loc=0, scale=tau**2, size=N)
    
    x_noise = x  + np.random.normal(loc=0, scale=σ**2, size=N)
    y_noise = y  + np.random.normal(loc=0, scale=σ**2, size=N)
    output_noise = np.c_[x_noise, y_noise]
    
    x2 = tf.square(x)
    y2 = tf.square(y)   
    xy = tf.multiply(x,y)
    
    iv = np.c_[const_one, x_noise, y_noise]#, x2, y2, xy]
    
    #print(output)
    #print(dxdt)
    
    beta3 = em_algorithm_for_multiple_linear_regression(iv, dxdt_noise)
    print(f"Estimated noise dxdt Coefficients: {beta3}")
    
    beta4 = em_algorithm_for_multiple_linear_regression(iv, dydt_noise)
    print(f"Estimated noise dydt Coefficients: {beta4}")
    
    libdata['beta3'] = beta3
    libdata['beta4'] = beta4
    libdata['output_noise']=output_noise
    
    return beta3, beta4

def vif_cal(libdata):
    # Example DataFrame with 4 predictors (replace with your own data)
    #df = pd.read_csv("ode_dataRHS_complex_roots.csv")
    
    x = libdata['x']
    y = libdata['y']
        
    x2 = tf.square(x)
    y2 = tf.square(y)   
    xy = tf.multiply(x,y)
    
    # df = pd.DataFrame({
    #     'X1': np.random.rand(100),
    #     'X2': np.random.rand(100),
    #     'X3': np.random.rand(100),
    #     'X4': np.random.rand(100)
    # })
    
    df = pd.DataFrame({
        'x': x,
        'x^2': x2,
        'y': y,
        'y^2': y2,
        'xy': xy
    })

    # Optional: standardize predictors (if units differ widely)
    # df = (df - df.mean()) / df.std()

    # ----- Step 1: Correlation matrix -----
    print("🔹 Correlation matrix:")
    corr_matrix = df.corr()
    print(corr_matrix)

    # Visual heatmap for quick overview
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.show()

    # ----- Step 2: VIF calculation -----
    print("\n🔹 Variance Inflation Factor (VIF):")
    X = df.copy()
    X["Intercept"] = 1  # add constant for VIF calc

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns[:-1]  # exclude intercept from report
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1] - 1)]  # skip intercept

    print(vif_data)

def est_sigma(libdata):
    
    x = libdata['x']
    y = libdata['y']
        
    x2 = tf.square(x)
    y2 = tf.square(y)   
    xy = tf.multiply(x,y)
    
    dxdt = libdata['dxdt_given']
    dydt = libdata['dydt_given']
    
    df = pd.DataFrame({
        'x': x,
        'x^2': x2,
        'y': y,
        'y^2': y2,
        'xy': xy,
        'dxdt':dxdt
    })
    
    # Suppose df is your real dataset with columns ['X1', 'X2', 'X3', 'X4', 'Y']
    X = df[['x', 'x^2', 'y', 'y^2', 'xy']].values
    y = df['dxdt'].values

    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Estimate sigma
    n, p = X.shape
    rss = np.sum((y - y_pred)**2)
    sigma_hat = np.sqrt(rss / (n *(p + 1)))  # subtract 1 for intercept

    print(f"Estimated sigma (σ): {sigma_hat:.4f}")

def get_mlr_sol(libdata):
    
    beta1 = libdata['beta1']
    beta2 = libdata['beta2']
   
    #print(np.shape(beta1))
    #print(beta2)
    
    w = np.c_[beta1,beta2]
    
    def func_check(t, x):
        
        #---------------sparse----------
        y1 = w[0,0]*x[0] #+ w[1,0]*(x[0]**2) #+ w[3,0]*(x[1]) + w[4,0]*(x[1]**2) + w[5,0]*(x[0]*x[1]) 
        y2 = w[0,1]*x[1] #+ w[1,1]*(x[1]**2) #+ w[3,1]*(x[0]**2) + w[4,1]*(x[1]**2) + w[5,1]*(x[0]*x[1])
        
        #---------------lasso----------
        #y1 = -3.0794e-5 + -4.9602e-1*x[0] + 9.9829e-1*x[1] + -8.1047e-9*(x[0]**2) + -5.1425*(x[1]**2) -7.5256e-9*(x[0]*x[1]) 
        #y2 = -10.9795 -26.5780*x[0] - 59.4701*x[1] + 532.9849*(x[0]**2) + 293.8160*(x[1]**2) + 265.4348*(x[0]*x[1])
       
        return np.array([y1, y2])
    
    sol_func = scint.solve_ivp(fun=func_check, t_span=(0,10), y0=[-0.1,0.3], method="RK45", t_eval=np.linspace(0,10,99))
    
    libdata['sol_func'] = sol_func

    return sol_func

def get_mlr_noise(libdata):
    
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
    
    sol_noise = scint.solve_ivp(fun=func_check, t_span=(0,10), y0=[-0.1,0.3], method="RK45", t_eval=np.linspace(0,10,99))
    
    libdata['sol_noise'] = sol_noise

    return sol_noise

def visualize(libdata):
    
    sol_func = libdata['sol_func']
    sol_noise = libdata['sol_noise']
   
    x = libdata['x']
    y = libdata['y']
    #dxdt = libdata['dxdt_given']
    #dydt = libdata['dydt_given']
      
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)
    
    output = libdata['output']
    output_noise = libdata['output_noise']
    
    #differences between u_train vs lambda_T (PINNs-Sparse) leads to loss_sparse
    fig,ax1 = plt.subplots(1,1)
    ax1.plot(output[:,0], output[:,1],'x', label = 'Data')    #(100,2)
    ax1.legend(loc='upper left')
    ax1.plot(sol_func.y[0,:],sol_func.y[1,:],'x', label='MpULFR_0')
    ax1.set_xlabel('x')#, fontdict=font)
    ax1.set_ylabel('y')#, fontdict=font)
    ax1.legend(loc='upper left')
    fig.savefig('functional trajectory mlr.png', dpi=300)

    # fig,ax2 = plt.subplots(1,1)
    # ax2.plot(output_noise[:,0], output_noise[:,1],'x', label = 'Data_noise')    #(100,2)
    # ax2.legend(loc='upper right')
    # ax2.plot(sol_noise.y[0,:],sol_noise.y[1,:],'x', label='mlr_noise')
    # ax2.set_xlabel('x')#, fontdict=font)
    # ax2.set_ylabel('y')#, fontdict=font)
    # ax2.legend(loc='upper right')
    # fig.savefig('noise trajectory comparison.png', dpi=300)

def main():
    libdata = make_data()     
    
    EM_model(libdata)
    #EM_noise_model(libdata)
    get_mlr_sol(libdata)
    #get_mlr_noise(libdata)
    vif_cal(libdata)
    est_sigma(libdata)
    visualize(libdata)
    
    return libdata

libdata = main()

    


