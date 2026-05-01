# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 10:42:55 2025

EM for functional
"""

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.integrate as scint
from scipy.optimize import minimize

import pickle

from box import Box
from scipy.linalg import solve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

t0, t1, gap= 0, 0.01, 100  #step size 0.01
x0=[1000]

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
    
    # Zero out covariance between Phi and y
    #Phi_centered = Phi - np.mean(Phi, axis=0)
    #y_centered = y - np.mean(y)
    #Phiy_cov = Phi_centered.T @ y_centered / (n - 1)
    #y = y - Phi_centered @ (Phiy_cov / np.maximum(np.linalg.norm(Phiy_cov), 1e-10))

    # Set Phi variance to zero via whitening
    #cov_Phi = np.cov(Phi_centered.T)
    #eigvals, eigvecs = np.linalg.eigh(cov_Phi)
    #eigvals[eigvals < 1e-10] = 1e-10
    #inv_sqrt_eigvals = np.diag(1.0 / np.sqrt(eigvals))
    #Phi = Phi_centered @ eigvecs @ inv_sqrt_eigvals @ eigvecs.T

    
    # Initialize parameters
    beta_est = np.random.randn(p)
    variance = np.var(y)
    
    for _ in range(num_iter):
        # E-step: Compute expected values of latent variables (errors assumed to be Gaussian)
        y_pred = Phi @ beta_est
        responsibilities = np.exp(-0.5 * ((y - y_pred) ** 2) / variance)
        responsibilities /= np.sum(responsibilities)
        
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

# Example usage for functional regression using polynomial basis function

# def generate_polynomial_basis(X, degree=3):
#     """ Generate polynomial basis functions. """
#     return np.column_stack([X**d for d in range(degree + 1)])

# # Simulated functional data
# t = np.linspace(0, 1, 10)  # Time or domain
# y = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, size=t.shape)  # Response
# Phi = generate_polynomial_basis(t, degree=3)  # Using polynomial basis 

# beta = em_algorithm_for_functional_regression(Phi, y)
# print(f"Estimated Functional Coefficients: {beta}")



#data = pd.read_csv("ode_dataRHS_real_roots.csv")
#train_data = tf.convert_to_tensor(data.values)
#train_data = data.values

def make_data():    
    with open ('stiff_linear.pkl','rb') as f:
        data = pickle.load(f)
    
        
    # global data
    libdata ={
        
           't' : data.t,
           #'output' : data.numerical_sol.y,
           'x' : data.numerical_sol.y[0,:],
           #'y' : data.numerical_sol.y[1,:],
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
           
           }
    return Box(libdata)

def vif_cal(libdata):
    # Example DataFrame with 4 predictors (replace with your own data)
    #df = pd.read_csv("ode_dataRHS_complex_roots.csv")
    
    x = libdata['x']
    #y = libdata['y']
        
    x2 = tf.square(x)
    #y2 = tf.square(y)   
    #xy = tf.multiply(x,y)
    
    # df = pd.DataFrame({
    #     'X1': np.random.rand(100),
    #     'X2': np.random.rand(100),
    #     'X3': np.random.rand(100),
    #     'X4': np.random.rand(100)
    # })
    
    df = pd.DataFrame({
        'x': x,
        'x^2': x2,
        #'y': y,
        #'y^2': y2,
        #'xy': xy
    })

    # Optional: standardize predictors (if units differ widely)
    # df = (df - df.mean()) / df.std()

    # ----- Step 1: Correlation matrix -----
    print("🔹 Correlation matrix:")
    corr_matrix = df.corr()
    print(corr_matrix)

    # Visual heatmap for quick overview
    #sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
   # plt.title("Correlation Matrix")
    #plt.show()

    # ----- Step 2: VIF calculation -----
    print("\n🔹 Variance Inflation Factor (VIF):")
    X = df.copy()
    X["Intercept"] = 1  # add constant for VIF calc

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns[:-1]  # exclude intercept from report
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1] - 1)]  # skip intercept

    print('VIF for given data',vif_data)

def est_sigma(libdata):
    
    x = libdata['x']
   # y = libdata['y']
        
    x2 = tf.square(x)
    #y2 = tf.square(y)   
   # xy = tf.multiply(x,y)
    
    dxdt = -10000*x
   # dydt = libdata['dydt_given']
    
    df = pd.DataFrame({
        'x': x,
        'x^2': x2,
        #'y': y,
        #'y^2': y2,
        #'xy': xy,
        'dxdt':dxdt,
        #'dydt':dydt
    })
    
    # Suppose df is your real dataset with columns ['X1', 'X2', 'X3', 'X4', 'Y']
    #X = df[['x', 'x^2', 'y', 'y^2', 'xy']].values
    X = df[['x', 'x^2']].values
    y = df['dxdt'].values
    #y2 = df['dydt'].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Estimate sigma
    n, p = X.shape
    rss = np.sum((y - y_pred)**2)
    sigma_hat = np.sqrt(rss / (n *(p + 1)))  # subtract 1 for intercept

   # model.fit(X, y2)
   # y_pred2 = model.predict(X)

    # Estimate sigma
 #   n2, p2 = X.shape
 #   rss2 = np.sum((y2 - y_pred2)**2)
 #   sigma_hat2 = np.sqrt(rss / (n2 *(p2 + 1)))  # subtract 1 for intercept

    print(f"Estimated dxdt_sigma (σ): {sigma_hat:.4f}")
  #  print(f"Estimated dydt_sigma (σ): {sigma_hat2:.4f}")

# Simulated functional data
def EM_func_model(libdata):
    t = libdata['t']
    #t_vec = np.array(t).reshape(-1,1)
    x = libdata['x']
  #  y = libdata['y']
    #dxdt = libdata['dxdt_given']
  #  dydt = libdata['dydt_given']
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)
        
    x2 = tf.square(x)
  #  y2 = tf.square(y)   
  #  xy = tf.multiply(x,y)
    const_one = tf.ones([100,])

    #t = np.linspace(0, 1, 10)  # Time or domain
    y = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, size=t.shape)  # Response
    #Phi = np.c_[const_one, x, y, x2, y2, xy]
    Phi = np.c_[const_one, x, x2]

    # Apply whitening using PCA to remove correlations between basis components
    #scaler = StandardScaler()
    #Phi_scaled = scaler.fit_transform(Phi)
    #pca_func = PCA(whiten=True)
    #Phi_whitened = pca_func.fit_transform(Phi_scaled)

    # Apply manual whitening to zero out covariance
    Phi_mean_centered = Phi - np.mean(Phi, axis=0) #(100,6)
    cov_phi = np.cov(Phi_mean_centered.T)  #(6,6)
    eigvals_phi, eigvecs_phi = np.linalg.eigh(cov_phi)  #(6,) and(6,6)
    eigvals_phi[eigvals_phi < 1e-10] = 1e-10  # Avoid division by zero or tiny eigenvalues
    print(eigvals_phi, eigvecs_phi)
    inv_sqrt_eigvals_phi = np.diag(1.0 / np.sqrt(eigvals_phi))
    Phi_whitened = Phi_mean_centered @ eigvecs_phi @ inv_sqrt_eigvals_phi @ eigvecs_phi.T

    dxdt = -10000*x
    
    beta1 = em_algorithm_for_functional_regression(Phi_whitened, dxdt)
    print(f"Estimated Functional dxdt Coefficients: {beta1}")
    
   # beta2 = em_algorithm_for_functional_regression(Phi_whitened, dydt)
  #  print(f"Estimated Functional dydt Coefficients: {beta1}")
    
    libdata['beta1'] = beta1
   # libdata['beta2'] = beta2
    
    return beta1#, beta2


    
def get_functional_sol(libdata):
    
    beta1 = libdata['beta1']
  #  beta2 = libdata['beta2']
   
    #print(np.shape(beta1))
    #print(beta2)
    
    w = np.c_[beta1]#,beta2]
    
    def func_check(t, x):
        
        #---------------sparse----------
        y1 = w[0,0] + w[1,0]*x[0] + w[2,0]*(x[0]**2) #x[1] + w[3,0]*(x[0]**2) + w[4,0]*(x[1]**2) + w[5,0]*(x[0]*x[1]) 
       # y2 = w[0,1] + w[1,1]*x[0] + w[2,1]*x[1] + w[3,1]*(x[0]**2) + w[4,1]*(x[1]**2) + w[5,1]*(x[0]*x[1])
        
        #---------------lasso----------
        #y1 = -3.0794e-5 + -4.9602e-1*x[0] + 9.9829e-1*x[1] + -8.1047e-9*(x[0]**2) + -5.1425*(x[1]**2) -7.5256e-9*(x[0]*x[1]) 
        #y2 = -10.9795 -26.5780*x[0] - 59.4701*x[1] + 532.9849*(x[0]**2) + 293.8160*(x[1]**2) + 265.4348*(x[0]*x[1])
       
        return np.array([y1])#, y2])
    
    sol_func = scint.solve_ivp(fun=func_check, t_span=(t0,t1), y0=x0, method="RK45", t_eval=np.linspace(t0,t1,gap))
    
    libdata['sol_func'] = sol_func

    return sol_func

def visualize(libdata):
    
    sol_func = libdata['sol_func']
   
    x = libdata['x']
    #y = libdata['y']
    #dxdt = libdata['dxdt_given']
    #dydt = libdata['dydt_given']
      
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)
    
    output = np.c_[x]#,y]
    
    #differences between u_train vs lambda_T (PINNs-Sparse) leads to loss_sparse
    fig,ax1 = plt.subplots(1,1)
    ax1.plot(output[:,0],'x', label = 'Data(RK45)')    #(100,2)
    ax1.legend(loc='upper right')
    ax1.plot(sol_func.y[0,:], label='Data(MpULFR-M)')
    ax1.xlim = (0,10)
    ax1.ylim = (-5,0)
    ax1.set_xlabel('x')#, fontdict=font)
    ax1.set_ylabel('y')#, fontdict=font)
    ax1.legend(loc='upper right')
    fig.savefig('functional trajectory frm comparison.png', dpi=300)

def main():
    libdata = make_data()     

    vif_cal(libdata)
    est_sigma(libdata)
    
    EM_func_model(libdata)
    get_functional_sol(libdata)
    visualize(libdata)
    
    return libdata

main()