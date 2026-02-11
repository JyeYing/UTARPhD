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

from box import Box
from scipy.linalg import solve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
import seaborn as sns

σ = 10
tau = 2
N=53
fore_N = 37

N2=47

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



#data = pd.read_csv("ode_dataRHS_nonlinear.csv")
data = pd.read_csv("population.csv")
#train_data = tf.convert_to_tensor(data.values)
train_data = data.values

dosm = pd.read_csv("dosm.csv")
forecast_data = tf.convert_to_tensor(dosm.values)

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
           'beta3':[],
           'beta4':[],
           'sol_func':[],
           'sol_noise':[],
           'sol_ml':[],
           
           'fore_t' : forecast_data[:-3,0],
           'fore_f' : forecast_data[:-3,1],
           'fore_m' : forecast_data[:-3,2],
           
           }
    return Box(libdata)

# Simulated functional data

def EM_func_model(libdata):
    t = libdata['t']
    #t_vec = np.array(t).reshape(-1,1)
    x = libdata['x']
    y = libdata['y']
    dxdt = libdata['dxdt_given']
    dydt = libdata['dydt_given']
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)

    #t = np.linspace(0, 1, 10)  # Time or domain
    #y = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, size=t.shape) # Response
    
    output = np.c_[x, y]
    
    x2 = tf.square(x)
    y2 = tf.square(y)   
    xy = tf.multiply(x,y)
    const_one = tf.ones([N,])
    
    
    Phi1 = np.c_[x, x2, xy]
    Phi2 = np.c_[y, y2, xy]
    
    # Apply whitening using PCA to remove correlations between basis components
    #scaler = StandardScaler()
    #Phi_scaled = scaler.fit_transform(Phi)
    #pca_func = PCA(whiten=True)
    #Phi_whitened = pca_func.fit_transform(Phi_scaled)

    # Apply manual whitening to zero out covariance
    # Phi_mean_centered = Phi - np.mean(Phi, axis=0) #(100,6)
    # cov_phi = np.cov(Phi_mean_centered.T)  #(6,6)
    # eigvals_phi, eigvecs_phi = np.linalg.eigh(cov_phi)  #(6,) and(6,6)
    # eigvals_phi[eigvals_phi < 1e-10] = 1e-10  # Avoid division by zero or tiny eigenvalues
    # #print(eigvals_phi, eigvecs_phi)
    # inv_sqrt_eigvals_phi = np.diag(1.0 / np.sqrt(eigvals_phi))
    # Phi_whitened = Phi_mean_centered @ eigvecs_phi @ inv_sqrt_eigvals_phi @ eigvecs_phi.T

    beta1 = em_algorithm_for_functional_regression(Phi1, dxdt)
    print(f"Estimated Functional dxdt Coefficients: {beta1}")
    
    beta2 = em_algorithm_for_functional_regression(Phi2, dydt)
    print(f"Estimated Functional dydt Coefficients: {beta2}")
    
    libdata['beta1'] = beta1
    libdata['beta2'] = beta2
    libdata['output']= output
    
    return beta1, beta2

def EM_func_noise_model(libdata):
    t = libdata['t']
    #t_vec = np.array(t).reshape(-1,1)
    x = libdata['x']
    y = libdata['y']
    dxdt = libdata['dxdt_given']
    dydt = libdata['dydt_given']
    #x_vec = np.array(x).reshape(-1,1)
    #y_vec = np.array(y).reshape(-1,1)

    #t = np.linspace(0, 1, 10)  # Time or domain
    #y = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, size=t.shape) # Response
    
    x_noise = x  + np.random.normal(loc=0, scale=σ**2, size=N)
    y_noise = y  + np.random.normal(loc=0, scale=σ**2, size=N)
    output_noise = np.c_[x_noise, y_noise]
    
    x2 = tf.square(x_noise)
    y2 = tf.square(y_noise)   
    xy = tf.multiply(x_noise,y_noise)
    const_one = tf.ones([N,])
    
    dxdt_noise = dxdt + np.random.normal(loc=0, scale=tau**2, size=N)
    dydt_noise = dydt + np.random.normal(loc=0, scale=tau**2, size=N)
    
    
    Phi3 = np.c_[x_noise, x2, xy]
    Phi4 = np.c_[y_noise, y2, xy]

    beta3 = em_algorithm_for_functional_regression(Phi3, dxdt_noise)
    print(f"Estimated noise dxdt Coefficients: {beta3}")
    
    beta4 = em_algorithm_for_functional_regression(Phi4, dydt_noise)
    print(f"Estimated noise dydt Coefficients: {beta4}")
    
    libdata['beta3'] = beta3
    libdata['beta4'] = beta4
    libdata['output_noise'] = output_noise
    
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

    print('VIF for given data',vif_data)
    
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
        'dxdt':dxdt,
        'dydt':dydt
    })
    
    # Suppose df is your real dataset with columns ['X1', 'X2', 'X3', 'X4', 'Y']
    X = df[['x', 'x^2', 'y', 'y^2', 'xy']].values
    y = df['dxdt'].values
    y2 = df['dydt'].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Estimate sigma
    n, p = X.shape
    rss = np.sum((y - y_pred)**2)
    sigma_hat = np.sqrt(rss / (n *(p + 1)))  # subtract 1 for intercept

    model.fit(X, y2)
    y_pred2 = model.predict(X)

    # Estimate sigma
    n2, p2 = X.shape
    rss2 = np.sum((y2 - y_pred2)**2)
    sigma_hat2 = np.sqrt(rss / (n2 *(p2 + 1)))  # subtract 1 for intercept

    print(f"Estimated dxdt_sigma (σ): {sigma_hat:.4f}")
    print(f"Estimated dydt_sigma (σ): {sigma_hat2:.4f}")


def vif_noise_cal(libdata):
    # Example DataFrame with 4 predictors (replace with your own data)
    #df = pd.read_csv("ode_dataRHS_complex_roots.csv")
    
    x = libdata['x']
    y = libdata['y']
    
    x_noise = x  + np.random.normal(loc=0, scale=σ**2, size=N)
    y_noise = y  + np.random.normal(loc=0, scale=σ**2, size=N)
    
    x2 = tf.square(x_noise)
    y2 = tf.square(y_noise)   
    xy = tf.multiply(x_noise,y_noise)
    
    # df = pd.DataFrame({
    #     'X1': np.random.rand(100),
    #     'X2': np.random.rand(100),
    #     'X3': np.random.rand(100),
    #     'X4': np.random.rand(100)
    # })
    
    df = pd.DataFrame({
        'x': x_noise,
        'x^2': x2,
        'y': y_noise,
        'y^2': y2,
        'xy': xy
    })

    # Optional: standardize predictors (if units differ widely)
    # df = (df - df.mean()) / df.std()

    # ----- Step 1: Correlation matrix -----
    print("🔹 Correlation matrix:")
    corr_matrix = df.corr()
    print('Noise_data corr_matrix', corr_matrix)

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

    print('VIF for noise data',vif_data)

def est_noise_sigma(libdata):
    
    x = libdata['x']
    y = libdata['y']
    dxdt_given = libdata['dxdt_given']
    dydt_given = libdata['dydt_given']
        
    x_noise = x  + np.random.normal(loc=0, scale=σ**2, size=N)
    y_noise = y  + np.random.normal(loc=0, scale=σ**2, size=N)
    
    dxdt_noise = dxdt_given + np.random.normal(loc=0, scale=tau**2, size=N)
    dydt_noise = dydt_given + np.random.normal(loc=0, scale=tau**2, size=N)
    
    
    x2 = tf.square(x_noise)
    y2 = tf.square(y_noise)   
    xy = tf.multiply(x_noise,y_noise)
    
    
    df = pd.DataFrame({
        'x': x_noise,
        'x^2': x2,
        'y': y,
        'y^2': y2,
        'xy': xy,
        'dxdt':dxdt_noise,
        'dydt':dydt_noise
    })
    
    # Suppose df is your real dataset with columns ['X1', 'X2', 'X3', 'X4', 'Y']
    X = df[['x', 'x^2', 'y', 'y^2', 'xy']].values
    y1 = df['dxdt'].values
    y2 = df['dydt'].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y1)
    y_pred1 = model.predict(X)
    
    model.fit(X,y2)
    y_pred2 = model.predict(X)
    
    # Estimate sigma for dxdt
    n1, p1 = X.shape
    rss1 = np.sum((y1 - y_pred1)**2)
    sigma_hat1 = np.sqrt(rss1 / (n1 *(p1 + 1)))  # subtract 1 for intercept

    # Estimate sigma for dydt
    n2, p2 = X.shape
    rss2 = np.sum((y2 - y_pred2)**2)
    sigma_hat2 = np.sqrt(rss2 / (n2 *(p2 + 1)))  # subtract 1 for intercept

    print(f"Estimated noise dxdt_sigma (σ): {sigma_hat1:.4f}")
    print(f"Estimated noise dydt_sigma (σ): {sigma_hat2:.4f}")

def get_functional_sol(libdata):
    
    beta1 = libdata['beta1']
    beta2 = libdata['beta2']
   
    #print(np.shape(beta1))
    #print(beta2)
    
    w = np.c_[beta1,beta2]
    
    def func_check(t, x):
        
        #---------------sparse----------
        y1 = w[0,0]*x[0] + w[1,0]*(x[0]**2) + w[2,0]*(x[0]*x[1]) #+ w[5,0]*(x[0]*x[1]) #+ w[2,0]*x[1] + w[4,0]*(x[1]**2) 
        y2 = w[0,1]*x[1] + w[1,1]*(x[1]**2) + w[2,1]*(x[0]*x[1]) #+ w[5,1]*(x[0]*x[1]) #+ w[1,1]*x[0] + w[3,1]*(x[0]**2)
        
        #---------------lasso----------
        #y1 = -3.0794e-5 + -4.9602e-1*x[0] + 9.9829e-1*x[1] + -8.1047e-9*(x[0]**2) + -5.1425*(x[1]**2) -7.5256e-9*(x[0]*x[1]) 
        #y2 = -10.9795 -26.5780*x[0] - 59.4701*x[1] + 532.9849*(x[0]**2) + 293.8160*(x[1]**2) + 265.4348*(x[0]*x[1])
       
        return np.array([y1, y2])
    
    sol_func = scint.solve_ivp(fun=func_check, t_span=(1971,2023), y0=[5500,5600], method="RK45", t_eval=np.linspace(1971,2023,53))
    
    sol_func2 = scint.solve_ivp(fun=func_check, t_span=(0,N2-1), y0=[5500,5600], method="RK45", t_eval=np.linspace(0,N2-1,N2-1))
    ml_start = sol_func2.y[:, -1] 
    sol_ml = scint.solve_ivp(fun=func_check, t_span=(0,N2+10), y0=[5500,5600], method="RK45", t_eval=np.linspace(0,N2+10,N2+10))

    
    forecast_start=[16150,17880]
    t_future = np.linspace(0, fore_N, fore_N)
    sol_future = scint.solve_ivp(func_check, (t_future[0], t_future[-1]), forecast_start, method='RK45',t_eval=t_future)

    #libdata['sol_ml'] = sol_ml
    libdata['sol_future'] = sol_future
    
    libdata['sol_func'] = sol_func

    libdata['sol_ml'] = sol_ml
    
    return sol_func

def get_func_noise_sol(libdata):
    
    beta3 = libdata['beta3']
    beta4 = libdata['beta4']
   
    #print(np.shape(beta1))
    #print(beta2)
    
    w = np.c_[beta3,beta4]
    
    def func_check(t, x):
        
        #---------------sparse----------
        y1 = w[0,0]*x[0] + w[1,0]*(x[0]**2) #+ w[2,0]*x[1]+ w[4,0]*(x[1]**2) + w[5,0]*(x[0]*x[1]) 
        y2 = w[0,1]*x[0] + w[1,1]*(x[1]**2) #+ w[2,1]*x[1] + w[3,1]*(x[0]**2)+ w[5,1]*(x[0]*x[1])
        
        #---------------lasso----------
        #y1 = -3.0794e-5 + -4.9602e-1*x[0] + 9.9829e-1*x[1] + -8.1047e-9*(x[0]**2) + -5.1425*(x[1]**2) -7.5256e-9*(x[0]*x[1]) 
        #y2 = -10.9795 -26.5780*x[0] - 59.4701*x[1] + 532.9849*(x[0]**2) + 293.8160*(x[1]**2) + 265.4348*(x[0]*x[1])
       
        return np.array([y1, y2])
    
    sol_noise = scint.solve_ivp(fun=func_check, t_span=(0,10), y0=[5500,5600], method="RK45", t_eval=np.linspace(0,10,99))
    
    libdata['sol_noise'] = sol_noise

    return sol_noise

def visualize(libdata):
    
    sol_func = libdata['sol_func']
    sol_noise = libdata['sol_noise']
    sol_future = libdata['sol_future']
    sol_ml = libdata['sol_ml']
    
    output = libdata['output']
    output_noise = libdata['output_noise']
    
    fore_t = libdata['fore_t']
    fore_f = libdata['fore_f']
    fore_m = libdata['fore_m']
    
    # x_vec = libdata['x_vec']
    # y_vec = libdata['y_vec']
    # output_noise = libdata['output_noise']
    
    forecast_fm = np.c_[fore_f, fore_m]
    
    #differences between u_train vs lambda_T (PINNs-Sparse) leads to loss_sparse
    fig,ax1 = plt.subplots(1,1)
    fig.suptitle("MpULFR-M_Total Population", fontsize=12)
    ax1.plot(output[:,0], output[:,1],'x', label = 'dosm')    #(100,2)
    ax1.legend(loc='upper left')
    ax1.plot(sol_func.y[0,:],sol_func.y[1,:],'x', label='MpULFR-M')
    ax1.xlim = (0,10)
    ax1.ylim = (-5,0)
    ax1.set_xlabel('female, x(t) (thousands)')#, fontdict=font)
    ax1.set_ylabel('male, y(t) (thousands)')#, fontdict=font)
    ax1.legend(loc='upper left')
    fig.savefig('functional trajectory comparison.png', dpi=300)

    fig,ax17 = plt.subplots(1,1)
    fig.suptitle("MpULFR-M_Forecast Total Population", fontsize=12)
    ax17.plot(forecast_fm[:,0], forecast_fm[:,1],'x', label = 'dosm')    #(100,2)
    ax17.legend(loc='upper left')
    ax17.plot(sol_future.y[0,:],sol_future.y[1,:],'x', label='MpULFR-M')
    ax17.set_xlabel('female, x(t) (thousands)')#, fontdict=font)
    ax17.set_ylabel('male, y(t) (thousands)')#, fontdict=font)
    ax17.legend(loc='upper left')
    fig.savefig('forecast xy.png', dpi=300)
    
    fig,ax18 = plt.subplots(1,1)
    fig.suptitle("MpULFR-M_Forecast Female Population", fontsize=12)
    ax18.plot(fore_t, forecast_fm[:,0], 'x', label = 'dosm_x')
    ax18.plot(fore_t, sol_future.y[0], 'x', label = 'MpULFR-M_x')
    #ax18.set_title('forecast_female, x')
    ax18.legend(loc='upper left')
    ax18.set_xlabel('year, t')#, fontdict=font)
    ax18.set_ylabel('female, x(t) (thousands)')#, fontdict=font)
    ax18.legend(loc='upper left')
    fig.savefig('forecast x.png', dpi=300)
    
    fig,ax19 = plt.subplots(1,1)
    fig.suptitle("MpULFR-M_Forecast Male Population", fontsize=12)
    ax19.plot(fore_t, forecast_fm[:,1], 'x', label = 'dosm_y')
    ax19.plot(fore_t, sol_future.y[1], 'x', label = 'MpULFR-M_y')
    #ax19.set_title('forecast_male, y')
    ax19.legend(loc='upper left')
    ax19.set_xlabel('year, t')#, fontdict=font)
    ax19.set_ylabel('male, y(t) (thousands)')#, fontdict=font)
    ax19.legend(loc='upper left')
    fig.savefig('forecast y.png', dpi=300)
    
    fig,ax20 = plt.subplots(1,1)
    ax20.plot(output[-10:,0], forecast_fm[-10:,1],'x', label = 'dosm')    #(100,2)
    ax20.legend(loc='upper right')
    ax20.plot(sol_ml.y[0,-10:],sol_ml.y[1,-10:],'x', label='forecast')
    ax20.set_xlabel('x')#, fontdict=font)
    ax20.set_ylabel('y')#, fontdict=font)
    ax20.legend(loc='upper right')
    fig.savefig('forecast given 20% xy.png', dpi=300)
    
    # fig,ax21 = plt.subplots(1,1)
    # ax21.plot(fore_t, forecast_fm[:,0], 'x', label = 'dosm_x')
    # ax21.plot(fore_t, sol_ml.y[0], 'x', label = 'model_x')
    # ax21.set_title('forecast_x')
    # ax21.legend(loc='upper left')
    # ax21.set_xlabel('t')#, fontdict=font)
    # ax21.set_ylabel('x')#, fontdict=font)
    # ax21.legend(loc='upper left')
    # fig.savefig('forecast 20% x.png', dpi=300)
    
    # fig,ax22 = plt.subplots(1,1)
    # ax22.plot(fore_t, forecast_fm[:,1], 'x', label = 'dosm_y')
    # ax22.plot(fore_t, sol_ml.y[1], 'x', label = 'model_y')
    # ax22.set_title('forecast_y')
    # ax22.legend(loc='upper left')
    # ax22.set_xlabel('t')#, fontdict=font)
    # ax22.set_ylabel('y')#, fontdict=font)
    # ax22.legend(loc='upper left')
    # fig.savefig('forecast 20% y.png', dpi=300)
    
def main():
    libdata = make_data()     
    
    EM_func_model(libdata)
    EM_func_noise_model(libdata)
    
    get_functional_sol(libdata)
    get_func_noise_sol(libdata)
    
    vif_cal(libdata)
    est_sigma(libdata)
    
    vif_noise_cal(libdata)
    est_noise_sigma(libdata)
    
    visualize(libdata)
    
    return libdata

libdata = main()