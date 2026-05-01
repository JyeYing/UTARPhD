# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 13:40:35 2025

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -------------------------
# Define the coupled ODE
# -------------------------
def coupled_ode(t, z):
    x, y = z
    dxdt = 1 + 0.2 * (x**2)
    dydt = 2 - 0.4 * x * y - 0.3 * y
    return [dxdt, dydt]

# -------------------------
# Simulation setup
# -------------------------
t_span = (0, 1)
t_eval = np.linspace(t_span[0], t_span[1], 100)  # fixed 200 points

# -------------------------
# Solve the true ODE
# -------------------------
sol = solve_ivp(coupled_ode, t_span, [1, 0.5], dense_output=True)
y_true = sol.sol(t_eval)   # interpolate to t_eval grid
x, y = y_true
t = t_eval

print("True system shapes:", x.shape, y.shape, t.shape)

# -------------------------
# Estimate derivatives for regression
# -------------------------
dxdt = np.gradient(x, t)
dydt = np.gradient(y, t)

# Features: [1, x, y, x*y]
X_features = np.vstack([x, y, x * y]).T

# Fit regression for dx/dt and dy/dt
reg_x = LinearRegression().fit(X_features, dxdt)
reg_y = LinearRegression().fit(X_features, dydt)

print("Estimated coefficients (dx):", reg_x.coef_, "Intercept:", reg_x.intercept_)
print("Estimated coefficients (dy):", reg_y.coef_, "Intercept:", reg_y.intercept_)

x_pred = reg_x.predict(X_features)
y_pred = reg_y.predict(X_features)

mse1 = mean_squared_error(dxdt, x_pred)
mse2 = mean_squared_error(dydt, y_pred)
mse = mse1 + mse2
print("MSE:", mse)

# -------------------------
# Define ODE using regression model
# -------------------------
def mlr_ode(t, z):
    x, y = z
    features = np.array([x, y, x*y]).reshape(1, -1)
    dxdt = reg_x.predict(features)[0]
    dydt = reg_y.predict(features)[0]
    return [dxdt, dydt]

# -------------------------
# Solve the regression-based ODE
# -------------------------
mlr_sol = solve_ivp(mlr_ode, t_span, [1, 0.5], t_eval=t_eval)

print("MLR system shapes:", mlr_sol.t.shape, mlr_sol.y.shape)

# -------------------------
# Plot comparison
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, x, 'b-', label="True x(t)")
plt.plot(t, y, 'r-', label="True y(t)")
plt.plot(mlr_sol.t, mlr_sol.y[0], 'b--', label="MLR x(t)")
plt.plot(mlr_sol.t, mlr_sol.y[1], 'r--', label="MLR y(t)")
plt.xlabel("Time t")
plt.ylabel("States")
plt.legend()
plt.title("True vs MLR-based ODE solution")
plt.grid(True)
plt.show()

fig,ax2 = plt.subplots(1,1)
ax2.plot(x, y,'x', label="Data(RK45)")
ax2.plot(mlr_sol.y[0, :], mlr_sol.y[1, :], label="Data(MpULFR_0)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
#plt.title("RK45 vs MpULFR_0")
plt.legend()
#plt.axis('equal')
plt.savefig("simple_mlr.png", dpi=300, bbox_inches="tight")
plt.show()