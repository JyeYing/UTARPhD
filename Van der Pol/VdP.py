import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1. Define the stiff Van der Pol system
# mu=1000 makes it very stiff.
def vdp_stiff(t, y, mu=1000):
    dydt = [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]
    return dydt

# Initial conditions and time span
y0 = [2.0, 0.0]
t_span = (0, 3000) # Long time span to see behavior
t_eval = np.linspace(0, 3000, 10000)

# 2. Try to solve with the default (NON-STIFF) method: 'RK45'
print("Attempting with RK45 (default, non-stiff)...")
sol_rk45 = solve_ivp(vdp_stiff, t_span, y0, method='RK45', 
                     t_eval=t_eval, rtol=1e-6, atol=1e-9)

# 3. Solve with a STIFF method: 'Radau'
print("Attempting with Radau (stiff)...")
sol_radau = solve_ivp(vdp_stiff, t_span, y0, method='Radau', 
                      t_eval=t_eval, rtol=1e-6, atol=1e-9)

# 4. Visualize the failure
plt.figure(figsize=(10, 5))
plt.plot(sol_radau.t, sol_radau.y[0], label='Stiff (Radau)', color='blue', alpha=0.7)
plt.plot(sol_rk45.t, sol_rk45.y[0], label='Non-Stiff (RK45)', color='red', linestyle='--', alpha=0.7)
plt.title(f'Van der Pol ($\mu=1000$) - Stiff vs Non-Stiff')
plt.xlabel('Time t')
plt.ylabel('y[0]')
plt.legend()
plt.grid(True)
plt.show()

print(f"RK45 Success: {sol_rk45.success}, Steps: {len(sol_rk45.t)}")
print(f"Radau Success: {sol_radau.success}, Steps: {len(sol_radau.t)}")
