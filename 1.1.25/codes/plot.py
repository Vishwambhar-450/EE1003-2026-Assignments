import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# System definition from Laplace transform
# G(s) = 10 / (s^2 + 11s + 11)
num = [10]
den = [1, 11, 11]

system = signal.TransferFunction(num, den)

# Input sine wave
f = 1                       # Hz
omega = 2 * np.pi * f       # angular frequency

dt = 0.001
t = np.arange(0, 10, dt)

u = np.sin(omega * t)

# Laplace / transfer function solution
t_tf, y_tf, _ = signal.lsim(system, U=u, T=t)

# RK4 solution
# Differential equation:
# y'' + 11y' + 11y = 10u(t)
y_rk4 = np.zeros(len(t))
ydot = np.zeros(len(t))

def f1(y, ydot, time):
    return ydot

def f2(y, ydot, time):
    inp = np.sin(omega * time)
    return 10 * inp - 11 * ydot - 11 * y

for i in range(len(t) - 1):

    k1_y = dt * f1(y_rk4[i], ydot[i], t[i])
    k1_v = dt * f2(y_rk4[i], ydot[i], t[i])

    k2_y = dt * f1(y_rk4[i] + k1_y/2,
                   ydot[i] + k1_v/2,
                   t[i] + dt/2)

    k2_v = dt * f2(y_rk4[i] + k1_y/2,
                   ydot[i] + k1_v/2,
                   t[i] + dt/2)

    k3_y = dt * f1(y_rk4[i] + k2_y/2,
                   ydot[i] + k2_v/2,
                   t[i] + dt/2)

    k3_v = dt * f2(y_rk4[i] + k2_y/2,
                   ydot[i] + k2_v/2,
                   t[i] + dt/2)

    k4_y = dt * f1(y_rk4[i] + k3_y,
                   ydot[i] + k3_v,
                   t[i] + dt)

    k4_v = dt * f2(y_rk4[i] + k3_y,
                   ydot[i] + k3_v,
                   t[i] + dt)

    y_rk4[i+1] = y_rk4[i] + (k1_y + 2*k2_y + 2*k3_y + k4_y)/6
    ydot[i+1] = ydot[i] + (k1_v + 2*k2_v + 2*k3_v + k4_v)/6

# Laplace output equation
jw = 1j * omega
Gjw = 10 / (jw**2 + 11*jw + 11)

A_L = abs(Gjw)
phi_L = np.angle(Gjw)

print("Laplace Transform Output Equation:")
print(f"y_L(t) = {A_L:.4f} sin({omega:.4f} t {phi_L:+.4f})")

# RK4 approximate equation
y_ss = y_rk4[-2000:]
t_ss = t[-2000:]

A_R = (np.max(y_ss) - np.min(y_ss)) / 2

# phase estimation using peak location
peak_index = np.argmax(y_ss)
t_peak = t_ss[peak_index]

phi_R = np.pi/2 - omega * t_peak

print("\nRK4 Approximate Output Equation:")
print(f"y_RK4(t) = {A_R:.4f} sin({omega:.4f} t {phi_R:+.4f})")

# Plot comparison
plt.figure(figsize=(10,5))
plt.plot(t_tf, y_tf, label='Laplace / Transfer Function')
plt.plot(t, y_rk4, '--', label='RK4 Solution')
plt.plot(t, u, alpha=0.4, label='Input')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Laplace vs RK4 Verification')
plt.grid(True)
plt.legend()
plt.show()
