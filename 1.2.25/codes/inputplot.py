import numpy as np
import matplotlib.pyplot as plt

Fs = 2000
T = 2
t = np.linspace(-T/2, T/2, Fs*T)
F = 50
x = np.sinc(2 * F * t)

def scale_signal(x, t, scale):
    t_scaled = t * scale
    return np.interp(t_scaled, t, x, left=0, right=0)

x_05 = scale_signal(x, t, 0.5)
x_1  = x
x_2  = scale_signal(x, t, 2)
y = x_05 + x_1 - x_2

plt.figure()
plt.plot(t, x)
plt.title("Input Signal x(t)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(-0.5, 0.5)
plt.grid()
plt.show()

plt.figure()
plt.plot(t, x_05, label="x(0.5t)")
plt.plot(t, x_1, label="x(t)")
plt.plot(t, x_2, label="x(2t)")
plt.title("Scaled Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.xlim(-0.5, 0.5)
plt.grid()
plt.show()

plt.figure()
plt.plot(t, y)
plt.title("Output Signal y(t) = x(0.5t) + x(t) - x(2t)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(-0.5, 0.5)
plt.grid()
plt.show()
