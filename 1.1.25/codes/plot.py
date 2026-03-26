import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Define matrices
A = np.array([[-1, 1],
              [-1, -10]])
B = np.array([[0],
              [10]])
C = np.array([[1, 0]])
D = np.array([[0]])

sys_ss = ctrl.StateSpace(A, B, C, D)
sys_tf = ctrl.ss2tf(sys_ss)

print("Transfer Function:")
print(sys_tf)

# STEP RESPONSE
t, y = ctrl.step_response(sys_tf)
plt.figure()
plt.plot(t, y)
plt.title("Step Response (ω / u)")
plt.xlabel("Time (s)")
plt.ylabel("ω(t)")
plt.grid()

# IMPULSE RESPONSE
t, y = ctrl.impulse_response(sys_tf)
plt.figure()
plt.plot(t, y)
plt.title("Impulse Response")
plt.xlabel("Time (s)")
plt.ylabel("ω(t)")
plt.grid()

# BODE PLOT
plt.figure()
ctrl.bode(sys_tf, dB=True)

plt.show()
