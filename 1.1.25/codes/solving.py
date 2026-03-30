import numpy as np
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
