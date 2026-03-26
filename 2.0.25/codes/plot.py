import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

x = cp.Variable(pos=True)
objective = cp.Maximize(cp.entr(x))
constraints = [x >= 1e-6]

prob = cp.Problem(objective, constraints)
prob.solve()

x_opt = x.value
y_opt = np.exp(-x_opt * np.log(x_opt))

x_vals = np.linspace(0.01, 2, 500)
y_vals = np.exp(-x_vals * np.log(x_vals))

plt.figure()
plt.plot(x_vals, y_vals, label="y = x^(-x)")
plt.scatter(x_opt, y_opt)
plt.text(x_opt, y_opt,
         f"Max\nx={x_opt:.3f}\ny={y_opt:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Maximization of y = x^(-x)")
plt.legend()
plt.grid()

plt.show()
