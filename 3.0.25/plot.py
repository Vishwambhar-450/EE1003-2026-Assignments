import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Number of samples
N = 200000

def analyze(X, Y, title):
    EX = np.mean(X)
    EY = np.mean(Y)
    EXY = np.mean(X * Y)
    cov = np.mean((X - EX) * (Y - EY))
    varX = np.var(X)
    varY = np.var(Y)
    var_sum = np.var(X + Y)
    EX2Y2 = np.mean((X**2) * (Y**2))

    print(f"\n================ {title} ================")

    print("\n--- Option 1: E(XY) vs E(X)E(Y) ---")
    print(f"E(XY)        = {EXY:.2f}")
    print(f"E(X)E(Y)     = {EX * EY:.2f}")

    print("\n--- Option 2: Cov(X,Y) ---")
    print(f"Cov(X,Y)     = {cov:.2f} (should be ~0 if independent)")

    print("\n--- Option 3: Var(X+Y) vs Var(X)+Var(Y) ---")
    print(f"Var(X+Y)     = {var_sum:.2f}")
    print(f"Var(X)+Var(Y)= {varX + varY:.2f}")

    print("\n--- Option 4: E(X^2 Y^2) vs (E(X))^2 (E(Y))^2 ---")
    print(f"E(X^2 Y^2)   = {EX2Y2:.2f}")
    print(f"(E(X))^2(E(Y))^2 = {(EX**2)*(EY**2):.2f}")

    return EXY, EX*EY, cov, var_sum, varX+varY, EX2Y2, (EX**2)*(EY**2)


# ================= CONTINUOUS CASE =================
mux = np.random.randint(1, 5)
muy = np.random.randint(1, 5)

print(f"mux = {mux}")
print(f"muy = {muy}")


Xc = np.random.normal(mux, 1, N)
Yc = np.random.normal(muy, 1, N)

results_cont = analyze(Xc, Yc, "Continuous Case")

# KDE PDFs
kde_X = gaussian_kde(Xc)
kde_Y = gaussian_kde(Yc)

x_range = np.linspace(-4, 4, 500)

plt.figure()
plt.plot(x_range, kde_X(x_range), label="PDF of X")
plt.plot(x_range, kde_Y(x_range), label="PDF of Y")
plt.title("Estimated PDFs (Continuous Case)")
plt.legend()
plt.grid()

labels = ["Option 1", "Option 2", "Option 3", "Option 4"]

cont_vals = [
    [results_cont[0], results_cont[1]],
    [results_cont[2], 0],
    [results_cont[3], results_cont[4]],
    [results_cont[5], results_cont[6]]
]

plt.figure()
for i in range(4):
    plt.scatter([i, i], cont_vals[i])
plt.xticks(range(4), labels)
plt.title("Continuous Case Verification")
plt.grid()

plt.show()
