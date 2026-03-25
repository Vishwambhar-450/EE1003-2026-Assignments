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

    print(f"\n--- {title} ---")
    print("1. E(XY) vs E(X)E(Y):", EXY, EX * EY)
    print("2. Cov(X,Y):", cov)
    print("3. Var(X+Y) vs Var(X)+Var(Y):", var_sum, varX + varY)
    print("4. E(X^2 Y^2) vs (E(X))^2 (E(Y))^2:", EX2Y2, (EX**2)*(EY**2))

    return EXY, EX*EY, cov, var_sum, varX+varY, EX2Y2, (EX**2)*(EY**2)

# CONTINUOUS CASE (PDF)
mux = np.random.randint(1, 5, 1)
muy = np.random.randint(1, 5, 1)
Xc = np.random.normal(mux, 1, N)
Yc = np.random.normal(muy, 1, N)

results_cont = analyze(Xc, Yc, "Continuous Case")

# Estimate PDFs using KDE
kde_X = gaussian_kde(Xc)
kde_Y = gaussian_kde(Yc)

x_range = np.linspace(-4, 4, 500)

plt.figure()
plt.plot(x_range, kde_X(x_range), label="PDF of X")
plt.plot(x_range, kde_Y(x_range), label="PDF of Y")
plt.title("Estimated PDFs (Continuous Case)")
plt.legend()
plt.grid()

# DISCRETE CASE (PMF)
Xd = np.random.randint(1, 7, N)
Yd = np.random.randint(1, 7, N)

results_disc = analyze(Xd, Yd, "Discrete Case")

# Compute PMFs
values_X, counts_X = np.unique(Xd, return_counts=True)
pmf_X = counts_X / N

values_Y, counts_Y = np.unique(Yd, return_counts=True)
pmf_Y = counts_Y / N

plt.figure()
plt.stem(values_X, pmf_X, label="PMF of X", basefmt=" ")
plt.stem(values_Y, pmf_Y, linefmt='r-', markerfmt='ro', label="PMF of Y", basefmt=" ")
plt.title("PMFs (Discrete Case)")
plt.legend()
plt.grid()

labels = [
    "Option 1",
    "Option 2",
    "Option 3",
    "Option 4"
]

# Continuous
cont_vals = [
    [results_cont[0], results_cont[1]],
    [results_cont[2], 0],
    [results_cont[3], results_cont[4]],
    [results_cont[5], results_cont[6]]
]

# Discrete
disc_vals = [
    [results_disc[0], results_disc[1]],
    [results_disc[2], 0],
    [results_disc[3], results_disc[4]],
    [results_disc[5], results_disc[6]]
]

plt.figure()
for i in range(4):
    plt.scatter([i, i], cont_vals[i])
plt.xticks(range(4), labels)
plt.title("Continuous Case Verification")
plt.grid()

plt.figure()
for i in range(4):
    plt.scatter([i, i], disc_vals[i])
plt.xticks(range(4), labels)
plt.title("Discrete Case Verification")
plt.grid()

plt.show()
