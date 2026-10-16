"""
Replication of Scheffer et al. (2009), "Early-warning signals for critical transitions".

Model (harvested population, Fig. 1 caption of the paper):
    dX/dt = X (1 - X/K) - c X^2 / (X^2 + 1)
    X = biomass, K = carrying capacity (10), c = maximum harvest rate

Part 1: fold bifurcation diagram + hysteresis   -> figures/fold_bifurcation.png
Part 2: early-warning signals (residuals, s.d. and AR(1)) -> figures/early_warning.png

Run:  python replication.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K = 10          # carrying capacity
SEED = 42       # random seed for the stochastic simulation


def f(x, c):
    """dX/dt: logistic growth minus saturating harvest."""
    return x * (1 - x / K) - c * x**2 / (x**2 + 1)


def run_to_equilibrium(x, c, t_end=200, dt=0.01):
    """Integrate dX/dt = f(X, c) (Euler) for a while so X settles at an equilibrium."""
    for _ in range(int(t_end / dt)):
        x = x + f(x, c) * dt
    return x


# =============================================================================
# Part 1: fold bifurcation and hysteresis
# =============================================================================
# Equilibria satisfy f(X, c) = 0. For X > 0 this gives c as a function of X:
#   c = (1 - X/K)(X^2 + 1) / X
x_eq = np.linspace(0.05, K, 5000)
c_eq = (1 - x_eq / K) * (x_eq**2 + 1) / x_eq

# Stability: an equilibrium is stable when the slope df/dX is negative there.
h = 1e-6
slope = (f(x_eq + h, c_eq) - f(x_eq - h, c_eq)) / (2 * h)
stable = slope < 0

# Fold points: where the curve turns back, i.e. where c(X) has a local min / max.
i_F1 = np.argmin(np.where(x_eq < 3, c_eq, np.inf))   # lower fold
i_F2 = np.argmax(np.where(x_eq > 3, c_eq, -np.inf))  # upper fold
c_F1, x_F1 = c_eq[i_F1], x_eq[i_F1]
c_F2, x_F2 = c_eq[i_F2], x_eq[i_F2]
print(f"F1 (recovery threshold): c = {c_F1:.3f}, X = {x_F1:.2f}")
print(f"F2 (collapse threshold): c = {c_F2:.3f}, X = {x_F2:.2f}")

# Hysteresis: slowly increase c, then slowly decrease it again,
# each time starting from the previous equilibrium.
c_values = np.arange(1.0, 3.2, 0.02)
x_up, x_down = [], []
x = 8.9
for c in c_values:
    x = run_to_equilibrium(x, c)
    x_up.append(x)
for c in c_values[::-1]:
    x = run_to_equilibrium(x, c)
    x_down.append(x)
x_down = x_down[::-1]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(c_eq[stable], x_eq[stable], "k.", ms=1, label="stable equilibrium")
ax.plot(c_eq[~stable], x_eq[~stable], "k--", lw=1, label="unstable equilibrium")
ax.plot(c_values, x_up, "o", ms=3, color="tab:blue", label="c increasing")
ax.plot(c_values, x_down, "o", ms=3, color="tab:orange", mfc="none", label="c decreasing")
ax.plot([c_F1, c_F2], [x_F1, x_F2], "ko", mfc="white", ms=7)
ax.text(c_F1 + 0.06, x_F1 - 0.6, f"F1 (c = {c_F1:.2f})")
ax.text(c_F2 + 0.03, x_F2 + 0.3, f"F2 (c = {c_F2:.2f})")
ax.set_xlim(1, 3.2)
ax.set_ylim(0, 10)
ax.set_xlabel("Harvest rate c")
ax.set_ylabel("Biomass X")
ax.set_title("Fold bifurcation and hysteresis")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("figures/fold_bifurcation.png", dpi=200)


# =============================================================================
# Part 2: early-warning signals
# =============================================================================
# Stochastic simulation: dX = f(X, c) dt + sigma dW, with c slowly increasing.
SIGMA = 0.1              # noise intensity (not given in the paper; our choice)
T, DT = 1000, 0.01       # run length and time step
c_start, c_end = 1.0, 2.8

rng = np.random.default_rng(SEED)
n = int(T / DT)
t_all = np.arange(n) * DT
c_all = c_start + (c_end - c_start) * t_all / T
x_all = np.empty(n)
x = 8.9
for i in range(n):
    x = x + f(x, c_all[i]) * DT + SIGMA * np.sqrt(DT) * rng.standard_normal()
    x = max(x, 0)
    x_all[i] = x

# Keep one observation per time unit (like a yearly measurement).
data = pd.DataFrame({"t": t_all, "c": c_all, "x": x_all}).iloc[::100].reset_index(drop=True)

# Only analyse the part before the transition (before c reaches F2).
before = data[data["c"] < c_F2].copy()

# Remove the slow trend with a moving average, keep the fluctuations (residuals).
before["trend"] = before["x"].rolling(51, center=True, min_periods=1).mean()
before["resid"] = before["x"] - before["trend"]

# Moving-window indicators (window = half of the pre-transition series).
window = len(before) // 2
before["sd"] = before["resid"].rolling(window).std()
before["ar1"] = before["resid"].rolling(window).apply(lambda w: w.autocorr(lag=1))


def kendall_tau(y):
    """Kendall's tau between time and y: +1 = always increasing, 0 = no trend."""
    y = y.dropna().to_numpy()
    s = np.sign(np.subtract.outer(y, y))
    n = len(y)
    return np.triu(-s, 1).sum() / (n * (n - 1) / 2)


print(f"Kendall tau s.d.:  {kendall_tau(before['sd']):.2f}")
print(f"Kendall tau AR(1): {kendall_tau(before['ar1']):.2f}")

fig, axes = plt.subplots(4, 1, figsize=(6, 9), sharex=True)
axes[0].plot(data["t"], data["x"], lw=0.8, color="tab:blue")
axes[0].plot(before["t"], before["trend"], "k", lw=1.5, label="trend")
axes[0].axvline(before["t"].iloc[-1], color="grey", ls=":", label="F2 reached")
axes[0].set_ylabel("Biomass X")
axes[0].set_title("Harvest rate c increases from 1.0 to 2.8 over time")
axes[0].legend(frameon=False, fontsize=8)

axes[1].plot(before["t"], before["resid"], lw=0.6, color="tab:blue")
axes[1].axhline(0, color="grey", lw=0.8)
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals after removing the trend")

axes[2].plot(before["t"], before["sd"], color="tab:blue")
axes[2].set_ylabel("s.d.")
axes[2].set_title(f"Standard deviation (Kendall tau = {kendall_tau(before['sd']):.2f})")

axes[3].plot(before["t"], before["ar1"], color="tab:blue")
axes[3].set_ylabel("AR(1)")
axes[3].set_title(f"Lag-1 autocorrelation (Kendall tau = {kendall_tau(before['ar1']):.2f})")
axes[3].set_xlabel("Time")
fig.tight_layout()
fig.savefig("figures/early_warning.png", dpi=200)

print("Figures saved in the folder 'figures'.")
