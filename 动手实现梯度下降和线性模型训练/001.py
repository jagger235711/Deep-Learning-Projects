# %%
# experiment_linreg.py
import numpy as np
import time
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(0)

# %% Data generation

def make_data(n=200, d=50, cond=10.0, sigma=0.1):
    # generate singular values spaced to create given condition number
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(d, d))
    # singular values from 1 to 1/cond (for X shape nxd we use min(n,d))
    k = min(n, d)
    s = np.linspace(1.0, 1.0/cond, k)
    S = np.zeros((n, d))
    S[:k, :k] = np.diag(s)
    X = (U @ S @ V.T)[:n, :d]
    wstar = np.random.randn(d)
    y = X.dot(wstar) + sigma * np.random.randn(n)
    return X, y, wstar

# %% Optimization methods
def mse_loss_and_grad(w, X, y):
    # return loss (scalar) and gradient (vector)
    n = X.shape[0]
    r = X.dot(w) - y
    loss = 0.5 * np.mean(r**2)   # 0.5* MSE
    grad = (X.T.dot(r)) / n
    return loss, grad

def gradient_descent(X, y, w0=None, lr=1.0, max_iter=5000, tol=1e-8, record_every=10):
    n, d = X.shape
    if w0 is None:
        w = np.zeros(d)
    else:
        w = w0.copy()
    history = {'loss': [], 'time': [], 'grad_norm': []}
    t0 = time.perf_counter()
    for it in range(1, max_iter+1):
        loss, grad = mse_loss_and_grad(w, X, y)
        gnorm = np.linalg.norm(grad)
        w -= lr * grad
        if it % record_every == 0 or it == 1:
            history['loss'].append(loss)
            history['time'].append(time.perf_counter() - t0)
            history['grad_norm'].append(gnorm)
        if gnorm < tol:
            break
    return w, history

# Data
X, y, wstar = make_data(n=500, d=50, cond=100.0, sigma=0.1)
n, d = X.shape

# initial guess
w0 = np.zeros(d)

# %% Run optimizers
# --- Gradient Descent (choose lr carefully) ---
# A heuristic: lr ~ 1 / L where L = max eigenvalue of (X^T X)/n
eigvals = np.linalg.eigvalsh((X.T @ X) / n)
L = eigvals.max()
print("L (Lipschitz):", L)
lr = 1.0 / L * 0.9  # safe step
w_gd, hist_gd = gradient_descent(X, y, w0=w0, lr=lr, max_iter=2000, record_every=5)

# --- BFGS via SciPy (requires objective and grad) ---
def obj(w, X, y):
    loss, _ = mse_loss_and_grad(w, X, y)
    return loss

def grad_fn(w, X, y):
    _, g = mse_loss_and_grad(w, X, y)
    return g

t0 = time.perf_counter()
res = minimize(fun=partial(obj, X=X, y=y),
               x0=w0,
               jac=partial(grad_fn, X=X, y=y),
               method='BFGS',
               options={'gtol':1e-8, 'maxiter':500})
t_bfgs = time.perf_counter() - t0
w_bfgs = res.x
print("BFGS success:", res.success, "nit:", res.nit, "time:", t_bfgs, "final loss:", res.fun)

# %% Results
# --- Compare final results ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

print("GD final loss:", hist_gd['loss'][-1], "param_err:", np.linalg.norm(w_gd - wstar))
print("BFGS final loss:", obj(w_bfgs, X, y), "param_err:", np.linalg.norm(w_bfgs - wstar))
# %%
# --- Plot loss curves (loss vs time) ---
plt.figure()
plt.plot(hist_gd['time'], hist_gd['loss'], label='GD')
plt.axhline(obj(w_bfgs, X, y), color='C1', linestyle='--', label='BFGS final')
plt.xlabel('time (s)')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.show()

# %%
# --- Plot 2: data + true curve + fitted curves ---
plt.figure(figsize=(6, 4))

# 取第一个特征作为横坐标
x_axis = X[:, 0]

# 排序后绘制平滑曲线
sort_idx = np.argsort(x_axis)
x_sorted = x_axis[sort_idx]

y_true = X @ wstar
y_gd = X @ w_gd
y_bfgs = X @ w_bfgs

plt.scatter(x_axis, y, s=20, alpha=0.6, label="Data (noisy)")
plt.plot(x_sorted, y_true[sort_idx], "k-", lw=2, label="True function")
plt.plot(x_sorted, y_gd[sort_idx], "C0--", lw=2, label="GD fit")
plt.plot(x_sorted, y_bfgs[sort_idx], "C1-.", lw=2, label="BFGS fit")

plt.xlabel("X[:,0] (first feature)")
plt.ylabel("y")
plt.legend()
plt.title("Data and Fits")
plt.tight_layout()
plt.show()

# %%
# --- Plot: side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# (1) Loss vs Time
axes[0].plot(hist_gd["time"], hist_gd["loss"], label="GD")
axes[0].axhline(obj(w_bfgs, X, y), color="C1", linestyle="--", label="BFGS final")
axes[0].set_xlabel("time (s)")
axes[0].set_ylabel("loss")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].set_title("Loss vs Time")

# (2) Data and Fits
x_axis = X[:, 0]
sort_idx = np.argsort(x_axis)
x_sorted = x_axis[sort_idx]

y_true = X @ wstar
y_gd = X @ w_gd
y_bfgs = X @ w_bfgs

axes[1].scatter(x_axis, y, s=20, alpha=0.6, label="Data (noisy)")
axes[1].plot(x_sorted, y_true[sort_idx], "k-", lw=2, label="True function")
axes[1].plot(x_sorted, y_gd[sort_idx], "C0--", lw=2, label="GD fit")
axes[1].plot(x_sorted, y_bfgs[sort_idx], "C1-.", lw=2, label="BFGS fit")
axes[1].set_xlabel("X[:,0] (first feature)")
axes[1].set_ylabel("y")
axes[1].legend()
axes[1].set_title("Data and Fits")

plt.tight_layout()
plt.show()

# %%
import pandas as pd

# --- Compute metrics ---
mse_gd = mse(y, y_gd)
mse_bfgs = mse(y, y_bfgs)

results = pd.DataFrame(
    {
        "Method": ["Gradient Descent", "BFGS"],
        "Final Loss": [hist_gd["loss"][-1], obj(w_bfgs, X, y)],
        "MSE": [mse_gd, mse_bfgs],
        "Param Error (||w - w*||)": [
            np.linalg.norm(w_gd - wstar),
            np.linalg.norm(w_bfgs - wstar),
        ],
    }
)

print("\n=== Results Comparison ===")
print(results.to_string(index=False))
