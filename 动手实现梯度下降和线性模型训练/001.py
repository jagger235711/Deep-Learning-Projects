# %%
# experiment_linreg.py
import numpy as np
import time
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging

logging.basicConfig(
    level=logging.DEBUG
)

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


def make_data_nonlinear_sigmoid(n=200, d=50, sigma=0.1):
    X = np.random.randn(n, d)
    wstar = np.random.randn(d)
    z = X.dot(wstar)
    y = 1 / (1 + np.exp(-z)) + sigma * np.random.randn(n)
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
# X, y, wstar = make_data(n=500, d=50, cond=100.0, sigma=0.1)
X, y, wstar = make_data(n=1000, d=200, cond=1000.0, sigma=0.5)
# X, y, wstar = make_data_nonlinear_sigmoid(n=1000, d=50, sigma=0.1)

n, d = X.shape

# initial guess
w0 = np.zeros(d)

# %% Run optimizers
# --- PyTorch Gradient Descent ---
import torch

def pytorch_gradient_descent(X, y, w0=None, lr=1.0, max_iter=5000, tol=1e-8, record_every=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.from_numpy(X).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)

    w = torch.nn.Parameter(torch.tensor(w0, device=device) if w0 is not None 
                          else torch.zeros(X.shape[1], device=device))
    optimizer = torch.optim.SGD([w], lr=lr)
    history = {'loss': [], 'time': [], 'grad_norm': []}
    t0 = time.perf_counter()

    for it in range(1, max_iter+1):
        optimizer.zero_grad()
        logging.debug("X_tensor.dtype %s",X_tensor.dtype)
        logging.debug("w.dtype %s",w.dtype)
        y_pred = X_tensor @ w.float()
        loss = 0.5 * torch.mean((y_pred - y_tensor)**2)
        loss.backward()
        grad_norm = torch.norm(w.grad).item()
        optimizer.step()

        if it % record_every == 0 or it == 1:
            history['loss'].append(loss.item())
            history['time'].append(time.perf_counter() - t0)
            history['grad_norm'].append(grad_norm)

        if grad_norm < tol:
            break

    return w.detach().cpu().numpy(), history

# --- Original Gradient Descent ---
# A heuristic: lr ~ 1 / L where L = max eigenvalue of (X^T X)/n
eigvals = np.linalg.eigvalsh((X.T @ X) / n)
L = eigvals.max()
print("L (Lipschitz):", L)
lr = 1.0 / L * 0.9  # safe step
# 运行PyTorch梯度下降
w_torch, hist_torch = pytorch_gradient_descent(X, y, w0=w0, lr=lr, max_iter=2000, record_every=5)

# 运行原始梯度下降
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

# 计算各方法预测值
y_gd = X @ w_gd
y_torch = X @ w_torch
y_bfgs = X @ w_bfgs

print("Manual GD final loss:", hist_gd['loss'][-1], "param_err:", np.linalg.norm(w_gd - wstar)) 
print("PyTorch GD final loss:", hist_torch['loss'][-1], "param_err:", np.linalg.norm(w_torch - wstar))
print("BFGS final loss:", obj(w_bfgs, X, y), "param_err:", np.linalg.norm(w_bfgs - wstar))

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Step 1: 主成分分析（降到一维，用作横轴） ---
pca = PCA(n_components=1)
x_pca = pca.fit_transform(X).ravel()  # shape (n,)

# --- Step 2: 按投影方向排序 ---
sort_idx = np.argsort(x_pca)
x_sorted = x_pca[sort_idx]

# --- Step 3: 各模型预测 ---
y_true = X @ wstar
y_gd = X @ w_gd
y_bfgs = X @ w_bfgs
y_torch_pred = X @ w_torch  # 假设你也有 torch 拟合的权重

# --- Step 4: 绘图 ---
plt.figure(figsize=(12, 4))

plt.scatter(x_pca, y, s=20, alpha=0.6, label="Data (noisy)")
plt.plot(x_sorted, y_true[sort_idx], "k-", lw=2, label="True function")
plt.plot(x_sorted, y_gd[sort_idx], "C0--", lw=2, label="Manual GD fit")
plt.plot(x_sorted, y_torch_pred[sort_idx], "C2:", lw=2, label="PyTorch GD fit")
plt.plot(x_sorted, y_bfgs[sort_idx], "C1-.", lw=2, label="BFGS fit")

plt.xlabel("1st Principal Component of X")
plt.ylabel("y")
plt.legend()
plt.title("Data and Fits (projected onto PCA-1 direction)")
plt.tight_layout()
plt.show()

# %%
import time
from functools import partial
from scipy.optimize import minimize

# 容器用于存历史
bfgs_hist = {"loss": [], "time": [], "w": []}


def obj_only(w, X, y):
    r = X.dot(w) - y
    return 0.5 * np.mean(r**2)


def grad_only(w, X, y):
    r = X.dot(w) - y
    return (X.T.dot(r)) / X.shape[0]


# callback 接收当前参数向量（scipy 对于 BFGS 支持 callback(xk)）
def make_callback(X, y, hist, t0):
    def callback(xk):
        hist["w"].append(xk.copy())
        hist["loss"].append(obj_only(xk, X, y))
        hist["time"].append(time.perf_counter() - t0)

    return callback


w0 = np.zeros(d)
t0 = time.perf_counter()
res = minimize(
    fun=partial(obj_only, X=X, y=y),
    x0=w0,
    jac=partial(grad_only, X=X, y=y),
    method="BFGS",
    callback=make_callback(X, y, bfgs_hist, t0),
    options={"gtol": 1e-8, "maxiter": 500, "disp": False},
)
bfgs_time_total = time.perf_counter() - t0

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from functools import partial
from scipy.optimize import minimize

# 假设已有 X, y, wstar, gradient_descent, mse, obj_only, grad_only
# 并且 hist_gd, w_gd 已经得出（如果还没，可先跑 GD）

# 1) 记录 BFGS 历史 (如前面)
bfgs_hist = {"loss": [], "time": [], "w": []}
t0 = time.perf_counter()
res = minimize(
    fun=partial(obj_only, X=X, y=y),
    x0=np.zeros(d),
    jac=partial(grad_only, X=X, y=y),
    method="BFGS",
    callback=make_callback(X, y, bfgs_hist, t0),
    options={"gtol": 1e-12, "maxiter": 500, "disp": False},
)
bfgs_total_time = time.perf_counter() - t0
w_bfgs = res.x

# 2) compute predictions
y_gd = X @ w_gd
y_bfgs = X @ w_bfgs

# 3) compute MSE etc
mse_gd = mse(y, y_gd)
mse_bfgs = mse(y, y_bfgs)
param_err_gd = np.linalg.norm(w_gd - wstar)
param_err_bfgs = np.linalg.norm(w_bfgs - wstar)

# 4) time-to-eps (示例 eps)
eps_list = [1e-2, 1e-4, 1e-6]
target_base = min(bfgs_hist["loss"][-1], hist_gd["loss"][-1])


def t_to_eps(hist_loss, hist_time, eps):
    target = target_base + eps
    for L, t in zip(hist_loss, hist_time):
        if L <= target:
            return t
    return np.inf


rows = []
for eps in eps_list:
    rows.append(
        {
            "eps": eps,
            "GD_time": t_to_eps(hist_gd["loss"], hist_gd["time"], eps),
            "torch_time": t_to_eps(hist_torch["loss"], hist_torch["time"], eps),
            "BFGS_time": t_to_eps(bfgs_hist["loss"], bfgs_hist["time"], eps),
        }
    )
time_df = pd.DataFrame(rows)

# 5) Plot side-by-side (loss vs time includes both histories)
fig, ax = plt.subplots(1, 1, figsize=(7, 4))

# loss vs time
ax.plot(hist_gd["time"], hist_gd["loss"], label="GD", color="C0")
ax.plot(hist_torch["time"], hist_torch["loss"], label="PyTorch GD", color="C1")
ax.plot(
    bfgs_hist["time"],
    bfgs_hist["loss"],
    label="BFGS",
    color="C2",
)
ax.set_xlabel("time (s)")
ax.set_ylabel("loss")
ax.set_yscale("log")
ax.legend()
ax.set_title("Loss vs Time (both methods)")


plt.show()

# 6) Pretty results table
results = pd.DataFrame(
    {
        "Method": ["Manual GD", "PyTorch GD", "BFGS"],
        "Final Loss": [hist_gd["loss"][-1], hist_torch["loss"][-1], obj(w_bfgs, X, y)],
        "MSE": [mse_gd, mse(y, X @ w_torch), mse_bfgs],
        "Param Error (||w - w*||)": [
            np.linalg.norm(w_gd - wstar),
            np.linalg.norm(w_torch - wstar),
            np.linalg.norm(w_bfgs - wstar),
        ],
        "Total Time (s)": [
            hist_gd["time"][-1],
            hist_torch["time"][-1],
            bfgs_total_time,
        ],
        "Iters": [len(hist_gd["loss"]), len(hist_torch["loss"]), res.nit],
    }
)
print("\n=== Results Comparison ===")
print(results.to_string(index=False))
print("\n=== Time-to-eps (s) ===")
print(time_df.to_string(index=False))
