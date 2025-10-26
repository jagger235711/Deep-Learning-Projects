# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --- Step 1: 数据生成 ---
def make_nonlinear_data(n=500, d=1, noise=0.1):
    X = torch.linspace(-3, 3, n).reshape(-1, d)
    y = torch.sin(X) + 0.3 * torch.cos(2 * X) + noise * torch.randn_like(X)
    return X, y


X, y = make_nonlinear_data()
X, y = X.to(device), y.to(device)


# --- Step 2: 模型定义 ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# --- Step 3: 训练函数 ---
def train(model, optimizer, max_iter=500, record_loss=True, name=""):
    model.train()
    losses = []
    start = time.time()
    for i in range(max_iter):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        if record_loss:
            losses.append(loss.item())
    elapsed = time.time() - start
    return losses, elapsed, len(losses)


# --- Step 4: 训练 - SGD ---
model_gd = MLP().to(device)
optimizer_gd = optim.SGD(model_gd.parameters(), lr=0.01, momentum=0.9)
losses_gd, time_gd, iters_gd = train(model_gd, optimizer_gd, max_iter=1000)

# --- Step 5: 训练 - L-BFGS ---
model_bfgs = MLP().to(device)
optimizer_bfgs = optim.LBFGS(
    model_bfgs.parameters(), lr=0.8, max_iter=100, history_size=10
)

losses_bfgs = []
start = time.time()


def closure():
    optimizer_bfgs.zero_grad()
    y_pred = model_bfgs(X)
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    losses_bfgs.append(loss.item())
    return loss


optimizer_bfgs.step(closure)
time_bfgs = time.time() - start
iters_bfgs = len(losses_bfgs)

# --- Step 6: 绘图 ---
with torch.no_grad():
    x_plot = torch.linspace(-3, 3, 500).reshape(-1, 1).to(device)
    y_true = torch.sin(x_plot) + 0.3 * torch.cos(2 * x_plot)
    y_gd = model_gd(x_plot)
    y_bfgs = model_bfgs(x_plot)

# 图1: 拟合曲线
plt.figure(figsize=(8, 5))
plt.scatter(X.cpu(), y.cpu(), s=10, alpha=0.4, label="Data")
plt.plot(x_plot.cpu(), y_true.cpu(), "k--", lw=2, label="True function")
plt.plot(x_plot.cpu(), y_gd.cpu(), "C0-", lw=2, label="GD fit")
plt.plot(x_plot.cpu(), y_bfgs.cpu(), "C1-", lw=2, label="BFGS fit")
plt.title("Function Fitting (Nonlinear Model)")
plt.legend()
plt.show()

# 图2: Loss曲线
plt.figure(figsize=(8, 5))
plt.plot(losses_gd, label=f"GD (time={time_gd:.2f}s)")
plt.plot(losses_bfgs, label=f"BFGS (time={time_bfgs:.2f}s)")
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.title("Loss Curve Comparison")
plt.legend()
plt.show()


# --- Step 7: 结果统计 ---
def mse(y_true, y_pred):
    return float(((y_true - y_pred) ** 2).mean().item())


mse_gd = mse(y, model_gd(X))
mse_bfgs = mse(y, model_bfgs(X))

results = pd.DataFrame(
    [
        ["GD (SGD)", losses_gd[-1], mse_gd, np.nan, time_gd, iters_gd],
        ["BFGS", losses_bfgs[-1], mse_bfgs, np.nan, time_bfgs, iters_bfgs],
    ],
    columns=[
        "Method",
        "Final Loss",
        "MSE",
        "Param Error (||w - w*||)",
        "Total Time (s)",
        "Iters",
    ],
)

print("\n=== Results Comparison ===")
print(results.to_string(index=False))


# --- Step 8: Time-to-eps (收敛时间分析) ---
def time_to_eps(losses, total_time, eps_values):
    times = []
    n = len(losses)
    for eps in eps_values:
        idx = next((i for i, l in enumerate(losses) if l < eps), None)
        if idx is None:
            times.append(np.inf)
        else:
            times.append((idx / n) * total_time)
    return times


eps_values = [1e-2, 1e-4, 1e-6]
time_eps_gd = time_to_eps(losses_gd, time_gd, eps_values)
time_eps_bfgs = time_to_eps(losses_bfgs, time_bfgs, eps_values)

time_eps_df = pd.DataFrame(
    {"eps": eps_values, "GD_time": time_eps_gd, "BFGS_time": time_eps_bfgs}
)

print("\n=== Time-to-eps (s) ===")
print(time_eps_df.to_string(index=False))
