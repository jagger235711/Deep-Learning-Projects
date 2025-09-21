# 实验001 动手实现梯度下降和线性模型的训练并且比较梯度下降和拟牛顿法的速度
## 实验目标
- 动手实现梯度下降算法
- 通过实现的算法进行线性模型的训练
- 比较梯度下降原生实现、手动实现和拟牛顿法的性能差距
## 实验设计
1. 实现通过给出一个具体的线性模型生成相应的带噪声的数据集
2. 实现梯度下降算法
3. 实现拟牛顿法
4. 使用框架自带实现
5. 进行线性模型的训练

---
太棒了——你这个实验目标很清晰，适合入门理解「梯度下降的工作原理、收敛行为以及与二阶方法（拟牛顿法）差异」的直观认识。下面我把你的草案**优化成可复现、可比较、容易扩展**的实验方案，并给出实现要点、超参数建议、评价指标、实验矩阵（哪些变量要扫）以及可直接复制的 Python 代码骨架（NumPy + SciPy + 可选 PyTorch 版本）——你可以拿去改跑。

# 实验总体思路

1. 生成线性回归问题 $y = X w^\star + \varepsilon$，可控制样本量 $n$、特征维度 $d$、噪声强度、以及特征矩阵 $X$ 的病态（condition number）。
2. 分别用以下方法拟合参数 $w$：

   * 自己实现的**批量（全梯度）梯度下降（GD）**
   * 自己实现的**小批量随机梯度下降（SGD/mini-batch）**（可选）
   * 使用 **拟牛顿法（BFGS / L-BFGS）**（调用 SciPy）
   * 框架自带优化器（PyTorch 的 LBFGS / Adam / SGD）——对照
3. 比较：迭代次数、运行时间、到达的训练损失、参数误差 $\|w-w^\star\|$、梯度范数、收敛曲线（loss vs iter、loss vs time）。
4. 做多组实验（不同 condition number、不同维度、不同噪声、不同学习率），观察差异与规律。

# 评价指标（必须记录）

* 最终训练 MSE（或均方误差），以及在一个独立验证集上的 MSE。
* 参数误差：$\|w_{\text{est}}-w^\star\|_2$。
* 迭代次数（或函数/梯度调用数）。
* 实际运行时间（wall-clock，`time.perf_counter()`）。
* 每次迭代的梯度范数 $\|\nabla L(w)\|_2$ 或相对下降率。
* 收敛曲线（loss vs iter、loss vs time）、对数刻度绘图也很有用（观察线性/指数衰减）。

# 关键实验变量（建议做成矩阵）

* `n`（样本数）: e.g. \[100, 1000, 10000]
* `d`（特征维度）: e.g. \[10, 100, 500]
* condition number（良态/病态）: e.g. \[1 (nice), 10, 100, 1000]
* 噪声标准差 `sigma`: e.g. \[0.01, 0.1, 1]
* 学习率 `lr`：需要为每种设置调参（或使用学习率衰减策略）
* 批量大小（对于 SGD）: e.g. \[1, 32, n]

# 设计细节与技巧（很重要）

* **可重复性**：固定随机种子（`np.random.seed(0)`）。
* **标准化**：为了公平比较，先对 `X` 做列均值0标准差1（或不做，观察效果差异；病态问题可不做标准化以产生困难）。
* **停止准则**：最大迭代 `max_iter`、梯度范数阈值 `tol`、或 loss 改变量小于阈值。
* **记录频率**：不要每次迭代都测时间（会影响性能），记录每 `k` 步的 loss/time（例如每 1 或每 10 步）。
* **时间测量**：分别测「纯算法时间」和「框架开销时间」（比如 PyTorch 需要 `torch.no_grad()`）。
* **拟牛顿使用**：用 `scipy.optimize.minimize` 的 `method='BFGS'` 或 `method='L-BFGS-B'` 来求解最小化问题（给出目标函数和梯度）。
* **数值稳定性**：对多维问题，检查 $X^T X$ 的特征值分布，画谱图能帮助理解为什么有些方法慢（尤其在病态问题上）。

# 预期观察（供你验证）

* 对良条件的问题，GD 收敛快且稳定；拟牛顿法通常在迭代次数上极其优越（超快），但每步开销更大（计算与矩阵操作）。
* 对高维和大数据（n ≫ d 或 n ≫）时，L-BFGS（有限记忆）比 BFGS 更实用。
* 在病态（condition number 高）时，GD 收敛会拖慢很多，拟牛顿法/二阶信息表现更好。
* SGD 在大数据上节省时间/内存，但震荡且收敛到附近（需要学习率衰减或动量）。

# 实验步骤（详细）

1. 生成 $X$（控制 condition）与 $w^\star$，生成 $y$。
2. 划分训练 / 验证集（例如 80/20）。
3. 为每种方法做多次独立运行（不同随机种子），统计均值与方差。
4. 绘制对比图（loss vs iter、loss vs time、参数误差 vs iter）。
5. 写实验报告，解释观察到的现象（结合谱图、step-size 等）。

---

# 参考实现（可直接运行的骨架，NumPy + SciPy）

下面的代码是最小、清晰的实现骨架，包含数据生成、GD、BFGS 比较与绘图（你可把它复制到一个 `.py` 或 notebook 里跑）。若要我直接运行并给你图表，我也可以用 python\_user\_visible 来执行并展示（你要我就告诉我）。

```python
# experiment_linreg.py
import numpy as np
import time
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(0)

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

# --- Compare final results ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

print("GD final loss:", hist_gd['loss'][-1], "param_err:", np.linalg.norm(w_gd - wstar))
print("BFGS final loss:", obj(w_bfgs, X, y), "param_err:", np.linalg.norm(w_bfgs - wstar))

# --- Plot loss curves (loss vs time) ---
plt.figure()
plt.plot(hist_gd['time'], hist_gd['loss'], label='GD')
plt.axhline(obj(w_bfgs, X, y), color='C1', linestyle='--', label='BFGS final')
plt.xlabel('time (s)')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.show()
```

**说明与改进建议**

* 我把 loss 定义为 $0.5*\text{MSE}$，因此导数简单。你可改为普通 MSE，但要同步调整步长估计。
* `L = max eigenvalue((X^T X)/n)` 用来估计 Lipschitz 常数，给出 GD 的稳定步长上界；这是理解为何某些 lr 会发散的好方法。
* SciPy 的 BFGS 在中小维度上效果很好；高维（d 很大）时用 `method='L-BFGS-B'` 并设置 `options={'maxiter':..., 'm':10}`。

# 进一步可做的扩展实验（进阶）

* 对比 **梯度下降（GD）**、**共轭梯度法（CG）**、**牛顿法（直接 Hessian）**，尤其当损失是二次函数时（线性回归），CG 对稀疏/大维二次问题非常有效。
* 用 **预条件（preconditioning）**：用 $D^{-1}$ 缩放特征（或对 X 做 PCA / SVD 截断），观察收敛速度变化。
* 比较 **时间复杂度**：记录每种方法每次迭代的计算复杂度（矩阵乘法、内存占用）。
* 在**真实数据集**上测试（比如小型 UCI 数据）验证结论是否迁移。
* 引入 **噪声/错配模型**：当模型不完全线性（有模型误差）时，各优化器的行为会怎样？

# 常见坑与注意

* 学习率 `lr` 选得太大：GD 发散；太小：收敛极慢。用 `L` 来估计上界是个好习惯。
* SciPy 的 BFGS 需要提供精确梯度以获得最佳表现（避免 finite-diff）。
* 记录时间时确保只计算法时间，不把 plot/打印时间计入。
* 在比较“速度”时，明确是指“达到相同精度所花费的时间”还是“每次迭代所花费的时间”——两者会得出不同结论。

---

如果你愿意，我可以：

* 把上面代码直接在 notebook 里跑一组默认配置并把结果图表给你（loss vs time、参数误差等），或者
* 帮你把代码扩展成完整的实验脚本（自动扫不同 condition number / d / n 并输出 CSV 汇总结果与图）。

你想先看**我跑一组示例结果**，还是直接要一个更完整的实验脚本供你在本地批量跑？
--- 
好 — 我把那段代码**逐块拆开讲清楚**，把每行背后的数学含义、实现动机、潜在坑、以及常见改进/调参技巧都讲明白。为了结构清晰，我分成几部分：总体、逐函数/逐段解释、关键数学细节与理由、性能/复杂度分析、调试与改进建议。正文尽量精炼但全面 — 有任何一块想要更深入（比如画谱、跑示例、做梯度检验的代码），告诉我我可以直接给你对应的片段。

# 总体说明（这段代码在做什么）

这段脚本实现并比较了两种求线性回归参数的优化方法：

1. 生成一个可控\*\*条件数（condition number）\*\*的线性回归问题 $y = X w^\star + \varepsilon$。
2. 用自写的\*\*批量梯度下降（GD）\*\*训练参数 $w$（并记录 loss / 时间 / 梯度范数）。
3. 用 `scipy.optimize.minimize(..., method='BFGS')`（拟牛顿法）求解同一问题（使用解析梯度）。
4. 比较两者的最终 loss、参数误差，并把 GD 的 loss 曲线和 BFGS 的最终值画出来。

# 逐块解释代码（按函数/段落）

### 开头的 imports 和随机种子

```python
import numpy as np
import time
from functools import partial
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(0)
```

* `numpy` 做数值计算；`time.perf_counter()` 用来精确测墙钟时间；`partial` 用来把 `X,y` 绑定到优化调用上；`matplotlib` 画图；`scipy.optimize.minimize` 提供 BFGS。
* 固定随机种子保证可重复性。

---

### `make_data(n, d, cond, sigma)`

```python
U, _ = np.linalg.qr(np.random.randn(n, n))
V, _ = np.linalg.qr(np.random.randn(d, d))
k = min(n, d)
s = np.linspace(1.0, 1.0/cond, k)
S = np.zeros((n, d))
S[:k, :k] = np.diag(s)
X = (U @ S @ V.T)[:n, :d]
wstar = np.random.randn(d)
y = X.dot(wstar) + sigma * np.random.randn(n)
```

* 目的：构造一个矩阵 $X$ ，其奇异值（singular values）被控制，使得 `cond` 成为 $X$ 的条件数（近似）。
* 做法：先用 QR 得到两组正交矩阵 $U\in\mathbb{R}^{n\times n}, V\in\mathbb{R}^{d\times d}$。构造对角矩阵 `S`，其对角元素从 1 线性降到 $1/\text{cond}$，再做 $X = U S V^\top$。这样 $X$ 的最大奇异值≈1，最小≈1/cond，因而 condition number ≈ cond。
* `wstar` 为真实参数，`y = X wstar + noise`，`sigma` 控制噪声强度。
* 注：`(U @ S @ V.T)[:n, :d]` 在这里其实是冗余的（`U@S@V.T` 本应已经是 n×d），但不影响结果。

---

### `mse_loss_and_grad(w, X, y)`

```python
n = X.shape[0]
r = X.dot(w) - y
loss = 0.5 * np.mean(r**2)   # 0.5* MSE
grad = (X.T.dot(r)) / n
```

* `loss = 0.5 * mean(r^2)`，写成 $f(w)=\tfrac{1}{2n}\|Xw-y\|^2$。乘 0.5 是为了方便导数（导数中不会出现 2）。
* 梯度：$\nabla f(w)=\frac{1}{n}X^\top(Xw-y)$。代码中 `X.T.dot(r)/n` 正是这个式子。
* 重要：梯度是解析的，能被直接传给 BFGS（比数值差分更精确、快）。

---

### `gradient_descent(...)`

```python
if w0 is None: w = np.zeros(d)
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
```

* `w` 初始（默认为 0 向量）。
* 每次迭代：计算当前 loss、梯度，做标准梯度下降更新 `w = w - lr * grad`。
* `record_every` 决定仅每隔若干步记录一次历史，避免记录过多开销（但注意计算 loss/grad 本身在每步都执行了）。
* 停止条件：梯度范数小于 `tol`（常见），或达到 `max_iter`。
* `history` 中保存了 loss、从 start 到当前的耗时、梯度范数，便于画 loss-vs-time。

---

### 主流程：生成数据、估计 Lipschitz 常数、跑 GD

```python
X, y, wstar = make_data(n=500, d=50, cond=100.0, sigma=0.1)
eigvals = np.linalg.eigvalsh((X.T @ X) / n)
L = eigvals.max()
lr = 1.0 / L * 0.9  # safe step
w_gd, hist_gd = gradient_descent(..., lr=lr)
```

* 对二次损失 $f(w)=\tfrac{1}{2n}\|Xw-y\|^2$，梯度的 Lipschitz 常数是 $L=\lambda_{\max}\!\big(\frac{X^\top X}{n}\big)$。
* 只要步长 `lr < 1/L`，GD 就是收敛的（对于纯二次问题）。因此用 `lr = 0.9/L` 是较保守的选择。
* 注意：计算全部特征值 `eigvalsh` 的复杂度是 $O(d^3)$，当 `d` 很大时应改为**幂迭代**估计最大特征值。

---

### 拟牛顿（BFGS）调用

```python
def obj(w, X, y): loss, _ = mse_loss_and_grad(w, X, y); return loss
def grad_fn(w, X, y): _, g = mse_loss_and_grad(w, X, y); return g

res = minimize(fun=partial(obj, X=X, y=y),
               x0=w0,
               jac=partial(grad_fn, X=X, y=y),
               method='BFGS',
               options={'gtol':1e-8, 'maxiter':500})
```

* `minimize` 接受目标函数和雅可比（梯度），`method='BFGS'` 使用 BFGS 更新来近似（逆）Hessian，从而得到超线性收敛。
* `res` 返回优化结果：`res.x`（参数），`res.success`（是否收敛）、`res.nit`（迭代数）、`res.fun`（最后的 loss）等。
* BFGS 在维度不太大时非常快；但它需要维护一个 $d\times d$ 的矩阵近似（内存和每步的计算复杂度高）。若 `d` 很大，用 `method='L-BFGS-B'`（有限记忆）更合适。

---

### 比较与画图

```python
print("GD final loss:", hist_gd['loss'][-1], "param_err:", np.linalg.norm(w_gd - wstar))
print("BFGS final loss:", obj(w_bfgs, X, y), "param_err:", np.linalg.norm(w_bfgs - wstar))

plt.plot(hist_gd['time'], hist_gd['loss'], label='GD')
plt.axhline(obj(w_bfgs, X, y), ...)
plt.yscale('log')
```

* 输出每种方法的最终 loss 与参数误差 $\|w-w^\star\|$（衡量估计的好坏）。
* 把 GD 的 loss 随时间画出，同时用水平线标出 BFGS 的最终 loss，`yscale('log')` 用对数刻度更容易看收敛速率差距。

# 关键数学与工程要点（为什么要这样做）

* **用 0.5\*MSE**：让梯度表达式更简洁（避免 2）。
* **Lipschitz 常数 L**：对二次问题，步长上界为 $1/L$。在实践中直接用 `1/L * factor` 是稳健做法。
* **BFGS 的优点**：通过近似二阶信息（Hessian）能大幅减少迭代次数，常在中小维度问题中比 GD 快很多（以达到同样精度的时间计）。
* **分母 `n`**：把 loss 定义为平均（mean），而不是 sum，这样在改变 `n` 时步长估计不用额外缩放，参数尺度统一。

# 性能/复杂度对比（简述）

* 每次 GD 迭代成本：主要是 `X.dot(w)` 和 `X.T.dot(r)`，两者都是 $O(n d)$。
* BFGS 每次迭代成本：需要维护和更新 $d\times d$ 矩阵（复杂度通常 $O(d^2)$ 或更高），内存 $O(d^2)$。因此当 $d$ 很大时不现实。
* L-BFGS：保存最近 $m$ 个向量，时间和内存近似 $O(m d)$，适合大维度。

# 调试、验证与改进建议（实战派）

1. **梯度检验**：用中心差分验证解析梯度是否正确（很重要，尤其你改了 loss）。例如：

   ```python
   eps = 1e-6
   def numeric_grad(f, w):
       g = np.zeros_like(w)
       for i in range(len(w)):
           e = np.zeros_like(w); e[i]=1
           g[i] = (f(w+eps*e)-f(w-eps*e)) / (2*eps)
       return g
   # 比较 numeric_grad(obj, w) 与 grad_fn(w)
   ```
2. **估算最大特征值（当 d 大时）**：用幂迭代代替 `np.linalg.eigvalsh`。

   ```python
   def power_iter(A, niter=50):
       v = np.random.randn(A.shape[1]); v /= np.linalg.norm(v)
       for _ in range(niter):
           v = A.dot(v); v /= np.linalg.norm(v)
       lb = v.dot(A.dot(v))
       return lb
   ```
3. **实验记录**：把每次运行的参数（n,d,cond,sigma,lr,seed）和结果写 CSV，便于后续统计和画图。
4. **学习率策略**：对非二次/实际问题，固定 lr 不够稳健，可用 backtracking line search、Armijo、或者带动量 / 自适应方法（Adam）。
5. **预处理**：标准化 X 的列（零均值、单位方差）通常能显著改善 GD 收敛；但如果你想研究 condition number 的影响，就不要做标准化。
6. **替代方法**：对线性二次问题，`np.linalg.solve(X.T@X, X.T@y)`（闭式解）在 d 小并且 X^T X 条件数可接受时最快（复杂度 O(d^3)）。共轭梯度（CG）也是直接针对二次问题的好方法，它在稀疏或大维度时表现优异。

# 常见坑（要注意）

* `lr` 太大会发散：一定用 `1/L` 估计上界或做 line-search。
* 计算全部特征值用于 L 时，`d` 很大会很慢／内存高。
* 记录频率太高会显著影响耗时统计（将 I/O 时间与算法时间混淆）。
* BFGS 对于噪声很高的目标（stochastic objective）不适用——它假设目标和梯度是相对准确的（用于确定性的目标很好）。

---

如果你愿意，我可以下一步：

* 把这段代码按注释逐行输出（加解释注释）；
* 或者**把代码实际跑一次**（在 notebook 中）并把 loss-vs-time、参数误差的图发给你；
  你要哪一个就直接回复 “跑一次” 或 “加注释版”，我马上给你对应的内容。

--- 

## 实验结果
- V1
```shell
L (Lipschitz): 0.0020000000000000018
BFGS success: False nit: 500 time: 0.10105513700545998 final loss: 0.004604025970992544
GD final loss: 0.0046034566746033555 param_err: 3.9953010832936897
BFGS final loss: 0.004604025970992544 param_err: 3.988922119130701
```
![20250921100756](https://cdn.jsdelivr.net/gh/jagger235711/coooool@main/img/20250921100756.png)
![20250921101354](https://cdn.jsdelivr.net/gh/jagger235711/coooool@main/img/20250921101354.png)
![20250921100846](https://cdn.jsdelivr.net/gh/jagger235711/coooool@main/img/20250921100846.png)

- 不足
    - loss下降太快
    - 对比不显著
- 问题及原因分析
    - 为什么BFGS直接就是一条直线？
    - 为什么loss下降