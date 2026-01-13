# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# %%
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
train = pd.read_csv(
    "/home/wwj/src/Deep-Learning-Projects/Predicting_Student_Test_Scores/data/train.csv"
)
test = pd.read_csv(
    "/home/wwj/src/Deep-Learning-Projects/Predicting_Student_Test_Scores/data/test.csv"
)

# %%
ID_COL = "id"
TARGET = "exam_score"

# %%
# 数据预处理 - 与LightGBM保持一致
X = train.drop(columns=[ID_COL, TARGET])
y = train[TARGET]
X_test = test.drop(columns=[ID_COL])

# %%
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# %%
print("Categorical: ", cat_cols)
print("Numerical: ", num_cols)

# %%
# 添加缺失值指示特征
for col in X.columns:
    X[col + "_is_null"] = X[col].isnull().astype(int)
    X_test[col + "_is_null"] = X_test[col].isnull().astype(int)

# %%
# 添加频率编码特征
for col in cat_cols:
    freq = X[col].value_counts()
    X[col + "_freq"] = X[col].map(freq)
    X_test[col + "_freq"] = X_test[col].map(freq)

# %%
# 删除原始分类变量
X = X.drop(columns=cat_cols)
X_test = X_test.drop(columns=cat_cols)

# %%
# 填充剩余缺失值
X = X.fillna(0)
X_test = X_test.fillna(0)

# %%
# 标准化数值特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# %%
# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y.values).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# %%
# 自定义ResNet模型适用于表格数据
class TabularResNet(nn.Module):
    def __init__(self, input_dim, num_classes=1):
        super(TabularResNet, self).__init__()
        self.input_dim = input_dim

        # 使用预训练的ResNet18作为特征提取器
        self.resnet = resnet18(pretrained=True)

        # 修改第一层以适应输入维度
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改全连接层
        self.resnet.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)  # 添加Dropout层

        # 添加适应层将表格数据转换为图像格式
        self.adapt_layer = nn.Sequential(
            nn.Linear(input_dim, 224 * 224),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加Dropout层
            nn.Unflatten(1, (1, 224, 224))
        )

    def forward(self, x):
        # 适应输入形状
        x = self.adapt_layer(x)
        x = self.resnet(x)
        x = self.dropout(x)  # 应用Dropout
        return x

# %%
# 初始化模型
input_dim = X_tensor.shape[1]
model = TabularResNet(input_dim).to(device)

# %%
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# %%
# 学习率warmup调度器
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            return self.base_lr * (self.current_epoch / self.warmup_epochs)
        return self.base_lr

# %%
# 早停机制
class EarlyStopping:
    def __init__(self, patience=200, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# %%
# KFold交叉验证
kf = KFold(n_splits=3, shuffle=True, random_state=42)

oof = np.zeros(len(X))
pred_test = np.zeros(len(X_test))

# %%
for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")

    # 创建数据加载器
    X_tr, X_val = X_tensor[trn_idx], X_tensor[val_idx]
    y_tr, y_val = y_tensor[trn_idx], y_tensor[val_idx]

    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化模型和优化器
    model = TabularResNet(input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-3)
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=5, base_lr=0.0005)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=100)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(10):
        model.train()
        train_loss = 0.0

        # 添加进度条
        train_loader_with_progress = tqdm(train_loader, desc=f"Fold {fold+1} - Training", leave=False)

        for batch_X, batch_y in train_loader_with_progress:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loader_with_progress.set_postfix(loss=loss.item())

        # 学习率调度
        if epoch < 5:  # warmup阶段
            warmup_scheduler.step()
        else:  # cosine衰减阶段
            cosine_scheduler.step()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')

    # 加载最佳模型
    model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))

    # 预测
    with torch.no_grad():
        model.eval()
        oof[val_idx] = model(X_val.to(device)).cpu().numpy().flatten()
        pred_test += model(X_test_tensor.to(device)).cpu().numpy().flatten() / 3

# %%
# 计算RMSE
rmse = mean_squared_error(y, oof, squared=False)
print("CV RMSE:", rmse)

# %%
# 生成提交文件
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: pred_test
})

submission.to_csv("submission_pytorch_resnet18.csv", index=False)

# %%
print("Training complete! Submission file generated.")
