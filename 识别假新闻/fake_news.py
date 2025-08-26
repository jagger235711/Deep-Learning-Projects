# %% [markdown]
# ## Dataset Information
# 
# Develop a Deep learning program to identify when an article might be fake news.
# 
# 
# ### Attributes
# - id: unique id for a news article
# - title: the title of a news article
# - author: author of the news article
# - text: the text of the article; could be incomplete
# - label: a label that marks the article as potentially unreliable
#     - 1: unreliable
#     - 0: reliable

# %% [markdown]
# ## Import Modules

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
import warnings
import os
%matplotlib inline

warnings.filterwarnings('ignore')

# nltk.download("stopwords")
# nltk.download('punkt_tab')  # 仅首次需要


# %% [markdown]
# ## Loading the Dataset

# %%
data_root = './fake-news/'
df = pd.read_csv(data_root + 'train.csv')
df.head()

# %%
df['title'][0]

# %%
df['text'][0]

# %%
df.info()

# %% [markdown]
# ## Data Proprocessing

# %%
# drop unnecessary columns
df = df.drop(columns=['id', 'title', 'author'], axis=1)

# %%
# drop null values
df = df.dropna(axis=0)

# %%
len(df)

# %%
# remove special characters and punctuations

# %%
df['clean_news'] = df['text'].str.lower()
df['clean_news']

# %%
df['clean_news'] = df['clean_news'].str.replace('[^A-Za-z0-9\s]', '')
df['clean_news'] = df['clean_news'].str.replace('\n', '')
df['clean_news'] = df['clean_news'].str.replace('\s+', ' ')
df['clean_news']

# %%
# remove stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['clean_news'] = df['clean_news'].apply(lambda x: " ".join([word for word in x.split() if word not in stop]))
df.head()

# %% [markdown]
# ## Exploratory Data Analysis

# %%
# visualize the frequent words
all_words = " ".join([sentence for sentence in df['clean_news']])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15, 9))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
# visualize the frequent words for genuine news
all_words = " ".join([sentence for sentence in df['clean_news'][df['label']==0]])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15, 9))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
# visualize the frequent words for fake news
all_words = " ".join([sentence for sentence in df['clean_news'][df['label']==1]])

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph
plt.figure(figsize=(15, 9))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# ## Create Word Embeddings

# %%
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
import multiprocessing as mp
from tqdm import tqdm  # 进度条，需安装：pip install tqdm
from collections import defaultdict

# %%
df['clean_news']

# %%
# 选择分词函数（根据需求切换）
tokenize_func = word_tokenize  # 稍慢但更准确


# --------------------------
# 2. 批量分词（多进程加速）
# --------------------------
def batch_tokenize(texts, num_workers=mp.cpu_count() // 2):
    """多进程批量分词，利用CPU多核加速"""
    with mp.Pool(num_workers) as pool:
        # 用tqdm显示进度
        tokenized = list(
            tqdm(pool.imap(tokenize_func, texts), total=len(texts), desc="分词中")
        )
    return tokenized


# --------------------------
# 3. 快速构建词表（过滤低频词）
# --------------------------
def build_vocab_fast(tokenized_texts, min_freq=2):
    """
    快速构建词表：
    - 过滤低频词（出现次数 < min_freq 的词视为未知词）
    - 用defaultdict统计词频，减少内存开销
    """
    word_counts = defaultdict(int)
    # 统计词频（单进程足够快，多进程反而有开销）
    for tokens in tqdm(tokenized_texts, desc="统计词频"):
        for token in tokens:
            word_counts[token] += 1

    # 构建词表：特殊符号 + 高频词
    special_tokens = ["<unk>", "<pad>"]
    # 过滤低频词，按词频排序（可选，不排序更省时间）
    high_freq_words = [word for word, cnt in word_counts.items() if cnt >= min_freq]
    vocab = special_tokens + high_freq_words
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word_to_idx


# --------------------------
# 4. 主流程：快速预处理
# --------------------------
def preprocess_texts(df, text_col="clean_news", maxlen=500, min_freq=2):
    # 提取文本列表
    texts = df[text_col].tolist()

    # 1. 批量分词
    tokenized_texts = batch_tokenize(texts)

    # 2. 构建词表
    vocab, word_to_idx = build_vocab_fast(tokenized_texts, min_freq=min_freq)
    unk_idx = word_to_idx["<unk>"]
    pad_idx = word_to_idx["<pad>"]
    vocab_size = len(vocab)

    # 3. 文本转索引（批量处理）
    sequences = []
    for tokens in tqdm(tokenized_texts, desc="转索引"):
        # 截断过长序列（提前处理，减少后续填充开销）
        if len(tokens) > maxlen:
            tokens = tokens[:maxlen]
        # 转索引（未知词用<unk>）
        seq = [word_to_idx.get(token, unk_idx) for token in tokens]
        sequences.append(torch.tensor(seq, dtype=torch.long))

    # 4. 填充序列（PyTorch原生函数，高效）
    padded_seq = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)

    return padded_seq, vocab, word_to_idx, vocab_size


padded, vocab, word2idx, vocab_size = preprocess_texts(df, maxlen=500)

# %%
# 1. 加载预训练词向量（和原代码逻辑一致）
embedding_index = {}
with open("glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

# 2. 构建嵌入矩阵（PyTorch 版）
embedding_dim = 100  # GloVe 100d
embedding_matrix = torch.zeros((vocab_size, embedding_dim))  # PyTorch 张量

for word, i in word2idx.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # 转换为 PyTorch 张量
        embedding_matrix[i] = torch.tensor(embedding_vector)

# 3. 初始化 Embedding 层（关键！）
embedding_layer = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=word2idx["<pad>"],  # 假设 <pad> 是填充符
)
# 将预训练矩阵加载到嵌入层
embedding_layer.weight.data.copy_(embedding_matrix)
# （可选）冻结嵌入层，不参与训练
embedding_layer.weight.requires_grad = True

# %% [markdown]
# ## Input Split

# %%
len(padded[1])

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(padded, df['label'], test_size=0.20, random_state=42, stratify=df['label'])

# %%
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts  # 已经padding处理的文本序列
        self.labels = labels  # 对应的标签

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 将数据转换为PyTorch张量
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        label = torch.tensor(
            self.labels.iloc[idx], dtype=torch.float32
        )  # 假设是二分类，用float32
        return text, label


# 分割数据并创建数据加载器
def create_data_loaders(
    padded_texts, labels, test_size=0.2, random_state=42, batch_size=32
):
    # 先使用sklearn分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        padded_texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # 创建数据集实例
    train_dataset = TextDataset(x_train, y_train)
    test_dataset = TextDataset(x_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱顺序
        num_workers=2,  # 多进程加载数据，根据CPU核心数调整
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=2,
    )

    return train_loader, test_loader


# 使用示例
train_loader, test_loader = create_data_loaders(
    padded_texts=padded,
    labels=df['label'],
    batch_size=32
)

# %%
x_train.shape, x_test.shape
train_loader,test_loader

# %% [markdown]
# ## Model Training

# %%
# from keras.layers import LSTM, Dropout, Dense, Embedding
# from keras import Sequential


# model = Sequential([
#     Embedding(vocab_size+1, 100, weights=[embedding_matrix], trainable=False),
#     Dropout(0.2),
#     LSTM(128, return_sequences=True),
#     LSTM(128),
#     Dropout(0.2),
#     Dense(512),
#     Dropout(0.2),
#     Dense(256),
#     Dense(1, activation='sigmoid')
# ])
def get_net(vocab_size, embedding_matrix, device="cuda"):
    """定义优化的LSTM模型结构"""
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=True,
                padding_idx=word2idx["<pad>"]
            )
            self.dropout1 = nn.Dropout(0.3)
            self.lstm = nn.LSTM(100, 256, bidirectional=True, batch_first=True, num_layers=2)
            self.dropout2 = nn.Dropout(0.5)
            self.fc = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
            self.num_classes = 2  # 关键：添加类别数属性
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.dropout1(x)
            x, _ = self.lstm(x)  # 直接使用LSTM输出的序列
            x = self.dropout2(x[:, -1, :])  # 取最后一个时间步输出
            return self.fc(x)
    
    return Net().to(device)

    # 确保嵌入矩阵是张量类型
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    # 定义模型
    model = nn.Sequential(
        # 嵌入层：vocab_size+1个词，每个词嵌入为100维向量
        nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,  # 不训练嵌入层参数
            padding_idx=word2idx["<pad>"],  # 可选：指定填充索引
        ),
        nn.Dropout(0.2),
        # LSTM层：输入特征100维，隐藏层128维，返回全部时间步输出
        nn.LSTM(
            input_size=100,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        ),
        LSTMOutputExtractor(),
        # nn.Dropout(0.2),
        # 线性层：双向LSTM输出是256维(128*2)，映射到2维
        nn.Linear(256, 2),
    )

    # 给模型添加num_classes属性，方便后续获取
    model.num_classes = 2  # 关键：添加类别数属性
    # 将模型移动到指定设备
    model = model.to(device)
    return model

# %%
get_net(vocab_size=vocab_size, embedding_matrix=embedding_matrix)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from tqdm import tqdm
import torchmetrics
import os
from typing import Optional, Tuple, Dict


def train(
    net: nn.Module,
    train_iter: torch.utils.data.DataLoader,
    valid_iter: torch.utils.data.DataLoader,
    num_epochs: int,
    lr: float,
    wd: float,
    devices: Tuple[torch.device],
    loss_fn: Optional[nn.Module] = None,
    model_path: str = "best_model.pth",
    patience: int = 10,  # 早停耐心值
    save_best_only: bool = True,  # 只保存最好的模型
) -> Tuple[nn.Module, Dict]:
    """
    训练模型并返回训练好的模型和训练历史

    Args:
        net: 要训练的模型
        train_iter: 训练数据加载器
        valid_iter: 验证数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        wd: 权重衰减
        devices: 用于训练的设备
        loss_fn: 损失函数，如果为None则使用CrossEntropyLoss
        model_path: 模型保存路径
        patience: 早停机制的耐心值，多少轮没有提升就停止
        save_best_only: 是否只保存性能最好的模型

    Returns:
        训练好的模型和训练历史字典
    """
    # 初始化损失函数
    if loss_fn is None:
        # 使用更适合二分类的损失函数
        loss_fn = nn.BCEWithLogitsLoss()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    best_acc = 0.0
    counter = 0  # 早停计数器
    history = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    # 数据并行处理
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    # 优化器 - 只优化需要梯度的参数
    trainer = optim.Adam(
        (param for param in net.parameters() if param.requires_grad),
        lr=lr,
        weight_decay=wd,
    )

    # 学习率调度器
    scheduler = CosineAnnealingLR(trainer, num_epochs)

    # 初始化评估指标
    train_metrics = {
        "loss": torchmetrics.MeanMetric().to(devices[0]),
            "acc": torchmetrics.Accuracy(
                task="binary",
                threshold=0.5
            ).to(devices[0]),
    }

    valid_metrics = {
        "loss": torchmetrics.MeanMetric().to(devices[0]),
            "acc": torchmetrics.Accuracy(
                task="binary",
                threshold=0.5
            ).to(devices[0]),
    }

    logging.info(f"开始训练，使用设备: {devices}")
    logging.info(f"训练轮数: {num_epochs}, 初始学习率: {lr}, 权重衰减: {wd}")

    for epoch in range(num_epochs):
        # 训练阶段
        net.train()
        # 重置训练指标
        for metric in train_metrics.values():
            metric.reset()

        train_pbar = tqdm(train_iter, desc=f"Epoch {epoch + 1}/{num_epochs} [训练]")
        for X, y in train_pbar:
            X, y = X.to(devices[0], non_blocking=True), y.to(
                devices[0], non_blocking=True
            )

            # 前向传播
            y_hat = net(X)
            # 确保标签维度匹配
            y = y.view(-1, 1).float()  # 转换为[batch_size, 1]
            loss = loss_fn(y_hat, y)

            # 反向传播和优化
            trainer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            loss.backward()
            trainer.step()

            # 更新指标
            train_metrics["loss"].update(loss.detach())
            train_metrics["acc"].update(y_hat.detach(), y)

            # 更新进度条
            train_pbar.set_postfix(
                {
                    "loss": f"{train_metrics['loss'].compute().item():.4f}",
                    "acc": f"{train_metrics['acc'].compute().item():.4f}",
                }
            )

        # 计算训练集指标
        train_loss = train_metrics["loss"].compute().item()
        train_acc = train_metrics["acc"].compute().item()

        # 记录训练历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # 打印并记录训练结果
        log_msg = f"Epoch {epoch + 1}/{num_epochs} | 训练集 | 损失: {train_loss:.4f} | 准确率: {train_acc:.4f}"
        print(log_msg)
        logging.info(log_msg)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        logging.debug(f"当前学习率: {current_lr:.6f}")

        # 验证阶段
        net.eval()
        # 重置验证指标
        for metric in valid_metrics.values():
            metric.reset()

        with torch.no_grad():
            valid_pbar = tqdm(valid_iter, desc=f"Epoch {epoch + 1}/{num_epochs} [验证]")
            for X, y in valid_pbar:
                X, y = X.to(devices[0], non_blocking=True), y.to(
                    devices[0], non_blocking=True
                )
                y_hat = net(X)
                # 调整标签维度匹配输出
                y = y.view(-1, 1).float()  # 确保标签形状为[batch_size, 1]
                loss = loss_fn(y_hat, y)
                
                # 更新指标，添加sigmoid处理
                preds = torch.sigmoid(y_hat)
                valid_metrics["acc"].update((preds > 0.5).float(), y)
                valid_metrics["loss"].update(loss)
                
                # 更新进度条
                valid_pbar.set_postfix(
                    {
                        "loss": f"{valid_metrics['loss'].compute().item():.4f}",
                        "acc": f"{valid_metrics['acc'].compute().item():.4f}",
                    }
                )


        # 计算验证集指标
        valid_loss = valid_metrics["loss"].compute().item()
        valid_acc = valid_metrics["acc"].compute().item()

        # 记录验证历史
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        # 打印并记录验证结果
        log_msg = f"Epoch {epoch + 1}/{num_epochs} | 验证集 | 损失: {valid_loss:.4f} | 准确率: {valid_acc:.4f}"
        print(log_msg)
        logging.info(log_msg)

        # 保存模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            counter = 0  # 重置早停计数器
            # 保存模型
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": trainer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc": best_acc,
                },
                model_path,
            )
            logging.info(f"保存新的最佳模型，准确率: {best_acc:.4f}")
        else:
            counter += 1
            logging.info(f"早停计数器: {counter}/{patience}")
            if counter >= patience:
                logging.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break

        # 如果不是只保存最好模型，每轮都保存
        if not save_best_only:
            epoch_model_path = (
                f"{os.path.splitext(model_path)[0]}_epoch_{epoch + 1}.pth"
            )
            torch.save(net.state_dict(), epoch_model_path)

    # 加载最佳模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"训练结束，加载最佳模型，准确率: {checkpoint['best_acc']:.4f}")

    return net, history

# %%
history = train(
    net=get_net(vocab_size=vocab_size, embedding_matrix=embedding_matrix),
    train_iter=train_loader,
    valid_iter=test_loader,
    num_epochs=10,
    lr=0.01,
    wd=1e-3,
    devices=[
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    ],
    model_path="best_model.pth",
    patience=5,  # 早停耐心值
    save_best_only=True,  # 只保存最好的模型
)

# %%
# visualize the results
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['Train', 'Test'])
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['Train', 'Test'])
plt.show()
