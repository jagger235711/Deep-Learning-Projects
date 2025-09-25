import numpy as np


class Perceptron:
    def __init__(self, input_dim=3, learning_rate=1.0):
        """
        初始化感知机
        :param input_dim: 输入维度（含偏置项I0，默认3维：I0, I1, I2）
        :param learning_rate: 学习率（默认1.0，与示例中权重更新幅度匹配）
        """
        # 初始化权重（参考文档中初始权重范围，随机生成[-0.5, 0.5)之间的数值）
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=input_dim)
        self.lr = learning_rate
        self.converged = False  # 收敛标志

    def activation(self, x):
        """
        激活函数（阶跃函数）：输入>=0输出1，否则输出0
        :param x: 加权和（I0*W0 + I1*W1 + I2*W2）
        :return: 感知机输出（0或1）
        """
        return 1 if x > 0 else 0

    def predict(self, inputs):
        """
        预测函数：计算输入的加权和并通过激活函数输出结果
        :param inputs: 输入向量（[I0, I1, I2]）
        :return: 预测结果（0或1）
        """
        weighted_sum = np.dot(inputs, self.weights)  # 计算加权和
        return self.activation(weighted_sum)

    def train_step(self, inputs, target):
        """
        单步训练：根据预测误差更新权重
        :param inputs: 输入向量（[I0, I1, I2]）
        :param target: 目标输出（T）
        :return: 权重是否更新（True=更新，False=未更新）
        """
        y_pred = self.predict(inputs)
        error = target - y_pred  # 计算误差

        # 若存在误差，更新权重：W_new = W_old + 学习率 * 误差 * 输入
        if error != 0:
            self.weights += self.lr * error * inputs
            return True
        return False

    def train(self, training_data, max_epochs=25):
        """
        完整训练过程：迭代训练直到收敛或达到最大轮次
        :param training_data: 训练数据集，格式为[(inputs1, target1), (inputs2, target2), ...]
        :param max_epochs: 最大训练轮次（参考文档中20轮后判断不可分，设为25轮留有余地）
        :return: 训练日志（每轮的详细信息）
        """
        log = []  # 存储训练日志：[训练回数, I0, I1, I2, W0, W1, W2, Out, T]
        epoch = 0

        while epoch < max_epochs and not self.converged:
            weights_updated = False  # 标记本轮是否更新过权重

            # 遍历所有训练样本（每组4个样本，对应逻辑运算的4种输入组合）
            for inputs, target in training_data:
                epoch += 1
                y_pred = self.predict(inputs)

                # 记录当前轮次信息
                log_entry = [
                    epoch,
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    round(self.weights[0], 1),
                    round(self.weights[1], 1),
                    round(self.weights[2], 1),
                    y_pred,
                    target,
                ]
                log.append(log_entry)

                # 单步训练并检查权重是否更新
                if self.train_step(inputs, target):
                    weights_updated = True

            # 若本轮未更新权重，说明所有样本预测正确，模型收敛
            if not weights_updated:
                self.converged = True
                # 在收敛轮次的日志末尾添加标记
                log[-1].append("ここで収束した")
                break

        # 若训练结束后仍未收敛，标记为线性不可分
        if not self.converged:
            log[-1].append("線形分離不可能、収束できない")

        return log

    def print_log(self, log, logic_name):
        """
        打印训练日志（格式化输出，与文档中表格格式一致）
        :param log: 训练日志（train方法返回的列表）
        :param logic_name: 逻辑运算名称（如“論理積”“論理和”“排他的論理和”）
        """
        print(f"\n===== {logic_name} 感知機訓練ログ =====")
        # 打印表头
        header = ["学習回数", "I0", "I1", "I2", "W0", "W1", "W2", "Out", "T", "備考"]
        print(
            f"{header[0]:<6} {header[1]:<2} {header[2]:<2} {header[3]:<2} "
            f"{header[4]:<4} {header[5]:<4} {header[6]:<4} {header[7]:<3} "
            f"{header[8]:<2} {header[9]:<10}"
        )
        print("-" * 60)

        # 打印每轮日志
        for entry in log:
            # 补充空字符串（若无备考信息）
            while len(entry) < 10:
                entry.append("")
            print(
                f"{entry[0]:<6} {entry[1]:<2} {entry[2]:<2} {entry[3]:<2} "
                f"{entry[4]:<4} {entry[5]:<4} {entry[6]:<4} {entry[7]:<3} "
                f"{entry[8]:<2} {entry[9]:<10}"
            )


# ---------------------- 定义训练数据集（三种逻辑运算） ----------------------
# 通用输入：I0恒为1（偏置项），I1和I2为逻辑输入（0或1），共4组样本
inputs_list = [
    np.array([1, 0, 0]),  # I0=1, I1=0, I2=0
    np.array([1, 0, 1]),  # I0=1, I1=0, I2=1
    np.array([1, 1, 0]),  # I0=1, I1=1, I2=0
    np.array([1, 1, 1]),  # I0=1, I1=1, I2=1
]

# 1. 論理積（AND）：所有输入为1时输出1，否则为0（参考文档示例）
and_targets = [0, 0, 0, 1]
and_training_data = list(zip(inputs_list, and_targets))

# 2. 論理和（OR）：任意输入为1时输出1，否则为0
or_targets = [0, 1, 1, 1]
or_training_data = list(zip(inputs_list, or_targets))

# 3. 排他的論理和（XOR）：输入不同时输出1，输入相同时输出0（线性不可分）
xor_targets = [0, 1, 1, 0]
xor_training_data = list(zip(inputs_list, xor_targets))


# ---------------------- 执行训练并打印结果 ----------------------
if __name__ == "__main__":
    # 1. 训练并打印“論理積”
    and_perceptron = Perceptron()
    and_perceptron.weights = np.array([-0.1, 0.1, 0.3])
    and_log = and_perceptron.train(and_training_data)
    and_perceptron.print_log(and_log, "論理積")

    # 2. 训练并打印“論理和”
    or_perceptron = Perceptron()
    or_perceptron.weights = np.array([-0.2, 0.1, 0.2])
    or_log = or_perceptron.train(or_training_data)
    or_perceptron.print_log(or_log, "論理和")

    # 3. 训练并打印“排他的論理和”
    xor_perceptron = Perceptron()
    xor_perceptron.weights = np.array([-0.1, 0.2, 0.3])
    xor_log = xor_perceptron.train(xor_training_data)
    xor_perceptron.print_log(xor_log, "排他的論理和")
