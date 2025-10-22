import torch
from torch import nn

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    print(f"Available CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# 测试GPU数据转移功能
def test_gpu_transfer():
    print("\n=== 测试GPU数据转移 ===")
    
    # 测试不同设备选择方式
    devices = [
        torch.device('cuda'),  # 默认GPU
        torch.device('cuda:0'),  # 指定GPU 0
        try_gpu(),  # 使用d2l的try_gpu()
        try_gpu(0)  # 明确指定索引0
    ]
    
    test_cases = [
        torch.randn(3, 3),  # 普通张量
        torch.zeros(1000, 1000)  # 大张量
    ]
    
    for i, device in enumerate(devices):
        print(f"\n测试设备选择方式 {i+1}: {device}")
        for j, data in enumerate(test_cases):
            print(f"\n测试案例 {j+1}: 类型={type(data)}, 形状={data.shape}")
            try:
                # 转移到GPU
                gpu_data = data.to(device)
                print(f"转移到GPU成功: {gpu_data.device}")
                
                # 转移回CPU
                cpu_data_back = gpu_data.cpu()
                print(f"转回CPU成功: {cpu_data_back.device}")
                
                # 验证数据一致性
                assert torch.allclose(data, cpu_data_back), "数据转移后不一致"
                print("数据验证通过")

                # 测试大张量的合并操作
                if data.shape[0] > 900:
                    print("大张量合并操作测试")
                    try:
                        merged = torch.cat([gpu_data, gpu_data], dim=0)
                        print(f"合并成功: {merged.device}")
                    except Exception as e:
                        print(f"合并失败: {str(e)}")
            except Exception as e:
                print(f"测试失败: {str(e)}")

# 执行测试
test_gpu_transfer()

# 其他测试代码
torch.cuda.device_count()
try_gpu(), try_gpu(10), try_all_gpus()

x = torch.tensor([1, 2, 3])
print("------------------")
print(x.device)
print("------------------")

X = torch.ones(2, 3, device=try_gpu())
print("------------------")
print(X.device)
print("------------------")

import os
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm version: {torch.version.hip}")
print(f"GPU count: {torch.cuda.device_count()}")

# 创建张量测试
try:
    x = torch.ones(2, 3, device="cuda")
    print(f"Tensor device: {x.device}")
    print("Success! GPU is working.")
except RuntimeError as e:
    print(f"Error: {e}")
    print("GPU information:")
    os.system("rocminfo | grep -i gfx")

X = torch.ones(2, 3, device=try_gpu())
Y = torch.rand(2, 3, device=try_gpu())
Z = X.cuda(0)
print(X)
print(Z)
print(Y + Z)
print(Z.cuda(0) is Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

print(net(X))
print(net[0].weight.data.device)
