import torch
import sys



print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print("-" * 30)

if torch.cuda.is_available():
    print("GPU (CUDA/ROCm) 可用！")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        try:
            print(f"  计算能力 (Compute Capability): {torch.cuda.get_device_capability(i)}")
        except Exception as cap_e:
            print(f"  无法获取计算能力: {cap_e}")
        try:
            print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        except Exception as mem_e:
             print(f"  无法获取显存信息: {mem_e}")
else:
    print("GPU (CUDA/ROCm) 不可用。请检查：")
    print("1. 是否安装了支持 GPU (CUDA 或 ROCm) 的 PyTorch 版本？")
    print("   (对于 AMD GPU, 需要 ROCm 版本的 PyTorch)")
    print("2. GPU 驱动是否正确安装？")
    print("   (对于 AMD GPU, 需要 AMD ROCm 驱动)")
    print("3. ROCm 版本是否与 PyTorch 版本兼容？")
    print("4. 你的 AMD GPU 型号是否被 ROCm 和 PyTorch 支持？")

print("-" * 30)
try:
    if torch.cuda.is_available():
        tensor = torch.tensor([1.0, 2.0]).cuda()
        print("成功将张量移动到 GPU:", tensor)
        print("当前设备:", tensor.device)
    else:
        print("无法将张量移动到 GPU，因为 GPU (CUDA/ROCm) 不可用。")
except Exception as e:
    print(f"尝试将张量移动到 GPU 时出错: {e}")
    