import torch

print(torch.cuda.is_available())

print(torch.nn.functional.relu(torch.randn(2, 3).to("cuda")))