import torch

# Детальная информация о CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(3, 3).to(device)
    print("Успешно создан тензор на GPU:", x.device)
else:
    print("CUDA не доступна для вычислений")