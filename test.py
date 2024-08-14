import torch
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for i in range(100):
    time.sleep(2)
    x = torch.rand(1000, 1000, device=device)
    y = torch.rand(1000, 1000, device=device)
    z = x + y  # Perform an addition to invoke CUDA operations
    print(z)   # Trigger computation and synchronize