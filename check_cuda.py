import torch
import time

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("cuDNN:", torch.backends.cudnn.version())

x = torch.randn(4096, 4096, device=device)
torch.cuda.synchronize()

start = time.time()
y = x @ x
torch.cuda.synchronize()

print("Matrix multiply OK")
print("Time:", round(time.time() - start, 3), "s")
print("Allocated VRAM:", round(torch.cuda.memory_allocated() / 1024**3, 3), "GB")
print("Reserved VRAM:", round(torch.cuda.memory_reserved() / 1024**3, 3), "GB")