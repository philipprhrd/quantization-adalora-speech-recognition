import torch
import torchao

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version as seen by PyTorch:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("torchao version:", getattr(torchao, "__version__", "unknown"))