import platform
import torch

print("=" * 50)
print("NORDIC AI CUP ENVIRONMENT")
print("=" * 50)

print("Machine:", platform.platform())
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("Device: CUDA")
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Device: Apple MPS")
else:
    print("Device: CPU")

print("=" * 50)
