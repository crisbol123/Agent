import torch

print("=== VERIFICACIÓN DE GPU ===\n")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"VRAM libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
    print("\n✅ GPU lista para usar")
else:
    print("\n❌ CUDA no disponible. Instala PyTorch con CUDA:")
    print("   pip uninstall torch")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu124")
