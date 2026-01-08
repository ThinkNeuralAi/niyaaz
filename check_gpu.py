"""Check GPU availability for the application"""
import sys
try:
    import torch
    print("=" * 60)
    print("GPU/CUDA Status Check")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
            print(f"  Allocated Memory: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
            print(f"  Free Memory: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / (1024**3):.2f} GB")
    else:
        print("\n⚠️  CUDA is NOT available")
        print("Possible reasons:")
        print("  1. No NVIDIA GPU installed")
        print("  2. CUDA Toolkit not installed")
        print("  3. PyTorch CPU-only version installed (not CUDA version)")
        print("\nTo enable GPU:")
        print("  1. Install CUDA Toolkit from NVIDIA")
        print("  2. Install PyTorch with CUDA support:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("=" * 60)
except ImportError:
    print("ERROR: PyTorch is not installed!")
    print("Install it with: pip install torch torchvision")
    sys.exit(1)



