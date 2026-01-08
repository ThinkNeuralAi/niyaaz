# Installing GPU Support for Sakshi.AI

## Current Status
- ✅ **GPU Available**: NVIDIA GeForce RTX 3050 (6GB VRAM)
- ✅ **CUDA Version**: 12.8
- ❌ **PyTorch CUDA**: Not installed (currently using CPU-only version)

## Installation Steps

### Step 1: Activate Virtual Environment

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**OR use Command Prompt (cmd):**
```cmd
venv\Scripts\activate.bat
```

### Step 2: Uninstall Current PyTorch (CPU-only)

```powershell
pip uninstall torch torchvision -y
```

### Step 3: Install PyTorch with CUDA 12.1 Support

For CUDA 12.1 (compatible with your CUDA 12.8):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Alternative:** If CUDA 12.1 doesn't work, try CUDA 11.8:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Verify Installation

Run this to check if CUDA is now available:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

You should see:
```
CUDA Available: True
CUDA Version: 12.1 (or 11.8)
GPU Name: NVIDIA GeForce RTX 3050 Laptop GPU
```

### Step 5: Restart Your Application

After installing CUDA-enabled PyTorch:
1. Stop your current application (Ctrl+C)
2. Restart it: `python app.py`
3. Check the logs - you should now see:
   - `YOLO model loaded successfully on cuda` (instead of `cpu`)
   - `gpu enabled: True` (in DeepSORT logs)
   - `Using GPU device. Free memory: X.X GB`

## Expected Performance Improvements

**Before (CPU):**
- Processing time: 1-2+ seconds per frame
- Multiple modules: 2-5 seconds per frame

**After (GPU):**
- Processing time: 0.1-0.5 seconds per frame
- Multiple modules: 0.3-1 second per frame

## Troubleshooting

### Issue: "CUDA out of memory"
- **Solution**: Reduce number of concurrent channels or use smaller batch sizes
- Your RTX 3050 has 6GB VRAM, which should handle multiple channels

### Issue: "CUDA version mismatch"
- **Solution**: Make sure PyTorch CUDA version matches your CUDA toolkit
- CUDA 12.1 PyTorch works with CUDA 12.8 (backward compatible)

### Issue: Still showing "cpu" after installation
- **Solution**: 
  1. Verify installation: `python -c "import torch; print(torch.cuda.is_available())"`
  2. If False, check CUDA drivers: `nvidia-smi`
  3. Reinstall PyTorch with correct CUDA version

## Quick Install Command (All-in-One)

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Uninstall CPU version
pip uninstall torch torchvision -y

# Install CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Notes

- **DeepSORT Tracker**: Will automatically use GPU if available (currently shows `gpu enabled: False` because PyTorch CUDA isn't installed)
- **YOLO Models**: Will automatically use GPU for inference
- **RTSP Decoding**: Still uses CPU (OpenCV), but model inference will use GPU
- **Memory**: Your 6GB VRAM should handle multiple models and channels efficiently



