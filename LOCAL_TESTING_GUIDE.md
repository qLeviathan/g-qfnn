# Local Windows Testing Guide

## Setup for Testing in Windows Environment

Follow these steps to test the fixed Field-Theoretic Language Model directly in your Windows environment:

### 1. Environment Setup (Using cmd or PowerShell)

```powershell
# Navigate to your project directory
cd C:\Users\casma\OneDrive\Desktop\physicsFirst\g-qfnn

# Create a virtual environment using Python 3.11
python -m venv venv_win

# Activate the virtual environment
.\venv_win\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Module Testing

Run individual module tests to verify fixes:

```powershell
# Test core modules first
python collapse.py
python perturbations.py
python data.py
python model.py

# Run the full test suite (this may take longer)
python test_all.py
```

### 3. Quick Training Test

Try a minimal training run to verify the model works:

```powershell
# Small model, minimal steps, small batch size
python main.py train --model-size small --dataset wikitext-2 --num-steps 50 --batch-size 1 --log-every 10
```

### 4. GPU Monitoring

While running training, monitor your GPU usage in a separate PowerShell window:

```powershell
# If you have nvidia-smi
while($true) { nvidia-smi; Start-Sleep -Seconds 5; Clear-Host }

# Alternative: Use Task Manager to monitor GPU usage
```

### 5. Text Generation

Test generating text from the model:

```powershell
python main.py generate "The quantum field" --temperature 0.8 --num-samples 1
```

### 6. Common Issues in Windows Environment

1. **Path Issues**: Windows uses backslashes (`\`) in paths, so some hardcoded paths might need adjustment.

2. **File Encoding**: Windows default encoding can still cause issues with some Unicode characters.

3. **CUDA Compatibility**: Ensure you have the correct CUDA version installed for your GPU.

4. **Memory Management**: Windows memory management differs from Linux; you may need to reduce batch sizes.

### 7. Memory Optimization

If you encounter memory issues, try these options:

```powershell
# Use smaller model
python main.py train --model-size small --batch-size 1 --seq-length 256

# Enable fp16 if supported
python main.py train --model-size small --fp16

# Reduce sequence length
python main.py train --model-size small --max-seq-len 256
```

### 8. Troubleshooting

If you encounter issues:

1. Verify CUDA is working:
   ```python
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
   ```

2. Check if there are any Windows-specific environment variables needed:
   ```powershell
   $env:PYTHONIOENCODING = "utf-8"
   ```

3. If HuggingFace downloads fail, try setting the cache directory:
   ```powershell
   $env:HF_HOME = "C:\Users\casma\.cache\huggingface"
   ```