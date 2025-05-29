"""
find_batch_size.py - Find optimal batch size for your GPU
"""

import torch
import gc
from model import create_small_model, create_base_model, create_large_model

def test_config(model_fn, model_name, batch_size, seq_length, device='cuda'):
    """Test if batch/seq config fits in memory"""
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create model
        model = model_fn("golden").to(device)
        model.eval()
        
        # Test forward pass
        input_ids = torch.randint(0, 50257, (batch_size, seq_length)).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, crystal_update=False)
        
        # Get memory usage
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
        
        # Cleanup
        del model, input_ids, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return True, memory_mb
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, 0
        raise e

def find_optimal_batch():
    """Find optimal batch sizes for each model"""
    configs = [
        ("small", create_small_model),
        ("base", create_base_model),
        ("large", create_large_model)
    ]
    
    seq_lengths = [128, 256, 512, 1024]
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    print("Finding optimal batch sizes for 4090...")
    print("=" * 60)
    
    results = {}
    
    for model_name, model_fn in configs:
        print(f"\n{model_name.upper()} MODEL:")
        results[model_name] = {}
        
        for seq_len in seq_lengths:
            max_batch = 0
            max_memory = 0
            
            for batch_size in batch_sizes:
                fits, memory = test_config(model_fn, model_name, batch_size, seq_len)
                
                if fits:
                    max_batch = batch_size
                    max_memory = memory
                    print(f"  seq={seq_len:4d}, batch={batch_size:2d}: ✓ {memory:6.1f} MB")
                else:
                    print(f"  seq={seq_len:4d}, batch={batch_size:2d}: ✗ OOM")
                    break
            
            results[model_name][seq_len] = (max_batch, max_memory)
    
    # Print recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIGURATIONS (80% safety margin):")
    print("=" * 60)
    
    for model_name in results:
        print(f"\n{model_name.upper()}:")
        for seq_len, (batch, memory) in results[model_name].items():
            safe_batch = int(batch * 0.8)  # 80% for safety
            if safe_batch > 0:
                print(f"  python main.py train --model-size {model_name} "
                      f"--batch-size {safe_batch} --seq-length {seq_len}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
        find_optimal_batch()
    else:
        print("No GPU found!")