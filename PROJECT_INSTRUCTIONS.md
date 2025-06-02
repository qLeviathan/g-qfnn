# 🚀 Field-Theoretic Language Model - Project Instructions

## 📋 Prerequisites

- **Python 3.11** (Required)
- **WSL2** (for Windows users)
- **CUDA-capable GPU** (recommended: RTX 4090 with 24GB VRAM)
- **Git** for version control

## 🔧 Setup Instructions

### 1. Environment Setup (WSL)

```bash
# Update WSL and install Python 3.11
sudo apt update && sudo apt upgrade -y
sudo apt install python3.11 python3.11-pip python3.11-venv -y

# Verify installation
python3.11 --version
```

### 2. Project Setup

```bash
# Navigate to project directory in WSL
cd /mnt/c/Users/casma/OneDrive/Desktop/physicsFirst/g-qfnn

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. GPU Setup (CUDA)

```bash
# Install CUDA toolkit for WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-0 -y

# Verify CUDA installation
nvcc --version
python3.11 -c "import torch; print(torch.cuda.is_available())"
```

## 🏃‍♂️ Quick Start

### 1. Validate Installation

```bash
# Test core components
cd /mnt/c/Users/casma/OneDrive/Desktop/physicsFirst/g-qfnn
source venv/bin/activate

python3.11 core.py          # ✓ Test embeddings and field evolution
python3.11 model.py         # ✓ Test full model architecture
python3.11 trainer.py       # ✓ Test training loop
python3.11 test_all.py      # ✓ Run comprehensive tests
```

### 2. Basic Training

```bash
# Small model for testing (recommended first run)
python3.11 main.py train --model-size small --dataset wikitext-2 --num-steps 1000

# Monitor GPU usage
nvidia-smi
```

### 3. Generation Test

```bash
# Generate text with trained model
python3.11 main.py generate "The quantum field" --perturbation levy --temperature 0.9
```

## 📚 Detailed Usage

### Training Commands

**Small Model (50M params, ~2GB VRAM):**
```bash
python3.11 main.py train \
    --model-size small \
    --dataset wikitext-2 \
    --num-steps 10000 \
    --batch-size 4 \
    --learning-rate 0.618
```

**Base Model (350M params, ~6GB VRAM):**
```bash
python3.11 main.py train \
    --model-size base \
    --dataset wikitext-103 \
    --num-steps 25000 \
    --batch-size 2 \
    --perturbation-schedule '{"0": "beta", "10000": "adaptive"}'
```

**Large Model (750M params, ~12GB VRAM):**
```bash
python3.11 main.py train \
    --model-size large \
    --dataset c4 \
    --num-steps 50000 \
    --batch-size 1 \
    --perturbation-schedule '{"0": "beta", "10000": "adaptive", "30000": "levy"}'
```

### Advanced Training Options

```bash
# Custom perturbation mixing
python3.11 main.py train \
    --model-size base \
    --dataset c4 \
    --perturbation levy \
    --alpha 1.618 \
    --coherence-threshold 0.91 \
    --crystal-lr 0.618

# Resume from checkpoint
python3.11 main.py train \
    --checkpoint outputs/checkpoint_10000.pt \
    --num-steps 20000
```

### Generation & Analysis

```bash
# Interactive generation
python3.11 main.py generate \
    "Once upon a time in the quantum realm" \
    --max-length 200 \
    --temperature 0.8 \
    --perturbation adaptive

# Batch generation
python3.11 main.py generate \
    --prompts-file prompts.txt \
    --output-file generated.txt \
    --checkpoint outputs/best_model.pt

# Field dynamics analysis
python3.11 main.py analyze \
    "Hello world" \
    --visualize \
    --save-path outputs/field_evolution.png
```

### Ablation Studies

```bash
# Compare embedding types
python3.11 main.py ablation embedding_comparison \
    --embeddings golden,log_phase,standard \
    --datasets wikitext-2,wikitext-103 \
    --steps 5000

# Compare perturbation strategies
python3.11 main.py ablation perturbation_comparison \
    --perturbations levy,beta,adaptive \
    --model-size small \
    --steps 10000

# Performance profiling
python3.11 main.py profile \
    --batch-sizes 1,2,4,8 \
    --seq-lengths 128,256,512 \
    --model-sizes small,base
```

## 🧪 Testing

### Run All Tests

```bash
# Comprehensive test suite
python3.11 test_all.py

# Individual test modules
python3.11 tests/test1_manifold.py
python3.11 tests/test2_dynamics.py
python3.11 tests/test3_embeddings.py
python3.11 tests/test4_quantum.py
python3.11 tests/test5_fibonacci.py
python3.11 tests/test6_consciousness.py
```

### Performance Tests

```bash
# Memory usage profiling
python3.11 -m memory_profiler main.py train --model-size small --num-steps 100

# Speed benchmarking
python3.11 -m cProfile -o profile.stats main.py train --model-size small --num-steps 100
```

## 📊 Monitoring & Outputs

### Training Outputs

- **Checkpoints**: `outputs/checkpoint_*.pt`
- **Logs**: `outputs/training.log`
- **Visualizations**: `outputs/*.png`
- **Metrics**: `outputs/metrics.json`

### Real-time Monitoring

```bash
# Watch training progress
tail -f outputs/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

## 🎮 Interactive Development

### Jupyter Notebook

```bash
# Launch notebook server
jupyter lab field_theoretic_lm.ipynb

# Or use VSCode notebook interface
code field_theoretic_lm.ipynb
```

### Python REPL

```python
# Interactive experimentation
python3.11
>>> from core import GoldenFieldEmbedding
>>> from model import FieldTheoreticLM
>>> model = FieldTheoreticLM()
>>> # Experiment with components
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size or model size
python3.11 main.py train --model-size small --batch-size 1
```

**WSL Path Issues:**
```bash
# Ensure correct WSL path
pwd  # Should show /mnt/c/Users/casma/OneDrive/Desktop/physicsFirst/g-qfnn
```

**Missing Dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Permission Errors:**
```bash
# Fix WSL permissions
sudo chmod -R 755 /mnt/c/Users/casma/OneDrive/Desktop/physicsFirst/g-qfnn
```

### Debug Mode

```bash
# Enable verbose logging
python3.11 main.py train --model-size small --debug --verbose

# Check model architecture
python3.11 -c "from model import FieldTheoreticLM; print(FieldTheoreticLM().summary())"
```

## 📈 Performance Expectations

### RTX 4090 Benchmarks

| Model Size | Memory Usage | Speed (tokens/sec) | Training Time (10k steps) |
|------------|--------------|-------------------|---------------------------|
| Small (50M) | ~2GB | 50,000 | ~5 minutes |
| Base (350M) | ~6GB | 15,000 | ~15 minutes |
| Large (750M) | ~12GB | 8,000 | ~30 minutes |

### Memory Optimization

```bash
# Use FP16 precision
python3.11 main.py train --model-size large --fp16

# Gradient checkpointing
python3.11 main.py train --model-size large --gradient-checkpointing

# Reduce sequence length
python3.11 main.py train --model-size large --max-seq-len 256
```

## 🔬 Research Features

### Custom Experiments

```bash
# Consciousness emergence study
python3.11 main.py experiment consciousness_emergence \
    --coherence-thresholds 0.5,0.7,0.9,0.95 \
    --steps 20000

# Fibonacci resonance analysis
python3.11 main.py experiment fibonacci_resonance \
    --phi-values 1.618,1.6,1.65 \
    --generate-plots

# Phase transition mapping
python3.11 main.py experiment phase_transitions \
    --temperature-range 0.1,2.0 \
    --resolution 100
```

### Custom Datasets

```bash
# Use custom text files
python3.11 main.py train \
    --dataset custom \
    --data-path /path/to/your/text/files \
    --model-size base

# Preprocess custom data
python3.11 data.py preprocess \
    --input-dir /path/to/raw/text \
    --output-dir /path/to/processed \
    --tokenizer-type golden
```

## 📝 Development Workflow

### Code Organization

- `core.py` - Golden embeddings, field evolution, crystal memory
- `perturbations.py` - Lévy, Beta, Adaptive perturbation strategies
- `collapse.py` - Field collapse dynamics and sampling
- `model.py` - Complete model architecture
- `data.py` - Streaming data loaders
- `trainer.py` - Hebbian training (no backprop)
- `inference.py` - Generation and analysis tools
- `main.py` - CLI interface

### Adding New Features

1. **New Perturbation Strategy:**
   - Add to `perturbations.py`
   - Update `main.py` CLI options
   - Add tests to `tests/`

2. **New Dataset:**
   - Add loader to `data.py`
   - Update `main.py` dataset options
   - Add preprocessing if needed

3. **New Analysis:**
   - Add to `inference.py`
   - Create visualization functions
   - Add CLI commands

## 🎯 Next Steps

1. **Start Small**: Begin with `small` model on `wikitext-2`
2. **Validate Physics**: Check that golden spiral and Hebbian learning work
3. **Scale Up**: Move to larger models and datasets
4. **Experiment**: Try different perturbation strategies
5. **Analyze**: Use visualization tools to understand field dynamics
6. **Optimize**: Profile and tune for your hardware

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test outputs in `outputs/`
3. Enable debug mode for detailed logging
4. Check GPU memory usage with `nvidia-smi`

Happy experimenting with field-theoretic language modeling! 🌌
