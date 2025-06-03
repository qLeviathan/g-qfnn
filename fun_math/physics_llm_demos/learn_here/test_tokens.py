#!/usr/bin/env python3
"""
Quick test to examine the LogHebbianNetwork tokens dictionary
"""

import os
import sys
import torch

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import LogHebbianNetwork
from log_coords import LogHebbianNetwork

# Create a network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

hebbian = LogHebbianNetwork(n_tokens=10, device=device)
print(f"Created LogHebbianNetwork with 10 tokens")

# Examine tokens dictionary
print("\nExamining tokens dictionary:")
print(f"Tokens keys: {list(hebbian.tokens.keys())}")
for key, value in hebbian.tokens.items():
    if isinstance(value, torch.Tensor):
        print(f"Token '{key}' shape: {value.shape}, type: {value.dtype}, device: {value.device}")
    else:
        print(f"Token '{key}' type: {type(value)}")

print("\nExamining constants:")
print(f"phi: {hebbian.phi}")
print(f"DT: {hebbian.DT}")
print(f"lambda_cutoff: {hebbian.lambda_cutoff}")
print(f"sigma_gate: {hebbian.sigma_gate}")
print(f"eps_freeze: {hebbian.eps_freeze}")
print(f"Z_step: {hebbian.Z_step}")
print(f"Z_rotor: {hebbian.Z_rotor}")

# Initialize with pattern
hebbian.initialize_tokens(pattern='golden_spiral')
print("\nAfter initialization:")
for key, value in hebbian.tokens.items():
    if isinstance(value, torch.Tensor):
        print(f"Token '{key}' shape: {value.shape}, type: {value.dtype}, device: {value.device}")
    else:
        print(f"Token '{key}' type: {type(value)}")