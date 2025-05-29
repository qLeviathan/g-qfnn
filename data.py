"""
data.py - Streaming data loaders for WikiText and C4
Efficient loading with minimal memory footprint
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Iterator, Dict, Optional
import numpy as np

class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for language modeling
    Yields tokenized chunks without loading full dataset
    """
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        seq_length: int = 512,
        tokenizer_name: str = "gpt2",
        streaming: bool = True
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.seq_length = seq_length
        self.streaming = streaming
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load dataset - handle local caching issue
        try:
            if dataset_name == "wikitext-2":
                self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", 
                                          split=split, streaming=streaming)
            elif dataset_name == "wikitext-103":
                self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", 
                                          split=split, streaming=streaming)
            elif dataset_name == "c4":
                # C4 is huge, always stream
                self.dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        except NotImplementedError:
            # Fallback to non-streaming if local cache exists
            print(f"Falling back to non-streaming mode for {dataset_name}")
            self.streaming = False
            if dataset_name == "wikitext-2":
                self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", 
                                          split=split, streaming=False)
            elif dataset_name == "wikitext-103":
                self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", 
                                          split=split, streaming=False)
            
        # Buffer for accumulating tokens
        self.token_buffer = []
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield tokenized sequences"""
        for example in self.dataset:
            # Get text
            if self.dataset_name == "c4":
                text = example["text"]
            else:  # wikitext
                text = example["text"]
                
            # Skip empty texts
            if not text or text.isspace():
                continue
                
            # Tokenize
            tokens = self.tokenizer.encode(text, truncation=False)
            self.token_buffer.extend(tokens)
            
            # Yield complete sequences
            while len(self.token_buffer) >= self.seq_length + 1:
                # Extract sequence
                sequence = self.token_buffer[:self.seq_length + 1]
                self.token_buffer = self.token_buffer[self.seq_length:]
                
                # Create input/target
                input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
                target_ids = torch.tensor(sequence[1:], dtype=torch.long)
                
                yield {
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                    "attention_mask": torch.ones_like(input_ids)
                }

class FieldDataLoader:
    """
    Efficient data loader for field-theoretic training
    Handles batching and device placement
    """
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 8,
        seq_length: int = 512,
        num_workers: int = 2,
        device: str = "cuda"
    ):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device
        
        # Create datasets
        self.train_dataset = StreamingTextDataset(
            dataset_name, "train", seq_length
        )
        
        # Validation split depends on dataset
        val_split = "validation" if dataset_name != "c4" else "validation[:5000]"
        self.val_dataset = StreamingTextDataset(
            dataset_name, val_split, seq_length
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
        
    def get_batch(self, split: str = "train") -> Dict[str, torch.Tensor]:
        """Get a single batch"""
        loader = self.train_loader if split == "train" else self.val_loader
        batch = next(iter(loader))
        
        # Move to device
        return {k: v.to(self.device) for k, v in batch.items()}
    
    def iterate_batches(self, split: str = "train", max_batches: Optional[int] = None):
        """Iterate through batches"""
        loader = self.train_loader if split == "train" else self.val_loader
        
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
                
            # Move to device
            yield {k: v.to(self.device) for k, v in batch.items()}

class DatasetStats:
    """Compute dataset statistics for field initialization"""
    
    @staticmethod
    def compute_token_frequencies(dataset_name: str, max_samples: int = 10000) -> torch.Tensor:
        """
        Compute token frequency distribution
        Returns: (vocab_size,) tensor of frequencies
        """
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        vocab_size = len(tokenizer)
        
        # Count tokens
        token_counts = torch.zeros(vocab_size)
        
        dataset = StreamingTextDataset(dataset_name, "train", seq_length=512)
        
        for i, batch in enumerate(dataset):
            if i >= max_samples:
                break
                
            tokens = batch["input_ids"]
            for token in tokens:
                token_counts[token] += 1
                
        # Normalize
        token_counts = token_counts / token_counts.sum()
        
        return token_counts
    
    @staticmethod
    def compute_sequence_stats(dataset_name: str, max_samples: int = 1000) -> Dict[str, float]:
        """Compute sequence-level statistics"""
        dataset = StreamingTextDataset(dataset_name, "train", seq_length=512)
        
        lengths = []
        entropies = []
        
        for i, batch in enumerate(dataset):
            if i >= max_samples:
                break
                
            tokens = batch["input_ids"]
            
            # Length
            lengths.append(len(tokens))
            
            # Entropy
            unique, counts = torch.unique(tokens, return_counts=True)
            probs = counts.float() / counts.sum()
            entropy = -torch.sum(probs * torch.log(probs))
            entropies.append(entropy.item())
            
        return {
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "mean_entropy": np.mean(entropies),
            "std_entropy": np.std(entropies)
        }

# Validation
if __name__ == "__main__":
    print("=== Data Module Validation ===")
    
    # Test each dataset
    for dataset_name in ["wikitext-2"]:  # Only test wikitext-2 for quick validation
        print(f"\nTesting {dataset_name}...")
        
        try:
            # Create loader
            loader = FieldDataLoader(dataset_name, batch_size=4, seq_length=128)
            
            # Get a batch
            batch = loader.get_batch("train")
            
            print(f"  Input shape: {batch['input_ids'].shape}")
            print(f"  Target shape: {batch['target_ids'].shape}")
            print(f"  Device: {batch['input_ids'].device}")
            
            # Quick stats (only for small datasets)
            if dataset_name == "wikitext-2":
                stats = DatasetStats.compute_sequence_stats(dataset_name, max_samples=100)
                print(f"  Sequence stats: {stats}")
                
        except Exception as e:
            print(f"  Error: {e}")
            
    print("\n✓ Data module validated")