"""
main.py - CLI interface for field-theoretic language model
Provides training, inference, and analysis commands
"""

import argparse
import torch
import json
from pathlib import Path
import sys

from model import create_small_model, create_base_model, create_large_model
from data import FieldDataLoader
from trainer import FieldTrainer, AblationTrainer
from inference import FieldInference, load_checkpoint

def train_command(args):
    """Execute training"""
    print("=== Field-Theoretic LM Training ===")
    print(f"Model size: {args.model_size}")
    print(f"Embedding: {args.embedding}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    
    # Create model
    if args.model_size == "small":
        model = create_small_model(args.embedding)
    elif args.model_size == "base":
        model = create_base_model(args.embedding)
    else:
        model = create_large_model(args.embedding)
    
    model = model.to(args.device)
    
    # Create data loader
    data_loader = FieldDataLoader(
        args.dataset,
        batch_size=args.batch_size,
        seq_length=args.seq_length
    )
    
    # Create trainer
    trainer = FieldTrainer(
        model,
        data_loader,
        output_dir=args.output_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every
    )
    
    # Train
    if args.perturbation_schedule:
        schedule = json.loads(args.perturbation_schedule)
        schedule = {int(k): v for k, v in schedule.items()}
    else:
        schedule = None
        
    trainer.train(args.num_steps, schedule)
    
def generate_command(args):
    """Execute generation"""
    print("=== Field-Theoretic Generation ===")
    
    # Load model
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model, _ = load_checkpoint(args.checkpoint, args.device)
    else:
        model = create_small_model(args.embedding).to(args.device)
    
    # Create inference engine
    inference = FieldInference(model, device=args.device)
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Perturbation: {args.perturbation}")
    print("\nGenerations:")
    print("-" * 50)
    
    generations = inference.generate_text(
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        num_samples=args.num_samples,
        perturbation_type=args.perturbation
    )
    
    for i, text in enumerate(generations):
        print(f"\n[Sample {i+1}]")
        print(text)
        
def analyze_command(args):
    """Execute field analysis"""
    print("=== Field Dynamics Analysis ===")
    
    # Load model
    if args.checkpoint:
        model, _ = load_checkpoint(args.checkpoint, args.device)
    else:
        model = create_small_model(args.embedding).to(args.device)
    
    # Create inference engine
    inference = FieldInference(model, device=args.device)
    
    # Analyze
    print(f"Analyzing: {args.text}")
    
    if args.visualize:
        inference.visualize_field_evolution(args.text, save_path=args.save_path)
    else:
        analysis = inference.analyze_field_dynamics(args.text)
        
        print(f"\nCoherence range: [{analysis['coherence'].min():.3f}, "
              f"{analysis['coherence'].max():.3f}]")
        print(f"Collapse positions: {(analysis['collapsed_tokens'] >= 0).sum().item()}")
        
        if 'layer_coherence' in analysis:
            mean_coherence = analysis['layer_coherence'].mean(dim=1)
            print(f"Layer coherence progression: {mean_coherence.tolist()}")
            
def ablation_command(args):
    """Execute ablation study"""
    print("=== Ablation Study ===")
    
    ablation = AblationTrainer({}, output_dir=args.output_dir)
    
    results = ablation.run_ablation(
        args.name,
        embedding_types=args.embeddings.split(","),
        perturbation_types=args.perturbations.split(","),
        datasets=args.datasets.split(","),
        num_steps=args.num_steps
    )
    
    print(f"\nResults saved to: {args.output_dir}/{args.name}_results.json")
    
def profile_command(args):
    """Execute performance profiling"""
    print("=== Performance Profiling ===")
    
    # Load or create model
    if args.checkpoint:
        model, _ = load_checkpoint(args.checkpoint, args.device)
    else:
        model = create_small_model("golden").to(args.device)
    
    # Create inference engine
    inference = FieldInference(model, device=args.device)
    
    # Profile
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    
    results = inference.profile_inference(batch_sizes, seq_lengths)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

def main():
    parser = argparse.ArgumentParser(
        description="Field-Theoretic Language Model CLI"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--model-size', choices=['small', 'base', 'large'], 
                            default='small')
    train_parser.add_argument('--embedding', choices=['golden', 'log_phase'], 
                            default='golden')
    train_parser.add_argument('--dataset', choices=['wikitext-2', 'wikitext-103', 'c4'],
                            default='wikitext-2')
    train_parser.add_argument('--batch-size', type=int, default=8)
    train_parser.add_argument('--seq-length', type=int, default=512)
    train_parser.add_argument('--num-steps', type=int, default=10000)
    train_parser.add_argument('--output-dir', default='outputs')
    train_parser.add_argument('--log-every', type=int, default=100)
    train_parser.add_argument('--save-every', type=int, default=1000)
    train_parser.add_argument('--eval-every', type=int, default=500)
    train_parser.add_argument('--perturbation-schedule', type=str, 
                            help='JSON string of step->perturbation mapping')
    train_parser.add_argument('--device', default='cuda')
    
    # Generation command
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('prompt', type=str, help='Generation prompt')
    gen_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    gen_parser.add_argument('--embedding', choices=['golden', 'log_phase'], 
                          default='golden')
    gen_parser.add_argument('--max-length', type=int, default=100)
    gen_parser.add_argument('--temperature', type=float, default=0.8)
    gen_parser.add_argument('--top-p', type=float, default=0.9)
    gen_parser.add_argument('--num-samples', type=int, default=1)
    gen_parser.add_argument('--perturbation', choices=['levy', 'beta', 'adaptive'],
                          default='adaptive')
    gen_parser.add_argument('--device', default='cuda')
    
    # Analysis command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze field dynamics')
    analyze_parser.add_argument('text', type=str, help='Text to analyze')
    analyze_parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
    analyze_parser.add_argument('--embedding', choices=['golden', 'log_phase'],
                              default='golden')
    analyze_parser.add_argument('--visualize', action='store_true')
    analyze_parser.add_argument('--save-path', type=str, help='Save visualization')
    analyze_parser.add_argument('--device', default='cuda')
    
    # Ablation command
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study')
    ablation_parser.add_argument('name', type=str, help='Study name')
    ablation_parser.add_argument('--embeddings', default='golden,log_phase')
    ablation_parser.add_argument('--perturbations', default='levy,beta,adaptive')
    ablation_parser.add_argument('--datasets', default='wikitext-2')
    ablation_parser.add_argument('--num-steps', type=int, default=1000)
    ablation_parser.add_argument('--output-dir', default='ablations')
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Profile performance')
    profile_parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
    profile_parser.add_argument('--batch-sizes', default='1,4,8,16')
    profile_parser.add_argument('--seq-lengths', default='128,256,512')
    profile_parser.add_argument('--output', type=str, help='Output JSON file')
    profile_parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'ablation':
        ablation_command(args)
    elif args.command == 'profile':
        profile_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()