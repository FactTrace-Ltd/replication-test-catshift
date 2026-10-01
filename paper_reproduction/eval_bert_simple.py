#!/usr/bin/env python3
"""
Simplified evaluation script for membership inference attacks.
Works with basic response files (no checkpoints required).
"""

from bert_score import BERTScorer
import torch
import json
import argparse
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
import pandas as pd

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='410m', help='model name')
parser.add_argument('--epoch', type=int, default=1, help='epoch')
parser.add_argument('--size', type=int, default=4, help='dataset size')
parser.add_argument('--subname', type=str, default='europarl', help='subset name')
parser.add_argument('--lr', type=float, default=8e-5, help='learning rate')
parser.add_argument('--output_dir', type=str, default='/home/ubuntu/replication-test-catshift/eval_results', help='output directory')

args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT scorer
print("Loading BERTScorer (roberta-large)...")
bert_scorer = BERTScorer('roberta-large', device=device, rescale_with_baseline=True, lang='en')

# Construct file paths
model_name = f'pythia-{args.model}'
base_path = f'/home/ubuntu/replication-test-catshift/responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp'

print(f"\nEvaluating:")
print(f"  Model: {model_name}")
print(f"  Dataset: {args.subname}")
print(f"  Size: {args.size}")
print(f"  Learning rate: {args.lr}")

# Dictionary to store results
results = {}

# Evaluate member and non-member models
for candidate in ['member', 'nonmember']:
    print(f"\n{'='*60}")
    print(f"Evaluating {candidate.upper()} model")
    print('='*60)
    
    # File naming convention
    log_str = f'{candidate}-{args.model}-epoch-{args.epoch}'
    
    orig_file = f'{base_path}/{model_name}-{log_str}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-orig.jsonl'
    ft_file = f'{base_path}/{model_name}-{log_str}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-ft.jsonl'
    
    print(f"Original responses: {orig_file}")
    print(f"Fine-tuned responses: {ft_file}")
    
    # Check if files exist
    if not os.path.exists(orig_file):
        print(f"  ⚠️  File not found: {orig_file}")
        continue
    if not os.path.exists(ft_file):
        print(f"  ⚠️  File not found: {ft_file}")
        continue
    
    # Load responses
    print("  Loading responses...")
    response_orig = load_jsonl(orig_file)
    response_ft = load_jsonl(ft_file)
    
    print(f"  Loaded {len(response_orig)} original samples")
    print(f"  Loaded {len(response_ft)} fine-tuned samples")
    
    # Extract output texts
    response_only_orig = [r['output_text'] for r in response_orig]
    response_only_ft = [r['output_text'] for r in response_ft]
    
    # Compute BERT scores
    print("  Computing BERT scores...")
    bert_scores = bert_scorer.score(response_only_ft, response_only_orig)[2].numpy()
    
    print(f"  BERT F1 scores: mean={np.mean(bert_scores):.4f}, std={np.std(bert_scores):.4f}")
    print(f"                 min={np.min(bert_scores):.4f}, max={np.max(bert_scores):.4f}")
    
    results[candidate] = bert_scores

# Calculate AUC-ROC for membership inference
print(f"\n{'='*60}")
print("MEMBERSHIP INFERENCE ATTACK EVALUATION")
print('='*60)

if len(results) == 2:
    member_scores = results['member']
    nonmember_scores = results['nonmember']
    
    # Create binary labels: 1 for member, 0 for non-member
    all_scores = np.concatenate([member_scores, nonmember_scores])
    all_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    
    # Calculate AUC-ROC
    auc_score = roc_auc_score(all_labels, all_scores)
    print(f"\n✓ AUC-ROC Score: {auc_score:.4f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    print(f"  Member scores:     mean={np.mean(member_scores):.4f}, std={np.std(member_scores):.4f}")
    print(f"  Non-member scores: mean={np.mean(nonmember_scores):.4f}, std={np.std(nonmember_scores):.4f}")
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = ks_2samp(member_scores, nonmember_scores)
    print(f"\n  Kolmogorov-Smirnov test:")
    print(f"    Statistic: {ks_stat:.6f}, p-value: {ks_p:.6e}")
    
    # Mann-Whitney U test
    mw_stat, mw_p = mannwhitneyu(member_scores, nonmember_scores, alternative='two-sided')
    print(f"\n  Mann-Whitney U test:")
    print(f"    Statistic: {mw_stat:.6f}, p-value: {mw_p:.6e}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'metric': ['AUC-ROC', 'KS p-value', 'MW p-value', 'Member mean', 'Nonmember mean'],
        'value': [auc_score, ks_p, mw_p, np.mean(member_scores), np.mean(nonmember_scores)]
    })
    
    results_csv = f'{args.output_dir}/results_{args.subname}_{args.size}_{args.lr}.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to: {results_csv}")
    
    # Plot distributions and AUC
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Score distributions
    axes[0].hist(member_scores, bins=20, alpha=0.6, label='Member', color='red')
    axes[0].hist(nonmember_scores, bins=20, alpha=0.6, label='Non-member', color='blue')
    axes[0].set_xlabel('BERT F1 Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'BERT Score Distributions\nAUC-ROC = {auc_score:.4f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    axes[1].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_score:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve - Membership Inference')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plot_file = f'{args.output_dir}/eval_plots_{args.subname}_{args.size}_{args.lr}.png'
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"✓ Plots saved to: {plot_file}")
    plt.close()
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print('='*60)

else:
    print("❌ Error: Could not load both member and non-member response files")
