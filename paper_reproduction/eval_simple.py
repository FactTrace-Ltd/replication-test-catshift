#!/usr/bin/env python3
"""Simple evaluation script for membership inference using BERTScore"""

from bert_score import BERTScorer
import torch
import json
import argparse
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu, anderson_ksamp
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def compare_distributions(sample1, sample2):
    """Statistical tests for distribution comparison"""
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p_value = ks_2samp(sample1, sample2)
    print(f"Kolmogorov-Smirnov test: p-value={ks_p_value:.6f}")
    
    # Mann-Whitney U Test
    mw_stat, mw_p_value = mannwhitneyu(sample1, sample2, alternative='two-sided')
    print(f"Mann-Whitney U test: p-value={mw_p_value:.6f}")
    
    # Anderson-Darling Test
    ad_stat, critical_values, ad_significance_level = anderson_ksamp([sample1, sample2])
    print(f"Anderson-Darling test: statistic={ad_stat:.6f}")
    
    return ks_p_value, mw_p_value, ad_stat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate membership inference attack')
    parser.add_argument('--model', type=str, default='410m', help='model size')
    parser.add_argument('--epoch', type=int, default=9, help='number of epochs')
    parser.add_argument('--size', type=int, default=600, help='fine-tune set size')
    parser.add_argument('--subname', type=str, default='Wikipedia (en)', help='Pile subset name')
    parser.add_argument('--lr', type=float, default=8e-5, help='learning rate')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_scorer = BERTScorer('roberta-large', device=device, rescale_with_baseline=True, lang='en')
    
    print(f"\n=== Evaluating {args.subname} ===")
    print(f"Model: pythia-{args.model}, Epoch: {args.epoch}, FT size: {args.size}, LR: {args.lr}\n")
    
    # Load responses
    response_member_orig = load_jsonl(f'./responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/pythia-{args.model}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-orig.jsonl')
    response_member_ft = load_jsonl(f'./responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/pythia-{args.model}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-ft.jsonl')
    
    response_nonmember_orig = load_jsonl(f'./responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/pythia-{args.model}-nonmember-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-orig.jsonl')
    response_nonmember_ft = load_jsonl(f'./responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/pythia-{args.model}-nonmember-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-ft.jsonl')
    
    print(f"Loaded {len(response_member_orig)} member (orig) responses")
    print(f"Loaded {len(response_member_ft)} member (ft) responses")
    print(f"Loaded {len(response_nonmember_orig)} non-member (orig) responses")
    print(f"Loaded {len(response_nonmember_ft)} non-member (ft) responses\n")
    
    # Extract text responses
    member_orig_text = [r['output_text'] for r in response_member_orig]
    member_ft_text = [r['output_text'] for r in response_member_ft]
    nonmember_orig_text = [r['output_text'] for r in response_nonmember_orig]
    nonmember_ft_text = [r['output_text'] for r in response_nonmember_ft]
    
    # Compute BERTScore for member and non-member
    print("Computing BERTScore for member samples...")
    member_scores = bert_scorer.score(member_ft_text, member_orig_text)[2]  # F1 scores
    print(f"Member BERTScore F1 (mean={member_scores.mean():.4f}, std={member_scores.std():.4f})")
    
    print("Computing BERTScore for non-member samples...")
    nonmember_scores = bert_scorer.score(nonmember_ft_text, nonmember_orig_text)[2]  # F1 scores
    print(f"Non-member BERTScore F1 (mean={nonmember_scores.mean():.4f}, std={nonmember_scores.std():.4f})\n")
    
    # Statistical comparison
    print("=== Statistical Tests ===")
    ks_p, mw_p, ad_stat = compare_distributions(member_scores.numpy(), nonmember_scores.numpy())
    
    # ROC-AUC computation
    print("\n=== Membership Inference Attack ===")
    # Lower scores indicate more difference between FT and orig (better attack)
    # So for attack: member samples should have LOW scores, non-member HIGH scores
    # Flip the scores so that high scores = member, low scores = non-member
    member_scores_np = 1.0 - member_scores.numpy()  # Flip: higher = more different = more member-like
    nonmember_scores_np = 1.0 - nonmember_scores.numpy()
    
    all_scores = np.concatenate([member_scores_np, nonmember_scores_np])
    all_labels = np.concatenate([np.ones(len(member_scores_np)), np.zeros(len(nonmember_scores_np))])
    
    auc = roc_auc_score(all_labels, all_scores)
    print(f"ROC-AUC: {auc:.4f}")
    
    # Also compute with original scores (lower difference = member)
    all_scores_orig = np.concatenate([member_scores.numpy(), nonmember_scores.numpy()])
    all_labels_orig = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    auc_orig = roc_auc_score(all_labels_orig, -all_scores_orig)  # Use negative so higher = more member
    print(f"ROC-AUC (alternative scoring): {auc_orig:.4f}")
    
    # Create plot
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {args.subname} Membership Inference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'./roc_curve_{args.subname.replace(" ", "_").replace("(", "").replace(")", "")}_epoch{args.epoch}.png', dpi=100, bbox_inches='tight')
    print(f"\nROC curve saved to roc_curve_{args.subname.replace(' ', '_').replace('(', '').replace(')', '')}_epoch{args.epoch}.png")
    
    # Save results
    results = {
        'subset': args.subname,
        'model': args.model,
        'epoch': args.epoch,
        'ft_size': args.size,
        'lr': args.lr,
        'auc': float(auc),
        'member_mean_score': float(member_scores.mean()),
        'member_std_score': float(member_scores.std()),
        'nonmember_mean_score': float(nonmember_scores.mean()),
        'nonmember_std_score': float(nonmember_scores.std()),
        'ks_p_value': float(ks_p),
        'mw_p_value': float(mw_p),
        'ad_statistic': float(ad_stat)
    }
    
    with open(f'results_{args.subname.replace(" ", "_").replace("(", "").replace(")", "")}_epoch{args.epoch}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved!")
