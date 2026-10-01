#!/usr/bin/env python3
"""
CatShift Membership Inference Evaluation.

Pipeline:
1. Compare original vs fine-tuned model responses using BERTScore
2. Member samples should have LOW similarity (catastrophic forgetting)
3. Non-member samples should have HIGH similarity (model unchanged)
4. Compute ROC-AUC and statistical tests on similarity distributions
"""

from bert_score import BERTScorer
import torch
import json
import argparse
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu, anderson_ksamp
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
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
    
    # Anderson-Darling Test (with warning suppression)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ad_stat, critical_values, ad_significance_level = anderson_ksamp([sample1, sample2])
            print(f"Anderson-Darling test: statistic={ad_stat:.6f}")
        except:
            ad_stat = np.nan
    
    return ks_p_value, mw_p_value, ad_stat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CatShift: Membership Inference via Catastrophic Forgetting')
    parser.add_argument('--model', type=str, default='410m', help='model size')
    parser.add_argument('--epoch', type=int, default=9, help='number of epochs')
    parser.add_argument('--size', type=int, default=600, help='fine-tune set size')
    parser.add_argument('--subname', type=str, default='Wikipedia (en)', help='Pile subset name')
    parser.add_argument('--lr', type=float, default=8e-5, help='learning rate')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    bert_scorer = BERTScorer('roberta-large', device=device, rescale_with_baseline=True, lang='en')
    
    print(f"\n{'='*60}")
    print(f"CatShift Membership Inference Evaluation")
    print(f"{'='*60}")
    print(f"Subset: {args.subname}")
    print(f"Model: pythia-{args.model}, Epoch: {args.epoch}, FT size: {args.size}, LR: {args.lr}\n")
    
    # Load responses
    print("Loading responses...")
    response_member_orig = load_jsonl(f'./responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/pythia-{args.model}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-orig.jsonl')
    response_member_ft = load_jsonl(f'./responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/pythia-{args.model}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-ft.jsonl')
    
    response_nonmember_orig = load_jsonl(f'./responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/pythia-{args.model}-nonmember-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-orig.jsonl')
    response_nonmember_ft = load_jsonl(f'./responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/pythia-{args.model}-nonmember-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-ft.jsonl')
    
    print(f"✓ Loaded {len(response_member_orig)} member (orig) responses")
    print(f"✓ Loaded {len(response_member_ft)} member (ft) responses")
    print(f"✓ Loaded {len(response_nonmember_orig)} non-member (orig) responses")
    print(f"✓ Loaded {len(response_nonmember_ft)} non-member (ft) responses\n")
    
    # Extract text responses
    member_orig_text = [r['output_text'] for r in response_member_orig]
    member_ft_text = [r['output_text'] for r in response_member_ft]
    nonmember_orig_text = [r['output_text'] for r in response_nonmember_orig]
    nonmember_ft_text = [r['output_text'] for r in response_nonmember_ft]
    
    # Compute BERTScore for member and non-member
    print(f"{'='*60}")
    print(f"Computing BERTScore Similarity (F1 scores)")
    print(f"{'='*60}")
    
    print("Computing BERTScore for member samples...")
    member_scores = bert_scorer.score(member_ft_text, member_orig_text)[2]  # F1 scores
    print(f"  Mean: {member_scores.mean():.4f}, Std: {member_scores.std():.4f}")
    print(f"  Expected: LOW (catastrophic forgetting diverges responses)")
    
    print("\nComputing BERTScore for non-member samples...")
    nonmember_scores = bert_scorer.score(nonmember_ft_text, nonmember_orig_text)[2]  # F1 scores
    print(f"  Mean: {nonmember_scores.mean():.4f}, Std: {nonmember_scores.std():.4f}")
    print(f"  Expected: HIGH (model unchanged on non-member data)")
    
    # Statistical comparison
    print(f"\n{'='*60}")
    print(f"Statistical Tests (Member vs Non-Member Distributions)")
    print(f"{'='*60}")
    ks_p, mw_p, ad_stat = compare_distributions(member_scores.numpy(), nonmember_scores.numpy())
    
    # ROC-AUC computation
    print(f"\n{'='*60}")
    print(f"Membership Inference Attack Performance")
    print(f"{'='*60}")
    
    # Strategy: low similarity = member (fine-tuned), high similarity = non-member (unchanged)
    # So we want: member_scores (low) < nonmember_scores (high)
    # For ROC curve: higher scores should indicate membership
    # Invert member scores so low similarity = high membership probability
    member_scores_inverted = 1.0 - member_scores.numpy()
    nonmember_scores_inverted = 1.0 - nonmember_scores.numpy()
    
    all_scores = np.concatenate([member_scores_inverted, nonmember_scores_inverted])
    all_labels = np.concatenate([np.ones(len(member_scores_inverted)), np.zeros(len(nonmember_scores_inverted))])
    
    auc = roc_auc_score(all_labels, all_scores)
    print(f"ROC-AUC Score: {auc:.4f}")
    print(f"  Expected: > 0.95 (paper reports 0.979)")
    
    # Also show the raw comparison
    print(f"\nDifference in mean similarity:")
    print(f"  Member - Non-member: {member_scores.mean() - nonmember_scores.mean():.6f}")
    print(f"  Expected: NEGATIVE (members have lower similarity)")
    
    # Create plot
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'CatShift: {args.subname} Membership Inference\nPythia-{args.model} (Epoch {args.epoch}, LR {args.lr})', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    
    safe_name = args.subname.replace(" ", "_").replace("(", "").replace(")", "")
    plot_path = f'roc_catshift_{safe_name}_epoch{args.epoch}.png'
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ ROC curve saved to {plot_path}")
    
    # Save results
    results = {
        'method': 'CatShift (Catastrophic Forgetting)',
        'subset': args.subname,
        'model': args.model,
        'epoch': args.epoch,
        'ft_size': args.size,
        'lr': args.lr,
        'auc': float(auc),
        'member_mean_similarity': float(member_scores.mean()),
        'member_std_similarity': float(member_scores.std()),
        'nonmember_mean_similarity': float(nonmember_scores.mean()),
        'nonmember_std_similarity': float(nonmember_scores.std()),
        'ks_p_value': float(ks_p),
        'mw_p_value': float(mw_p),
        'ad_statistic': float(ad_stat) if not np.isnan(ad_stat) else None
    }
    
    result_path = f'results_catshift_{safe_name}_epoch{args.epoch}.json'
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {result_path}")
    print(f"\n{'='*60}\n")
