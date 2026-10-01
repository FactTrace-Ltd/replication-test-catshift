from bert_score import BERTScorer
import torch
import json
import argparse
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu, anderson_ksamp
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import re
import os
import pandas as pd

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def dump_txt(data, file_path):
    with open(file_path, 'w') as file:
        file.write(str(data) + '\n')

def compare_distributions(sample1, sample2):
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p_value = ks_2samp(sample1, sample2)
    print(f"Kolmogorov-Smirnov test statistic: {ks_stat}, p-value: {ks_p_value}")
    if ks_p_value < 0.05:
        print("Kolmogorov-Smirnov test: The two samples likely come from different distributions.")
    else:
        print("Kolmogorov-Smirnov test: The two samples likely come from the same distribution.")
    
    # Mann-Whitney U Test
    mw_stat, mw_p_value = mannwhitneyu(sample1, sample2, alternative='two-sided')
    print(f"Mann-Whitney U test statistic: {mw_stat}, p-value: {mw_p_value}")
    if mw_p_value < 0.05:
        print("Mann-Whitney U test: The two samples likely come from different distributions.")
    else:
        print("Mann-Whitney U test: The two samples likely come from the same distribution.")
    
    # Anderson-Darling Test
    ad_stat, critical_values, ad_significance_level = anderson_ksamp([sample1, sample2])
    print(f"Anderson-Darling test statistic: {ad_stat}, significance level: {ad_significance_level}")
    if ad_stat > critical_values[2]:  # Using 5% significance level
        print("Anderson-Darling test: The two samples likely come from different distributions.")
    else:
        print("Anderson-Darling test: The two samples likely come from the same distribution.")

    return ks_p_value, mw_p_value, ad_stat, critical_values[2]


def get_num_from_directory(directory_path):


    # List to store the extracted numbers
    numbers = []
    
    # Iterate over each file/directory in the specified path
    for filename in os.listdir(directory_path):
        # Use regex to find numbers in the filename
        match = re.search(r'checkpoint-(\d+)', filename)
        if match:
            # Append the extracted number to the list as an integer
            numbers.append(int(match.group(1)))

    return numbers



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='160m',help='model name') #160m 410m 1b 1.4b 2.8b 6.9b 12b
parser.add_argument('--epoch', type=int, default=9,help='model name')
parser.add_argument('--size', type=int, default=600,help='model name')
parser.add_argument('--subname', type=str, default='arxiv', help='subset name')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--temp', type=float, default=0.0, help='generation temperature')
parser.add_argument('--topp', type=float, default=1.0, help='generation top_p')
parser.add_argument('--logging', type=str, default='', help='logging of the file')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_scorer = BERTScorer('roberta-large', device=device, rescale_with_baseline=True, lang='en')


loss_file_member = f'/home/ubuntu/replication-test-catshift/copyright/output_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/pythia-{args.model}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}/checkpoint-675/trainer_state.json'

loss_file_nonmember = f'/home/ubuntu/replication-test-catshift/copyright/output_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/pythia-{args.model}-nonmember-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}/checkpoint-675/trainer_state.json'
loss_datafile_member = json.load(open(loss_file_member))['log_history']
loss_datafile_nonmember = json.load(open(loss_file_nonmember))['log_history']
loss_l_member = []
loss_l_nonmember = []

for i in range(len(loss_datafile_member)):
    try:
        loss_data_memeber = loss_datafile_member[i]['loss']
        loss_l_member.append(loss_data_memeber)
    except:
        continue


for i in range(len(loss_datafile_nonmember)):
    try:
        loss_data_nonmember = loss_datafile_nonmember[i]['loss']
        loss_l_nonmember.append(loss_data_nonmember)
    except:
        continue

# Find the largest value in the list
max_value_member = max(loss_l_member)
max_value_nonmember = max(loss_l_nonmember)

# Divide each value by the largest value
normalized_loss_l_member = [x / max_value_member for x in loss_l_member]
normalized_loss_l_nonmember = [x / max_value_nonmember for x in loss_l_nonmember]

results_dict = {}
ks_p_value_l=[]
mw_p_value_l=[]
auc_scores_l=[]


    directory_path = f"/home/ubuntu/replication-test-catshift/copyright/output_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/pythia-{args.model}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}"
numbers = get_num_from_directory(directory_path)
numbers.sort()
for num in numbers:
    for candidate in ['member', 'nonmember']:
        print(f"#############{num}############")
        model_name = f'pythia-{args.model}'
        log_str = f'{candidate}-{args.model}-epoch-{args.epoch}'
        response_orig = load_jsonl(f'/home/ubuntu/replication-test-catshift/copyright/responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/{model_name}-{log_str}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-orig.jsonl')
        response_ft = load_jsonl(f'/home/ubuntu/replication-test-catshift/copyright/responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/all_checkpoint/{model_name}-{log_str}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-{num}-ft.jsonl')
        
        response_only_orig = []
        response_only_ft = []
        
        for i in range(len(response_orig)):
            response_only_orig.append(response_orig[i]['output_text'])
            response_only_ft.append(response_ft[i]['output_text'])
        
        ctc_scores = bert_scorer.score(response_only_ft, response_only_orig)[2]
        
        results_dict[candidate]=ctc_scores
    
    ks_p_value, mw_p_value, ad_stat, adcv=compare_distributions(results_dict['member'], results_dict['nonmember'])
    
    # Calculate AUC-ROC for membership inference
    member_scores = results_dict['member']
    nonmember_scores = results_dict['nonmember']
    
    # Create labels: 1 for member, 0 for non-member
    all_scores = np.concatenate([member_scores, nonmember_scores])
    all_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    
    # Calculate AUC-ROC
    auc_score = roc_auc_score(all_labels, all_scores)
    print(f"AUC-ROC Score at checkpoint {num}: {auc_score}")
    
    os.makedirs(f'bert_results_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}', exist_ok=True)
    os.makedirs(f'p_value_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}', exist_ok=True)
    file_path =f'/home/ubuntu/replication-test-catshift/copyright/bert_results_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/pile_full_bert_{args.model}_{args.epoch}_{args.subname}_{args.size}_{args.lr}_test.txt'
    txt_info=f'''
    Checkpoint: {num}
    
    AUC-ROC Score: {auc_score}
    
    Kolmogorov-Smirnov test statistic: p-value: {ks_p_value}
    
    Mann-Whitney U test statistic: p-value: {mw_p_value}
    
    Anderson-Darling test statistic: {ad_stat} critical-value:{adcv}
    '''
    dump_txt(txt_info, file_path)
    ks_p_value_l.append(ks_p_value)
    mw_p_value_l.append(mw_p_value)
    auc_scores_l.append(auc_score)
# Plot AUC scores over checkpoints
plt.figure(figsize=(10, 6))
plt.plot(auc_scores_l, marker='o', linestyle='-', color='purple', linewidth=2, markersize=8, label='AUC-ROC')
plt.axhline(y=0.5, color='gray', linestyle='--', label='Random Baseline')
plt.title(f'AUC-ROC Score over Checkpoints - {args.subname} (lr={args.lr})')
plt.xlabel('Checkpoint')
plt.ylabel('AUC-ROC Score')
plt.ylim([0.4, 1.0])
plt.legend()
plt.grid(True)
plt.savefig(f'/home/ubuntu/replication-test-catshift/copyright/p_value_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/{args.model}-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-auc.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(ks_p_value_l, marker='o', linestyle='-', color='b', label='P-value')
# Plot the second line
plt.plot(normalized_loss_l_member, marker='s', linestyle='--', color='g', label='member loss')

# Plot the third line
plt.plot(normalized_loss_l_nonmember, marker='^', linestyle='-.', color='r', label='nonmember loss')
# Add title and labels
plt.title(f'P-Value subsets-{args.subname}-{args.lr}')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# Add legend
plt.legend()

# Show grid
plt.grid(True)

plt.savefig(f'/home/ubuntu/replication-test-catshift/copyright/p_value_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/{args.model}-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-ks.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(mw_p_value_l, marker='o', linestyle='-', color='b', label='Loss')
plt.plot(normalized_loss_l_member, marker='s', linestyle='--', color='g', label='member loss')

# Plot the third line
plt.plot(normalized_loss_l_nonmember, marker='^', linestyle='-.', color='r', label='nonmember loss')

# Add title and labels
plt.title(f'P-Value subsets-{args.subname}-{args.lr}')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# Add legend
plt.legend()

# Show grid
plt.grid(True)

plt.savefig(f'/home/ubuntu/replication-test-catshift/copyright/p_value_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/{args.model}-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-mw.png')
plt.close()
print(len(loss_l_member))
print(len(loss_l_nonmember))
print(len(ks_p_value_l))
print(len(mw_p_value_l))
print(f"AUC Scores: {auc_scores_l}")     
df_dict = {'member_loss': loss_l_member, 'nonmember_loss': loss_l_nonmember}
df_loss = pd.DataFrame(df_dict)
df_dict_test = {'ks_pvalue': ks_p_value_l, 'mw_pvalue': mw_p_value_l, 'auc_score': auc_scores_l}
df_pvalue = pd.DataFrame(df_dict_test)
df_normalized_loss_dict = {'member_loss': normalized_loss_l_member, 'nonmember_loss': normalized_loss_l_nonmember}
df_normalized_loss = pd.DataFrame(df_normalized_loss_dict)
df_loss.to_csv(f'/home/ubuntu/replication-test-catshift/copyright/pile_{args.subname}_temp_{args.temp}_topp_{args.topp}_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}.csv', index=False)
df_pvalue.to_csv(f'/home/ubuntu/replication-test-catshift/copyright/pile_{args.subname}_temp_{args.temp}_topp_{args.topp}_pvalue_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}.csv', index=False)
df_normalized_loss.to_csv(f'/home/ubuntu/replication-test-catshift/copyright/pile_{args.subname}_temp_{args.temp}_topp_{args.topp}_normalized_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}.csv', index=False)

print(f"\n=== AUC-ROC Results ===")
print(f"Mean AUC-ROC: {np.mean(auc_scores_l):.4f}")
print(f"Max AUC-ROC: {np.max(auc_scores_l):.4f}")
print(f"Min AUC-ROC: {np.min(auc_scores_l):.4f}")