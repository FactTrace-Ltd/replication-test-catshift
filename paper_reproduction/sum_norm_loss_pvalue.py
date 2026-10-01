import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--subname', type=str, default='arxiv', help='subset name')
parser.add_argument('--temp', type=float, default=0.0, help='generation temperature')
parser.add_argument('--topp', type=float, default=1.0, help='generation top_p')
parser.add_argument('--epoch', type=int, default=9, help='epoch')
parser.add_argument('--logging', type=str, default='', help='logging name')

args = parser.parse_args()

# 文件路径
p_value_path = f'/home/ubuntu/replication-test-catshift/copyright/pile_{args.subname}_temp_{args.temp}_topp_{args.topp}_pvalue_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}.csv'
loss_path = f'/home/ubuntu/replication-test-catshift/copyright/pile_{args.subname}_temp_{args.temp}_topp_{args.topp}_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}.csv'

# 提取 "ks_pvalue" 列为列表
p_value_df = pd.read_csv(p_value_path)
p_value_ls = p_value_df['ks_pvalue'].tolist()

# 提取 "member_loss" 和 "nonmember_loss" 列为列表
loss_df = pd.read_csv(loss_path)
member_loss_ls = loss_df['member_loss'].tolist()
nonmember_loss_ls = loss_df['nonmember_loss'].tolist()

sum_loss_ls = [sum(x) for x in zip(member_loss_ls, nonmember_loss_ls)]

member_loss_ls_norm = (member_loss_ls - np.min(member_loss_ls)) / (np.max(member_loss_ls) - np.min(member_loss_ls))
nonmember_loss_ls_norm = (nonmember_loss_ls - np.min(nonmember_loss_ls)) / (np.max(nonmember_loss_ls) - np.min(nonmember_loss_ls))
sum_loss_ls_norm = (sum_loss_ls - np.min(sum_loss_ls)) / (np.max(sum_loss_ls) - np.min(sum_loss_ls))

df_dict = {"sum_loss":sum_loss_ls_norm}
df_pvalue_dict = {"pvalue":p_value_ls}
df_out = pd.DataFrame(df_dict)
df_pvalue_out = pd.DataFrame(df_pvalue_dict)
df_out.to_csv(f"/home/ubuntu/replication-test-catshift/copyright/p_value_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/pile-full-subsets-{args.subname}_temp_{args.temp}_topp_{args.topp}-sum-loss_epoch_9.csv", index=False)
df_pvalue_out.to_csv(f"/home/ubuntu/replication-test-catshift/copyright/p_value_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/pile-full-subsets-{args.subname}_temp_{args.temp}_topp_{args.topp}-pvalue_epoch_9.csv", index=False)

# 绘制第一个折线图：p_value_ls vs member_loss_ls_norm
plt.figure(figsize=(10, 5))
plt.plot(p_value_ls, label='p_value_ls')
plt.plot(sum_loss_ls_norm, label='sum_loss_ls_norm')
plt.title('p_value_ls vs sum_loss_ls_norm')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()
plt.savefig(f'/home/ubuntu/replication-test-catshift/copyright/p_value_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/pile-full-subsets-{args.subname}_temp_{args.temp}_topp_{args.topp}-sum-loss-pvalue_epoch_9.png')

plt.figure(figsize=(10, 5))
plt.plot(p_value_ls, label='p_value_ls')
plt.plot(nonmember_loss_ls_norm, label='nonmember_loss_ls_norm')
plt.title('p_value_ls vs nonmember_loss_ls_norm')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()
plt.savefig(f'/home/ubuntu/replication-test-catshift/copyright/p_value_loss_ft_more_layers_{args.subname}_epoch_{args.epoch}_{args.logging}/pile-full-subsets-{args.subname}_temp_{args.temp}_topp_{args.topp}-nonmember-loss-pvalue_epoch_9.png')