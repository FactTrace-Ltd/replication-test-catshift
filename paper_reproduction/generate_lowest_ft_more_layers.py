import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import datasets
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from peft import get_peft_model, LoraConfig, TaskType
import json
from tqdm import tqdm
import pandas as pd
from functools import partial
import argparse
import re

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='160m',help='model name') #160m 410m 1b 1.4b 2.8b 6.9b 12b
parser.add_argument('--epoch', type=int, default=3,help='model name') #160m 410m 1b 1.4b 2.8b 6.9b 12b
parser.add_argument('--subname', type=str, default='arxiv',help='model name')
parser.add_argument('--size', type=int, default=600 ,help='model name')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--temp', type=float, default=0.0, help='generation temperature')
parser.add_argument('--topp', type=float, default=1.0, help='generation top_p')
parser.add_argument('--candidate', type=str, default='member', help='learning rate')
args = parser.parse_args()



# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"



loss_file = f'/home/ubuntu/replication-test-catshift/copyright/output_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/gpt-neo-{args.model}-{args.candidate}-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}/checkpoint-675/trainer_state.json'

loss_datafile = json.load(open(loss_file))['log_history']

loss_l = []

for i in range(len(loss_datafile)):
    try:
        loss_data = loss_datafile[i]['loss']
        loss_l.append(loss_data)
    except:
        continue

model_name = f'pythia-{args.model}'
# Load the tokenizer and model
model_name_hf_ori = f"/home/ubuntu/replication-test-catshift/{model_name}"  # You can choose other sizes as well
tokenizer = AutoTokenizer.from_pretrained(model_name_hf_ori)
tokenizer.padding_side = "left"
# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
data_files = f"/home/ubuntu/replication-test-catshift/data_inference/{args.subname}_train.jsonl"
raw_train_data_df = pd.read_json(data_files, lines=True)

#Pile Validation Set
val_data_files = f"/home/ubuntu/replication-test-catshift/data_inference/{args.subname}_val.jsonl"
raw_val_data_df = pd.read_json(val_data_files, lines=True)

tds=Dataset.from_pandas(raw_train_data_df)
vds=Dataset.from_pandas(raw_val_data_df)

raw_data = DatasetDict()

raw_data['train'] = tds
raw_data['validation'] = vds


# Tokenize the input data
def tokenize_function(examples,max_length=384):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    #tokens["labels"] = tokens["input_ids"].copy()
    return tokens

data_num = 1000
A_members = raw_data['validation'].shuffle(seed=42).select(range(0, args.size)).map(partial(tokenize_function,max_length=512), batched=True, remove_columns=["text"])
A_nonmembers = raw_data['validation'].shuffle(seed=42).select(range(0, args.size)).map(partial(tokenize_function,max_length=512), batched=True, remove_columns=["text"])

B_members = raw_data['validation'].shuffle(seed=42).select(range(data_num, data_num*2)).map(tokenize_function, batched=True, remove_columns=["text"])
B_nonmembers = raw_data['validation'].shuffle(seed=42).select(range(data_num, data_num*2)).map(tokenize_function, batched=True, remove_columns=["text"])

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

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def dump_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

def generate_responses(model,ds):
    response_list = []
    for item in tqdm(ds):
        input_ids = torch.tensor(item['input_ids']).reshape(1,-1).to("cuda")
        input_len = input_ids.shape[1]
        pred = model.generate(input_ids, max_new_tokens=100)
        input_text = tokenizer.decode(pred[0][:input_len], skip_special_tokens=True)
        output_text = tokenizer.decode(pred[0][input_len:], skip_special_tokens=True)
        response_list.append({'output_text':output_text,'input_text':input_text})
    return response_list

def generate_responses(model,ds,temperature,top_p):
    model.eval()
    #print(type(ds[0]))
    #print(ds[0])
    inputs = torch.tensor([item['input_ids'] for item in ds]).to("cuda")
    masks = torch.tensor([item['attention_mask'] for item in ds]).to("cuda")
    num_input,input_len = inputs.shape
    input_text = []
    output_text = []
    bs = 10
    for i in tqdm(range(0,num_input,bs)):
        pred = model.generate(inputs=inputs[i:i+bs], attention_mask=masks[i:i+bs],max_new_tokens=100, temperature=temperature, top_p=top_p).detach()
        input_text += tokenizer.batch_decode(pred[:,:input_len], skip_special_tokens=True)
        output_text += tokenizer.batch_decode(pred[:,input_len:], skip_special_tokens=True)

    return [{'output_text':a,'input_text':b} for a,b in zip(output_text,input_text)]

def run(train_dataset,eval_dataset,log_str, loss_l, args):
    directory_path = f"/home/ubuntu/replication-test-catshift/copyright/output_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/gpt-neo-{args.model}-{args.candidate}-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}"
    numbers = get_num_from_directory(directory_path)
    min_loss_index = loss_l.index(min(loss_l))
    os.makedirs(f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/all_checkpoint', exist_ok=True)
    for num in numbers:
        model_name_hf = f"/home/ubuntu/replication-test-catshift/copyright/output_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/gpt-neo-{args.model}-{args.candidate}-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}/checkpoint-{num}"  # You can choose other sizes as well
        model = AutoModelForCausalLM.from_pretrained(model_name_hf,device_map='auto')
        #model.to(device)
    
        model.eval()
        response_list = generate_responses(model,eval_dataset, args.temp, args.topp)
        if num == numbers[min_loss_index]:
            dump_jsonl(response_list,f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/all_checkpoint/{model_name}-{log_str}-{num}-ft.jsonl')
        else:
            dump_jsonl(response_list,f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/all_checkpoint/{model_name}-{log_str}-{num}-ft.jsonl')


run(A_members,B_members,f'member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}', loss_l, args)
run(A_nonmembers,B_nonmembers,f'nonmember-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}', loss_l, args)