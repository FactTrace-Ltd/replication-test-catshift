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
parser = argparse.ArgumentParser()

# LLM settings
parser.add_argument('--model', type=str, default='1.4b',help='model name') #160m 410m 1b 1.4b 2.8b 6.9b 12b
parser.add_argument('--epoch', type=int, default=3,help='model name') #160m 410m 1b 1.4b 2.8b 6.9b 12b
parser.add_argument('--size', type=int, default=100,help='split size')
parser.add_argument('--subname', type=str, default='wikipedia', help='subset name')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
args = parser.parse_args()

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

model_name = f'pythia-{args.model}'
# Load the tokenizer and model
model_name_hf = f"./pythia-{args.model}"  # Local model path (relative to this script)
tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
tokenizer.padding_side = "left"
# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# process data - Load directly from HF Pile without saving to disk
def load_pile_subset(subset_name, num_samples=2000):
    """Load samples from the Pile streaming dataset"""
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    filtered = dataset.filter(lambda x: x["meta"]["pile_set_name"] == subset_name)
    samples = list(filtered.take(num_samples))
    return pd.DataFrame(samples)

print(f"Loading {args.subname} subset from Pile...")
raw_train_data_df = load_pile_subset(args.subname, num_samples=2000)  # Need 2000 for all splits
raw_val_data_df = load_pile_subset(args.subname, num_samples=2000)    # Need 2000 for all splits

tds = Dataset.from_pandas(raw_train_data_df)
vds = Dataset.from_pandas(raw_val_data_df)

raw_data = DatasetDict()
raw_data['train'] = tds
raw_data['validation'] = vds


# Tokenize the input data
def tokenize_function(examples,max_length=384):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)
    #tokens["labels"] = tokens["input_ids"].copy()
    return tokens

data_num = 1000
A_members = raw_data['train'].shuffle(seed=42).select(range(0, args.size)).map(partial(tokenize_function,max_length=512), batched=True, remove_columns=["text"])
A_nonmembers = raw_data['validation'].shuffle(seed=42).select(range(0, args.size)).map(partial(tokenize_function,max_length=512), batched=True, remove_columns=["text"])

B_members = raw_data['train'].shuffle(seed=42).select(range(data_num, data_num*2)).map(tokenize_function, batched=True, remove_columns=["text"])
B_nonmembers = raw_data['validation'].shuffle(seed=42).select(range(data_num, data_num*2)).map(tokenize_function, batched=True, remove_columns=["text"])
'''
model = AutoModelForCausalLM.from_pretrained(model_name_hf)
input_ids = torch.tensor(B_members[0]["input_ids"]).reshape(1,-1)
input_len = input_ids.shape[1]
output = model.generate(input_ids, max_new_tokens =128)
print('!!!!!!!!!!!!!!!!inputs',input_len)
print(tokenizer.decode(output[0][:input_len], skip_special_tokens=True))
print('!!!!!!!!!!!!!!!!outputs',len(output[0])-input_len)
print(tokenizer.decode(output[0][input_len:], skip_special_tokens=True))
exit(0)
'''
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

def generate_responses(model,ds):
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
        pred = model.generate(inputs=inputs[i:i+bs], attention_mask=masks[i:i+bs],max_new_tokens=100, temperature=0.0, top_p=1.0).detach()
        input_text += tokenizer.batch_decode(pred[:,:input_len], skip_special_tokens=True)
        output_text += tokenizer.batch_decode(pred[:,input_len:], skip_special_tokens=True)

    return [{'output_text':a,'input_text':b} for a,b in zip(output_text,input_text)]

# Define a data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(train_dataset,eval_dataset,log_str, args):
    model = AutoModelForCausalLM.from_pretrained(model_name_hf,device_map='auto')
    #model.to(device)
    model.eval()
    os.makedirs(f'output_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp',exist_ok=True)
    os.makedirs(f'model_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp',exist_ok=True)
    os.makedirs(f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp',exist_ok=True)
    response_list = generate_responses(model,eval_dataset)
    dump_jsonl(response_list,f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-{log_str}-orig.jsonl')

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Define training arguments with mixed precision
    training_args = TrainingArguments(
        output_dir=f"./output_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-{log_str}",
        evaluation_strategy="steps",
        learning_rate=args.lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        logging_dir='./logs',  # Directory for storing logs
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        fp16=True,  # Enable mixed precision training
        load_best_model_at_end=True,
    )

    # Ensure the model and datasets are on the same device


    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(f"./model_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-{log_str}")

    # Evaluate the model
    results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in results.items():
        print(f"{key}: {value}")

    model.eval()
    response_list = generate_responses(model,eval_dataset)
    dump_jsonl(response_list,f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-{log_str}-ft.jsonl')


run(A_members,B_members,f'member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}', args)
run(A_nonmembers,B_nonmembers,f'nonmember-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}', args)