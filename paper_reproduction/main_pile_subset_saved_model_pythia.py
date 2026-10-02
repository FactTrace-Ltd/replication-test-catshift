import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import datasets
from datasets import load_dataset
from datasets import Dataset, DatasetDict, concatenate_datasets
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
# Use absolute path to local model - check current directory first, then parent
script_dir = os.path.dirname(os.path.abspath(__file__))
model_in_script_dir = os.path.join(script_dir, f"pythia-{args.model}")
parent_dir = os.path.dirname(script_dir)
model_in_parent_dir = os.path.join(parent_dir, f"pythia-{args.model}")

if os.path.exists(model_in_script_dir):
    model_name_hf = model_in_script_dir
elif os.path.exists(model_in_parent_dir):
    model_name_hf = model_in_parent_dir
else:
    raise FileNotFoundError(f"Model not found in {model_in_script_dir} or {model_in_parent_dir}")

tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
tokenizer.padding_side = "left"

# Add proper padding token to avoid EOS-as-PAD issues during generation
if tokenizer.pad_token is None:
    # Try to use a special token first, or create one
    if tokenizer.unk_token_id is not None:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
    else:
        # Last resort: add a new pad token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Suppress the warning by directly setting the pad token
    tokenizer.padding_side = "left"

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
# DEBUG TEST COMPLETE - continue with main training

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

def generate_responses(model, ds, temperature=0.0, top_p=1.0):
    """Generate responses - process one at a time to handle variable input lengths correctly"""
    response_list = []
    model.eval()
    
    for item in tqdm(ds):
        input_ids = torch.tensor(item['input_ids']).reshape(1, -1).to("cuda")
        attention_mask = torch.tensor(item['attention_mask']).reshape(1, -1).to("cuda")  # Move to cuda
        actual_input_len = (attention_mask == 1).sum().item()  # Count non-padding tokens
        
        pred = model.generate(
            input_ids, 
            attention_mask=attention_mask,  # Pass attention mask explicitly
            max_new_tokens=100, 
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False  # Greedy decoding - temperature/top_p ignored
        ).detach()
        
        # Slice using actual input length (not padded length)
        input_text = tokenizer.decode(pred[0][:actual_input_len], skip_special_tokens=True)
        output_text = tokenizer.decode(pred[0][actual_input_len:], skip_special_tokens=True)
        response_list.append({'output_text': output_text, 'input_text': input_text})
    
    return response_list


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


def run(train_dataset, member_eval_dataset, nonmember_eval_dataset, log_str, args):
    """
    CatShift membership inference attack.
    Trains ONE model on member data, evaluates on both member and non-member data.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_hf,device_map='auto')
    model.eval()
    os.makedirs(f'output_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp',exist_ok=True)
    os.makedirs(f'model_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp',exist_ok=True)
    os.makedirs(f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp',exist_ok=True)
    
    # Generate responses from ORIGINAL model on BOTH member and non-member data
    print(f"Generating responses from original model...")
    member_response_list = generate_responses(model, member_eval_dataset, temperature=0.0, top_p=1.0)
    dump_jsonl(member_response_list, f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-orig.jsonl')
    
    nonmember_response_list = generate_responses(model, nonmember_eval_dataset, temperature=0.0, top_p=1.0)
    dump_jsonl(nonmember_response_list, f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-nonmember-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-orig.jsonl')

    
    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)

    # Combine member and non-member eval datasets
    # Per paper: "model with lowest normalized combined loss (on both member and non-member sets)"
    combined_eval_dataset = concatenate_datasets([member_eval_dataset, nonmember_eval_dataset])
    print(f"Combined eval dataset size: {len(combined_eval_dataset)} (member: {len(member_eval_dataset)}, non-member: {len(nonmember_eval_dataset)})")

    # Define training arguments with mixed precision
    training_args = TrainingArguments(
        output_dir=f"./output_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}",
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
        metric_for_best_model="eval_loss",  # Track combined eval loss
        greater_is_better=False,  # Lower loss is better
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=combined_eval_dataset,  # Use COMBINED eval dataset per paper
        data_collator=data_collator,
    )

    # Train the model on member data
    print(f"Training model on member data...")
    trainer.train()

    # Save the model
    trainer.save_model(f"./model_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}")

    # Evaluate the model
    results = trainer.evaluate()
    print("Evaluation results on member data:")
    for key, value in results.items():
        print(f"{key}: {value}")

    # Generate responses from FINE-TUNED model on BOTH member and non-member data
    print(f"Generating responses from fine-tuned model on member data...")
    model.eval()
    member_response_list = generate_responses(model, member_eval_dataset, temperature=0.0, top_p=1.0)
    dump_jsonl(member_response_list, f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-ft.jsonl')
    
    print(f"Generating responses from fine-tuned model on non-member data...")
    nonmember_response_list = generate_responses(model, nonmember_eval_dataset, temperature=0.0, top_p=1.0)
    dump_jsonl(nonmember_response_list, f'responses_ft_more_layers_{args.subname}_epoch_{args.epoch}_mlp/{model_name}-nonmember-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}-ft.jsonl')


run(A_members, B_members, B_nonmembers, f'member-{args.model}-epoch-{args.epoch}-pile-full-{args.size}-subsets-{args.subname}-{args.lr}', args)