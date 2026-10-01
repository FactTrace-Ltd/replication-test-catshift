"""
Llama 3 Transfer Learning with LoRA Fine-tuning

This script fine-tunes Llama 3 7B using LoRA adapters on a custom dataset.
Adapted from CatShift methodology but for general transfer learning tasks.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from peft import get_peft_model, LoraConfig, TaskType
import json
from tqdm import tqdm
import pandas as pd
from functools import partial
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-3-7b', help='model name')
parser.add_argument('--epoch', type=int, default=3, help='number of epochs')
parser.add_argument('--size', type=int, default=600, help='fine-tune set size')
parser.add_argument('--dataset', type=str, default='wikitext', help='dataset name')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
args = parser.parse_args()

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# ============================================================================
# MODEL SETUP
# ============================================================================

# Llama 3 7B model from HF
model_name_hf = "meta-llama/Llama-2-7b-hf"  # Using Llama 2 as proxy (Llama 3 requires acceptance)
# To use Llama 3: "meta-llama/Llama-3-8B" or "meta-llama/Llama-2-7b-chat-hf"

print(f"Loading {model_name_hf}...")
tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ============================================================================
# DATA LOADING
# ============================================================================

def load_custom_dataset(dataset_name, num_samples=2000):
    """
    Load dataset from HuggingFace.
    Modify this function to load your custom dataset.
    
    TODO: Replace with your actual dataset loading logic
    """
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split='train')
        return dataset.select(range(min(num_samples, len(dataset))))
    elif dataset_name == 'pile':
        from datasets import load_dataset
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        samples = list(dataset.take(num_samples))
        return Dataset.from_list(samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

print(f"Loading {args.dataset} dataset...")
raw_data_df = load_custom_dataset(args.dataset, num_samples=2000)

# Convert to Dataset
if isinstance(raw_data_df, list):
    raw_data = Dataset.from_list(raw_data_df)
else:
    raw_data = raw_data_df

# Split into train/val
split_data = raw_data.train_test_split(test_size=0.2, seed=42)
train_data = split_data['train']
val_data = split_data['test']

# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_function(examples, max_length=512):
    """Tokenize input text"""
    # Handle different dataset structures
    if 'text' in examples:
        text_column = 'text'
    elif 'content' in examples:
        text_column = 'content'
    else:
        # Use first text-like column
        text_columns = [k for k in examples.keys() if isinstance(examples[k], list) and examples[k]]
        if text_columns:
            text_column = text_columns[0]
        else:
            raise ValueError(f"No text column found. Columns: {examples.keys()}")
    
    tokens = tokenizer(
        examples[text_column], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing datasets...")
train_dataset = train_data.map(
    partial(tokenize_function, max_length=512), 
    batched=True, 
    remove_columns=[col for col in train_data.column_names if col != 'text' and col != 'content']
)
val_dataset = val_data.map(
    partial(tokenize_function, max_length=512), 
    batched=True, 
    remove_columns=[col for col in val_data.column_names if col != 'text' and col != 'content']
)

# For evaluation, use a subset
eval_dataset = train_dataset.select(range(min(1000, len(train_dataset))))

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# ============================================================================
# MODEL SETUP WITH LORA
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name_hf,
    device_map='auto',
    torch_dtype=torch.float16,
)

# Configure LoRA for Llama
# Llama uses: q_proj, k_proj, v_proj, o_proj (attention)
#             gate_proj, up_proj, down_proj (feedforward)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                              # LoRA rank
    lora_alpha=32,                     # LoRA scaling
    lora_dropout=0.1,                  # Dropout
    bias="none",
    target_modules=["q_proj", "v_proj"],  # Llama attention projections
)

print("Applying LoRA...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============================================================================
# TRAINING
# ============================================================================

output_dir = f"./output_llamat_{args.dataset}_epoch_{args.epoch}"
model_dir = f"./model_llamat_{args.dataset}_epoch_{args.epoch}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=50,
    learning_rate=args.lr,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=args.epoch,
    weight_decay=0.01,
    warmup_ratio=0.05,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# Save the fine-tuned model
print(f"Saving model to {model_dir}...")
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Evaluation
print("Evaluating...")
results = trainer.evaluate()
print("Evaluation results:")
for key, value in results.items():
    print(f"  {key}: {value}")

# Save results
results_file = f"./results_llamat_{args.dataset}_epoch_{args.epoch}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_file}")

print("✓ Transfer learning complete!")
