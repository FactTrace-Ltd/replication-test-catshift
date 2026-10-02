"""
LLaMA Transfer Learning: Academic Paper Auditing with CatShift Methodology

This script adapts the CatShift membership inference attack to detect if a language model
has been fine-tuned on specific academic papers (suspect papers).

Methodology:
1. Fine-tune LLaMA on suspect papers (member data)
2. Evaluate on both suspect and control papers (non-member data)
3. Use BERTScore to compare original vs fine-tuned responses
4. Detect membership via catastrophic forgetting on suspect papers

Dataset: LLaMat Auditing Dataset from S3
- Suspect papers: Fine-tuning set (600 papers)
- Control papers: Evaluation baseline (200 papers)
- Evaluation suspect: Held-out test set (200 papers)
"""

import os
import json
import boto3
import xml.etree.ElementTree as ET
from io import BytesIO
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset, DatasetDict, concatenate_datasets
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
import pandas as pd
from functools import partial
import argparse
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-7b', help='Model name')
parser.add_argument('--epoch', type=int, default=3, help='Number of epochs')
parser.add_argument('--suspect_size', type=int, default=600, help='Suspect papers for fine-tuning')
parser.add_argument('--control_eval_size', type=int, default=200, help='Control papers for evaluation')
parser.add_argument('--suspect_eval_size', type=int, default=200, help='Suspect papers for evaluation')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
args = parser.parse_args()

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# ============================================================================
# S3 CONFIGURATION
# ============================================================================

S3_BUCKET = "llamat-auditing-dataset"
S3_DATASET_PREFIX = "llamat-oa-suspect-vs-unseen-jats-complete-pairs-dedup-20260926"
S3_METADATA_SUSPECT = f"s3://{S3_BUCKET}/{S3_DATASET_PREFIX}/metadata/suspect"
S3_METADATA_CONTROL = f"s3://{S3_BUCKET}/{S3_DATASET_PREFIX}/metadata/control"
S3_XML_SUSPECT = f"s3://{S3_BUCKET}/{S3_DATASET_PREFIX}/raw_xml/suspect"
S3_XML_CONTROL = f"s3://{S3_BUCKET}/{S3_DATASET_PREFIX}/raw_xml/control"

s3_client = boto3.client('s3')

# ============================================================================
# XML PARSING UTILITIES
# ============================================================================

def parse_jats_xml(xml_content):
    """
    Extract text from JATS XML format (common for academic papers).
    Extracts: title, abstract, and body sections.
    """
    try:
        root = ET.fromstring(xml_content)
        text_parts = []
        
        # Define JATS namespaces
        namespaces = {
            'jats': 'http://jats.nlm.nih.gov/publishing/1.2/',
            '': 'http://jats.nlm.nih.gov/publishing/1.2/'
        }
        
        # Extract title
        title = root.find('.//article-title') or root.find('.//title')
        if title is not None and title.text:
            text_parts.append(title.text.strip())
        
        # Extract abstract
        abstract = root.find('.//abstract')
        if abstract is not None:
            abstract_text = ' '.join([p.text for p in abstract.findall('.//p') if p.text])
            if not abstract_text:
                abstract_text = abstract.text
            if abstract_text:
                text_parts.append(abstract_text.strip())
        
        # Extract body sections (introduction, methods, results, discussion, conclusion)
        body = root.find('.//body')
        if body is not None:
            for sec in body.findall('.//sec'):
                for p in sec.findall('.//p'):
                    if p.text:
                        text_parts.append(p.text.strip())
        
        combined_text = ' '.join(text_parts)
        return combined_text if combined_text else None
        
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None

def download_s3_json(bucket, key):
    """Download JSON metadata from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return None

def download_s3_file(bucket, key):
    """Download file from S3 as bytes"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return None

# ============================================================================
# DATA LOADING
# ============================================================================

def load_papers_from_s3(group='suspect', num_papers=None):
    """
    Load papers from S3.
    group: 'suspect' or 'control'
    Returns: list of (doc_id, title, text, group)
    """
    print(f"Loading {group} papers from S3...")
    
    # List metadata files
    prefix = f"{S3_DATASET_PREFIX}/metadata/{group}"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    
    papers = []
    
    if 'Contents' not in response:
        print(f"No {group} papers found!")
        return papers
    
    metadata_keys = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')]
    if num_papers:
        metadata_keys = metadata_keys[:num_papers]
    
    print(f"Found {len(metadata_keys)} {group} metadata files")
    
    for i, metadata_key in enumerate(tqdm(metadata_keys)):
        # Download metadata
        metadata = download_s3_json(S3_BUCKET, metadata_key)
        if not metadata:
            continue
        
        # Get XML path
        xml_path = metadata.get('xml_path', '').replace('datasets/', '')
        if not xml_path:
            continue
        
        # Download XML
        xml_content = download_s3_file(S3_BUCKET, xml_path)
        if not xml_content:
            continue
        
        # Parse XML to extract text
        text = parse_jats_xml(xml_content)
        if not text or len(text.split()) < 50:  # Skip if too short
            continue
        
        papers.append({
            'doc_id': metadata.get('doc_id'),
            'title': metadata.get('title', 'Unknown'),
            'doi': metadata.get('doi'),
            'journal': metadata.get('journal'),
            'text': text,
            'group': group
        })
    
    print(f"Loaded {len(papers)} valid {group} papers")
    return papers

# ============================================================================
# MODEL SETUP
# ============================================================================

print(f"Loading {args.model}...")
model_name_hf = "meta-llama/Llama-2-7b-hf"
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

# ============================================================================
# LOAD DATASET FROM S3
# ============================================================================

print("Loading LLaMat Auditing Dataset from S3...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load suspect papers
suspect_papers = load_papers_from_s3('suspect', num_papers=args.suspect_size + args.suspect_eval_size)
if len(suspect_papers) < args.suspect_size + args.suspect_eval_size:
    print(f"Warning: Expected {args.suspect_size + args.suspect_eval_size} suspect papers, got {len(suspect_papers)}")

# Load control papers
control_papers = load_papers_from_s3('control', num_papers=args.control_eval_size)
if len(control_papers) < args.control_eval_size:
    print(f"Warning: Expected {args.control_eval_size} control papers, got {len(control_papers)}")

# Split suspect papers
suspect_train = suspect_papers[:args.suspect_size]
suspect_eval = suspect_papers[args.suspect_size:args.suspect_size + args.suspect_eval_size]

print(f"\n{'='*60}")
print(f"Dataset Splits:")
print(f"  Suspect (training): {len(suspect_train)}")
print(f"  Suspect (evaluation): {len(suspect_eval)}")
print(f"  Control (evaluation): {len(control_papers)}")
print(f"{'='*60}\n")


# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_function(examples, max_length=512):
    """Tokenize paper text"""
    tokens = tokenizer(
        examples['text'],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Convert to HF Dataset
train_data = Dataset.from_list(suspect_train)
suspect_eval_data = Dataset.from_list(suspect_eval)
control_eval_data = Dataset.from_list(control_papers)

print("Tokenizing datasets...")
train_dataset = train_data.map(
    partial(tokenize_function, max_length=512),
    batched=True,
    remove_columns=['doc_id', 'title', 'doi', 'journal', 'text', 'group']
)

suspect_eval_dataset = suspect_eval_data.map(
    partial(tokenize_function, max_length=512),
    batched=True,
    remove_columns=['doc_id', 'title', 'doi', 'journal', 'text', 'group']
)

control_eval_dataset = control_eval_data.map(
    partial(tokenize_function, max_length=512),
    batched=True,
    remove_columns=['doc_id', 'title', 'doi', 'journal', 'text', 'group']
)

# Combined eval datasets (per CatShift methodology)
combined_eval_dataset = concatenate_datasets([suspect_eval_dataset, control_eval_dataset])
print(f"Combined eval dataset size: {len(combined_eval_dataset)}")

# ============================================================================
# LORA CONFIGURATION
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name_hf,
    device_map='auto',
    torch_dtype=torch.float16,
)

# Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
)

print("Applying LoRA...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ============================================================================
# RESPONSE GENERATION FUNCTION
# ============================================================================

def generate_responses(model, ds, temperature=0.0, top_p=1.0):
    """Generate responses using batch decoding (matches original CatShift code)"""
    model.eval()
    inputs = torch.tensor([item['input_ids'] for item in ds]).to("cuda")
    masks = torch.tensor([item['attention_mask'] for item in ds]).to("cuda")
    num_input, input_len = inputs.shape
    input_text = []
    output_text = []
    bs = 10
    
    for i in tqdm(range(0, num_input, bs)):
        pred = model.generate(
            inputs=inputs[i:i+bs], 
            attention_mask=masks[i:i+bs],
            max_new_tokens=100, 
            temperature=temperature, 
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,  # Suppress temperature warning
        ).detach()
        
        # Use batch_decode like original code (handles padding better)
        input_text += tokenizer.batch_decode(pred[:, :input_len], skip_special_tokens=True)
        output_text += tokenizer.batch_decode(pred[:, input_len:], skip_special_tokens=True)

    return [{'output_text':a,'input_text':b} for a,b in zip(output_text,input_text)]

def dump_jsonl(data, file_path):
    """Save data to JSONL format"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')

# ============================================================================
# TRAINING
# ============================================================================

output_dir = f"./output_llamat_auditing_{timestamp}"
model_dir = f"./model_llamat_auditing_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=50,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
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
    eval_dataset=combined_eval_dataset,  # Combined per CatShift
    data_collator=data_collator,
)

print(f"{'='*60}")
print(f"Starting Fine-tuning on Suspect Papers")
print(f"{'='*60}\n")

# Generate responses from ORIGINAL model on BOTH suspect and control eval data
print("Generating responses from original model...")
suspect_eval_list = [{'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']} for item in suspect_eval_dataset]
control_eval_list = [{'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']} for item in control_eval_dataset]

suspect_response_orig = generate_responses(model, suspect_eval_list, temperature=0.0, top_p=1.0)
control_response_orig = generate_responses(model, control_eval_list, temperature=0.0, top_p=1.0)

responses_dir = f"./responses_llamat_auditing_{timestamp}"
dump_jsonl(suspect_response_orig, f"{responses_dir}/suspect-orig.jsonl")
dump_jsonl(control_response_orig, f"{responses_dir}/control-orig.jsonl")

# Start training
trainer.train()

# Save the fine-tuned model
print(f"\nSaving model to {model_dir}...")
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Generate responses from FINE-TUNED model
print("Generating responses from fine-tuned model...")
model.eval()
suspect_response_ft = generate_responses(model, suspect_eval_list, temperature=0.0, top_p=1.0)
control_response_ft = generate_responses(model, control_eval_list, temperature=0.0, top_p=1.0)

dump_jsonl(suspect_response_ft, f"{responses_dir}/suspect-ft.jsonl")
dump_jsonl(control_response_ft, f"{responses_dir}/control-ft.jsonl")

# Evaluation
print("Evaluating on combined dataset...")
results = trainer.evaluate()
print("Evaluation results:")
for key, value in results.items():
    print(f"  {key}: {value}")

# Save results
results_file = f"./results_llamat_auditing_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump({
        'timestamp': timestamp,
        'config': {
            'suspect_train_size': len(suspect_train),
            'suspect_eval_size': len(suspect_eval),
            'control_eval_size': len(control_papers),
            'epochs': args.epoch,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
        },
        'results': results
    }, f, indent=2)

print(f"\nResults saved to {results_file}")
print(f"Model saved to {model_dir}")
print(f"\n✓ Fine-tuning complete!")
print(f"Next: Run membership inference attack using BERTScore comparison")
