import os
import pandas as pd
from datasets import load_dataset

os.makedirs("data_inference", exist_ok=True)

print("Loading Pile data (using two independent streams)...")
# Use two independent streams from the same dataset - they will iterate independently
# Stream 1: for training data (1600 samples)
train_dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
# Stream 2: for validation data (1000 samples) - separate iterator, different samples
validation_dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# The 17 available subsets matching your metadata query
pile_subsets = [
    'PubMed Abstracts', 'Pile-CC', 'PhilPapers', 'Enron Emails', 
    'HackerNews', 'Gutenberg (PG-19)', 'USPTO Backgrounds', 'ArXiv', 
    'StackExchange', 'Github', 'Ubuntu IRC', 'DM Mathematics', 
    'FreeLaw', 'EuroParl', 'NIH ExPorter', 'PubMed Central', 'Wikipedia (en)'
]

for subset in pile_subsets:
    print(f"Processing {subset}...")
    
    # Filter the distinct streams
    train_filtered = train_dataset.filter(lambda x: x["meta"]["pile_set_name"] == subset)
    val_filtered = validation_dataset.filter(lambda x: x["meta"]["pile_set_name"] == subset)
    
    # Extract 1600 training samples and 1000 validation samples
    train_samples = list(train_filtered.take(1600))
    val_samples = list(val_filtered.take(1000))
    
    # ... (The rest of your dataframe splitting and saving code remains exactly the same)
    
    # Ensure you actually got enough samples
    if len(train_samples) < 1600 or len(val_samples) < 1000:
        print(f"  Warning: {subset} did not have enough samples. Got {len(train_samples)} train, {len(val_samples)} val.")
        continue
    
    # Split the training data to guarantee disjoint sets for fine-tuning and evaluation
    train_finetune_df = pd.DataFrame(train_samples[:600])
    train_eval_df = pd.DataFrame(train_samples[600:])
    val_df = pd.DataFrame(val_samples)
    
    # Format the subset name so it's safe for a file path
    safe_name = subset.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
    
    # Save the files
    train_finetune_df.to_json(f"data_inference/{safe_name}_finetune.jsonl", orient="records", lines=True)
    train_eval_df.to_json(f"data_inference/{safe_name}_train_eval.jsonl", orient="records", lines=True)
    val_df.to_json(f"data_inference/{safe_name}_val.jsonl", orient="records", lines=True)
    
    print(f"  ✓ Saved {subset}: 600 finetune, 1000 train eval, 1000 val")

print("\n✓ All datasets extracted and saved!")
