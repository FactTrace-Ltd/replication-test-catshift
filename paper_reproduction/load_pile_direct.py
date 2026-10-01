import os
import pandas as pd
from datasets import load_dataset

def get_pile_subset(subset_name, split_type="train", num_samples=1000):
    """
    Load samples from the Pile without saving to disk.
    
    Args:
        subset_name: Name of the subset (e.g., "Wikipedia (en)")
        split_type: "train" or "validation"
        num_samples: Number of samples to load
    
    Returns:
        pandas DataFrame with the requested samples
    """
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    
    # Filter by subset
    filtered = dataset.filter(lambda x: x["meta"]["pile_set_name"] == subset_name)
    
    # Take samples
    samples = list(filtered.take(num_samples))
    
    return pd.DataFrame(samples)

# Example usage - load Wikipedia subset without saving to disk
if __name__ == "__main__":
    subset = "Wikipedia (en)"
    
    print(f"Loading {subset}...")
    train_finetune = get_pile_subset(subset, num_samples=600)
    train_eval = get_pile_subset(subset, num_samples=1000)
    validation = get_pile_subset(subset, num_samples=1000)
    
    print(f"✓ Loaded {subset}:")
    print(f"  - Fine-tuning: {len(train_finetune)} samples")
    print(f"  - Train evaluation: {len(train_eval)} samples")
    print(f"  - Validation: {len(validation)} samples")
