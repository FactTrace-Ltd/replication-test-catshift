# LLaMA Transfer Learning: Academic Paper Auditing

Fine-tune LLaMA 2 7B on suspect academic papers from the LLaMat Auditing Dataset, then detect membership inference attacks using CatShift methodology.

## Overview

This project adapts the **CatShift membership inference attack** to detect if a language model has been fine-tuned on specific academic research papers:

- **Suspect papers** = Member data (fine-tuning set)
- **Control papers** = Non-member data (evaluation baseline)
- **Goal**: Detect if model memorized/specializes in suspect papers via text analysis

## Dataset

**LLaMat Auditing Dataset** (from S3: `s3://llamat-auditing-dataset`)
- 816 suspect vs control paper pairs
- Full JATS XML format (title + abstract + body)
- Metadata: DOI, journal, timestamp

Default configuration:
- **600 suspect papers** for fine-tuning
- **200 suspect papers** for evaluation (held-out)
- **200 control papers** for evaluation (baseline)

## Installation

```bash
# Install dependencies
pip install torch transformers peft datasets tqdm boto3

# Or from requirements
pip install -r requirements.txt
```

AWS credentials must be configured:
```bash
aws configure
```

## Usage

### Basic Fine-tuning

```bash
cd /home/ubuntu/replication-test-catshift/llamat_transfer
python llamat_transfer.py --epoch 3 --lr 2e-4 --batch_size 8
```

### Custom Configuration

```bash
python llamat_transfer.py \
  --epoch 5 \
  --suspect_size 400 \
  --control_eval_size 100 \
  --suspect_eval_size 100 \
  --lr 1e-4 \
  --batch_size 4
```

**Arguments:**
- `--epoch`: Training epochs (default: 3)
- `--suspect_size`: Suspect papers for fine-tuning (default: 600)
- `--control_eval_size`: Control papers for evaluation (default: 200)
- `--suspect_eval_size`: Suspect papers for evaluation (default: 200)
- `--lr`: Learning rate (default: 2e-4)
- `--batch_size`: Batch size (default: 8)
- `--model`: Model name (default: llama-2-7b)

## Methodology

### 1. Fine-tuning Strategy
- Train ONE model on suspect papers only
- Use **combined eval loss** (suspect + control) for checkpoint selection
- Per paper: "model with lowest normalized combined loss (on both member and non-member sets)"

### 2. LoRA Configuration
- Rank: 8
- Alpha: 32
- Dropout: 0.1
- Target modules: `q_proj`, `v_proj` (Llama attention projections)

### 3. Evaluation Setup
- **Training set**: 600 suspect papers
- **Eval set**: 200 suspect + 200 control papers (combined)
- **Mixed precision**: FP16 for efficient training

## Output Files

Training produces:
- `model_llamat_auditing_YYYYMMDD_HHMMSS/` - Fine-tuned model weights
- `results_llamat_auditing_YYYYMMDD_HHMMSS.json` - Training results
- `output_llamat_auditing_YYYYMMDD_HHMMSS/` - Checkpoints
- `logs/` - Training logs

## Next Steps: Membership Inference

After fine-tuning, run BERTScore analysis:

1. Generate responses from original LLaMA on suspect + control papers
2. Generate responses from fine-tuned model on suspect + control papers  
3. Compare semantic similarity using BERTScore
4. Detect membership via ROC-AUC

Expected result:
- Suspect papers (members): **LOW similarity** (catastrophic forgetting)
- Control papers (non-members): **HIGH similarity** (unchanged)
- ROC-AUC > 0.9 indicates successful membership detection

## References

- CatShift paper: Membership inference via catastrophic forgetting
- LLaMA: Meta's large language model
- PEFT/LoRA: Parameter-efficient fine-tuning

## License

This work is based on the CatShift methodology and uses the LLaMat Auditing Dataset.
