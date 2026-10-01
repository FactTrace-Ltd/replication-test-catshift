# Research Projects

This workspace contains multiple research projects organized in separate folders.

## Folder Structure

### `/paper_reproduction/`
CatShift paper reproduction - Dataset membership inference using catastrophic forgetting.

**Status:** Wikipedia subset training in progress (epoch 9/9)

**Contents:**
- `src/` - Source code for training and evaluation
- `scripts/` - Shell scripts for batch processing
- `data/` - Input datasets (EuroParl, Pile subsets)
- `models/` - Pretrained base models (Pythia-410m)
- `results/` - Training outputs, models, and evaluation results
  - `pile_subsets/` - Results organized by Pile subset
  - `europarl_test/` - EuroParl test run results
  - `logs/` - Training and evaluation logs
- `README.md` - Full documentation

**Quick Start:**
```bash
cd paper_reproduction/
bash scripts/run_all_subsets.sh  # Run all 17 Pile subsets
```

### `/llamat_transfer/`
(Placeholder for next project - instructions to follow)

---

## Next Steps

1. **Complete paper_reproduction:**
   - Wait for Wikipedia run to finish (~1 hour from 20:47 UTC)
   - Run additional Pile subsets or full 17 subsets
   - Generate evaluation metrics and figures

2. **Start llamat_transfer:**
   - Follow instructions provided

---

For detailed documentation, see `paper_reproduction/README.md`
