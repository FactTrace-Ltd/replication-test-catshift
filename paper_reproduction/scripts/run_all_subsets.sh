#!/bin/bash

# CatShift - Run full paper on all 17 Pile subsets
# Usage: bash run_all_subsets.sh

MODEL="410m"
EPOCH="9"
SIZE="600"
LR="8e-5"

# All 17 available Pile subsets
SUBSETS=(
    "Wikipedia (en)"
    "PubMed Central"
    "ArXiv"
    "Github"
    "HackerNews"
    "Gutenberg (PG-19)"
    "USPTO Backgrounds"
    "StackExchange"
    "Ubuntu IRC"
    "DM Mathematics"
    "FreeLaw"
    "EuroParl"
    "NIH ExPorter"
    "PhilPapers"
    "Enron Emails"
    "Pile-CC"
    "Pile (remaining)"  # Catch-all for any remaining samples
)

echo "=========================================="
echo "CatShift - Full Paper Replication"
echo "Model: Pythia-${MODEL}"
echo "Epochs: ${EPOCH}"
echo "Fine-tune size: ${SIZE}"
echo "Learning rate: ${LR}"
echo "Subsets: ${#SUBSETS[@]}"
echo "=========================================="

# Track which subsets complete successfully
COMPLETED=()
FAILED=()

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: ${subset}"
    echo "=========================================="
    
    LOG_FILE="results/logs/${subset// /_}_epoch${EPOCH}.log"
    mkdir -p results/logs
    
    # Run training
    python main_pile_subset_saved_model_pythia.py \
        --model $MODEL \
        --epoch $EPOCH \
        --size $SIZE \
        --subname "$subset" \
        --lr $LR \
        > "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${subset}"
        COMPLETED+=("$subset")
        
        # Move results to organized folder
        SAFE_NAME="${subset// /_}" 
        SAFE_NAME="${SAFE_NAME//(/}"
        SAFE_NAME="${SAFE_NAME//)/}"
        SAFE_NAME="${SAFE_NAME,,}"
        
        mkdir -p "results/pile_subsets/${SAFE_NAME}_epoch${EPOCH}"/{outputs,models,responses}
        mv "output_ft_more_layers_${subset}_epoch_${EPOCH}_mlp"* "results/pile_subsets/${SAFE_NAME}_epoch${EPOCH}/outputs/" 2>/dev/null
        mv "model_ft_more_layers_${subset}_epoch_${EPOCH}_mlp"* "results/pile_subsets/${SAFE_NAME}_epoch${EPOCH}/models/" 2>/dev/null
        mv "responses_ft_more_layers_${subset}_epoch_${EPOCH}_mlp"* "results/pile_subsets/${SAFE_NAME}_epoch${EPOCH}/responses/" 2>/dev/null
    else
        echo "✗ Failed: ${subset}"
        FAILED+=("$subset")
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Completed: ${#COMPLETED[@]}/${#SUBSETS[@]}"
for subset in "${COMPLETED[@]}"; do
    echo "  ✓ $subset"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED[@]}"
    for subset in "${FAILED[@]}"; do
        echo "  ✗ $subset"
    done
fi

echo ""
echo "All results organized in results/pile_subsets/"
echo "Next step: Run evaluation with eval_bert_test_all.py for each subset"
