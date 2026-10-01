#!/bin/bash

# CatShift - Evaluate all subsets and aggregate results
# Usage: bash run_all_eval.sh

MODEL="410m"
EPOCH="9"
SIZE="600"
LR="8e-5"
LOGGING="mlp"

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
)

echo "=========================================="
echo "CatShift - Evaluation (All Subsets)"
echo "Model: Pythia-${MODEL}"
echo "=========================================="

mkdir -p results/eval_results

# Aggregate results
AUC_SCORES=()
COMPLETED=0

for subset in "${SUBSETS[@]}"; do
    echo ""
    echo "Evaluating: ${subset}"
    
    # Run evaluation
    python eval_bert_test_all.py \
        --model $MODEL \
        --epoch $EPOCH \
        --size $SIZE \
        --subname "$subset" \
        --logging $LOGGING \
        --lr $LR 2>&1 | tee "results/eval_results/${subset// /_}_eval.log"
    
    if [ $? -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "✓ Evaluated: ${subset}"
    else
        echo "✗ Evaluation failed: ${subset}"
    fi
done

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "Completed: ${COMPLETED}/${#SUBSETS[@]}"
echo "Results saved to: results/eval_results/"
echo "=========================================="
