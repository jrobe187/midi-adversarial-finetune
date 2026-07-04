#!/bin/bash

DATASET="odir_preprocessed"
CSV_PATH="../attribute_inference_csv/clean_odir_labels_for_classification.csv"
EMBEDDING_DIR="/home/jrobe187/midi/embeddings/${DATASET}"
LINEAR_NB_CLASSES=2

python extract_features.py \
    --input_dir "/home/jrobe187/midi/preprocessed_datasets/${DATASET}" \
    --output_dir "$EMBEDDING_DIR"

for label in age_range gender_numeric; do

    if [ "$label" == "age_range" ]; then
        MLP_NB_CLASSES=3
    elif [ "$label" == "gender_numeric" ]; then
        MLP_NB_CLASSES=2
    fi

    for HIDDEN1 in 512 256 128; do
        HIDDEN2=$((HIDDEN1 / 2))
        echo "Running hidden1=${HIDDEN1}, hidden2=${HIDDEN2} with ${label} for label"
        python mlp.py \
            --embedding_dir "$EMBEDDING_DIR" \
            --csv_path "$CSV_PATH" \
            --label_col "$label" \
            --nb_classes "$MLP_NB_CLASSES" \
            --output_path "checkpoints/mlp/${DATASET}_${label}_${HIDDEN1}_${HIDDEN2}.pth" \
            --model_type "mlp" \
            --hidden1 "$HIDDEN1" \
            --hidden2 "$HIDDEN2"
    done

done

python mlp.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --csv_path "$CSV_PATH" \
    --label_col "label" \
    --nb_classes $LINEAR_NB_CLASSES \
    --output_path "checkpoints/linear/${DATASET}_best_linear_head_no_finetune.pth" \
    --model_type "linear"
