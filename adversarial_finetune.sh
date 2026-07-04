for priv in age_range gender_numeric; do

    if [ "$priv" == "age_range" ]; then
        NB_PRIV=3
    else
        NB_PRIV=2
    fi

    # ---- ODIR ----
    python adversary_finetun.py \
        --data_path ../preprocessed_datasets/odir \
        --csv_path ../attribute_inference_csv/clean_odir_labels_for_classification.csv \
        --label_col label \
        --private_label_col "$priv" \
        --nb_classes 8 \
        --nb_private_classes "$NB_PRIV" \
        --finetune RETFound_mae_natureCFP \
        --ac_weights checkpoints/mlp/odir_${priv}_128_64.pth \
        --c_weights odir_best_linear_head_no_finetune.pth \
        --hidden1 128 --hidden2 64 \
        --save_prefix finetuned_odir_${priv}

    # ---- GRAPE ----
    python adversary_finetun.py \
        --data_path ../preprocessed_datasets/grape_cfp \
        --csv_path ../attribute_inference_csv/grape_multiclass_labels.csv \
        --label_col label \
        --private_label_col "$priv" \
        --nb_classes <GRAPE_TASK_CLASSES> \
        --nb_private_classes "$NB_PRIV" \
        --finetune RETFound_mae_natureCFP \
        --ac_weights checkpoints/mlp/grape_cfp_${priv}_256_128.pth \
        --c_weights grape_cfp_best_linear_head_no_finetune.pth \
        --hidden1 256 --hidden2 128 \
        --save_prefix finetuned_grape_cfp_${priv}
done
