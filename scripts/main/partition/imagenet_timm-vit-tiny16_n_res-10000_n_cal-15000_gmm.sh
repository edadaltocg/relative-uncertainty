#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/imagenet/imagenet_n_res-10000_seed-split-9.yml \
    --config_model configs/models/imagenet_timm_vit_tiny16.yml \
    --config_detection configs/postprocessors/clustering/imagenet_timm-vit-tiny16_gmm.yml \
    --root_dir ./results_main/partition_gmm/imagenet_timm-vit-tiny16 \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/imagenet_timm-vit-tiny16/ \
    --metric roc_auc \
    --mode search




