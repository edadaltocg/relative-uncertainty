#!/usr/bin/env bash

python -m code.detection_ablation \
    --config_dataset configs/datasets/imagenet/imagenet_ablation_with_res-split-10000.yml \
    --config_model configs/models/imagenet_timm_vit_tiny16.yml \
    --config_detection configs/postprocessors/clustering/clustering_imagenet_timm_vit_tiny16_ablation_fair.yml \
    --root_dir ./results_ablation/imagenet_timm_vit_tiny16_n_cal_with-res-10000_fair/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/imagenet_timm-vit-tiny16_n_cal_with-res-10000/ \


