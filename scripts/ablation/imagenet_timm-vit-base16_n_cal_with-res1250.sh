#!/usr/bin/env bash

python -m code.detection_ablation \
    --config_dataset configs/datasets/imagenet/imagenet_ablation_with_res-split-1250.yml \
    --config_model configs/models/imagenet_timm_vit_base16.yml \
    --config_detection configs/postprocessors/clustering/clustering_imagenet_timm_vit_base16_ablation.yml \
    --root_dir ./results_ablation/imagenet_timm_vit_base16_n_cal_with-res-1250/ \
    --seed 1 \
    --gpu_id 1 \
    --latent_dir ./latent/ablation/imagenet_timm-vit-base16_n_cal_with-res-1250/ \

