#!/usr/bin/env bash

# python -m code.detection_ablation \
#     --config_dataset configs/datasets/imagenet/imagenet_ablation_with_res-split-12500.yml \
#     --config_model configs/models/imagenet_timm_vit_tiny16.yml \
#     --config_detection configs/postprocessors/clustering/clustering_imagenet_timm_vit_tiny16_ablation.yml \
#     --root_dir ./results_ablation/imagenet_timm_vit_tiny16_n_cal_with-res-12500/ \
#     --seed 1 \
#     --gpu_id 0 \
#     --latent_dir ./latent/ablation/imagenet_timm-vit-tiny16_n_cal_with-res-12500/ \

python -m code.detection_ablation \
    --config_dataset configs/datasets/imagenet/imagenet_ablation_with_res-split-12500.yml \
    --config_model configs/models/imagenet_timm_vit_tiny16.yml \
    --config_detection configs/postprocessors/clustering/clustering_imagenet_timm_vit_base16_ablation.yml \
    --root_dir ./results_ablation/imagenet_timm_vit_tiny16_n_cal_with-res-12500_v2/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/imagenet_timm-vit-tiny16_n_cal_with-res-12500/ \

