#!/usr/bin/env bash

python -m code.detection_ablation \
    --config_dataset configs/datasets/imagenet/imagenet_ablation.yml \
    --config_model configs/models/imagenet_timm-vit-base16.yml \
    --config_detection configs/postprocessors/clustering.yml \
    --root_dir ./results_ablation/imagenet_timm-vit-base16_n_cal/ \
    --seed 1 \
    --gpu_id 1 \
    --latent_dir ./latent/ablation/imagenet_timm-vit-base16_n_cal/ 

