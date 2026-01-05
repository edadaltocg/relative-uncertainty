#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/imagenet/imagenet_n_cal-25000_seed-split-9.yml \
    --config_model configs/models/imagenet_timm-vit-base16.yml \
    --config_detection configs/postprocessors/conformal/imagenet_timm-vit-base16.yml \
    --root_dir ./results_main/conformal/imagenet_timm-vit-base16/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/imagenet_timm-vit-base16/ \
    --mode evaluation

