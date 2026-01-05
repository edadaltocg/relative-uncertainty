#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/imagenet/imagenet_n-cal-25000.yml \
    --config_model configs/models/imagenet_timm-vit-tiny16.yml \
    --config_detection configs/postprocessors/msp/imagenet_timm-vit-tiny16.yml \
    --root_dir ./results_main/msp/imagenet_timm-vit-tiny16/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/imagenet_timm-vit-tiny16_n_cal/ \
    --metric fpr \
    --mode search_no_fit

