#!/usr/bin/env bash

python -m code.detection_hyperparams \
    --config_dataset configs/datasets/imagenet/imagenet_n-cal-25000_seed-split-9.yml \
    --config_model configs/models/imagenet_timm-vit-tiny16.yml \
    --config_detection configs/postprocessors/clustering/imagenet_timm-vit-tiny16/unif-width.yml \
    --root_dir ./results_hyperparams/partition_unif-width/imagenet_timm-vit-tiny16/n_cal-25000/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/imagenet_timm-vit-tiny16_n_cal/ \
    --metric fpr \
    --quantizer_metric fpr \
    --mode evaluation





