#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/imagenet/imagenet_n_res-10000_seed-split-9.yml \
    --config_model configs/models/imagenet_timm_vit_tiny16.yml \
    --config_detection configs/postprocessors/clustering/mutinfo_vs_soft-kmeans/fixed_cov_random_init/imagenet_timm_vit_tiny16_mutinfo_opt.yml \
    --root_dir ./results_mutinfo_vs_soft-kmeans/fixed_cov_random_init/imagenet_timm_vit_tiny16_n_res-10000/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/imagenet_timm-vit-tiny16_n_cal_with-res-10000/ \


