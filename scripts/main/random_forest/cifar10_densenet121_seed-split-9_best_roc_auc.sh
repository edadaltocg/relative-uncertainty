#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar10/cifar10_n_cal-5000_seed-split-9.yml \
    --config_model configs/models/cifar10_densenet121.yml \
    --config_detection configs/postprocessors/random_forest/cifar10_densenet121_best_roc_auc.yml \
    --root_dir ./results_main/random_forest/cifar10_densenet121/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/cifar10_densenet121_n_cal/ \
    --metric roc_auc \
    --mode evaluation

