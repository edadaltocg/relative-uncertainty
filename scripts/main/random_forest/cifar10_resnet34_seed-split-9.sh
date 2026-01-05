#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar10/cifar10_n_cal-5000_seed-split-9.yml \
    --config_model configs/models/cifar10_resnet34.yml \
    --config_detection configs/postprocessors/random_forest/cifar10_resnet34_hyperparams_search.yml \
    --root_dir ./results_main/random_forest/cifar10_resnet34/ \
    --seed 1 \
    --gpu_id 1 \
    --latent_dir ./latent/ablation/cifar10_resnet34_n_cal/ \

