#!/usr/bin/env bash

python -m code.detection_ablation_hyperparams \
    --config_dataset configs/datasets/cifar10/cifar10_ablation_hyperparams.yml \
    --config_model configs/models/cifar10_densenet121.yml \
    --config_detection configs/postprocessors/clustering/clustering_cifar10_densenet121_ablation_hyperparams.yml \
    --root_dir ./results_ablation_hyperparams/cifar10_densenet121/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/cifar10_densenet121_n_cal/ \

