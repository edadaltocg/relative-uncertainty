#!/usr/bin/env bash

python -m code.detection_ablation_hyperparams \
    --config_dataset configs/datasets/cifar100/cifar100_ablation_hyperparams.yml \
    --config_model configs/models/cifar100_densenet121.yml \
    --config_detection configs/postprocessors/clustering/clustering_cifar100_densenet121_ablation_hyperparams.yml \
    --root_dir ./results_ablation_hyperparams/cifar100_densenet121/ \
    --seed 1 \
    --gpu_id 1 \
    --latent_dir ./latent/ablation/cifar100_densenet121_n_cal/ 

