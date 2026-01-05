#!/usr/bin/env bash

python -m code.detection_ablation \
    --config_dataset configs/datasets/cifar100/cifar100_ablation.yml \
    --config_model configs/models/cifar100_densenet121.yml \
    --config_detection configs/postprocessors/clustering/clustering_cifar100_densenet121_ablation2.yml \
    --root_dir ./results_ablation/cifar100_densenet121_n_cal_predweights-0/ \
    --seed 1 \
    --gpu_id 1 \
    --latent_dir ./latent/ablation/cifar100_densenet121_n_cal/ 

