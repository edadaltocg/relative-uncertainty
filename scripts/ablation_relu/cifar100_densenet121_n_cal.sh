#!/usr/bin/env bash

python -m code.detection_ablation \
    --config_dataset configs/datasets/cifar100/cifar100_ablation.yml \
    --config_model configs/models/cifar100_densenet121.yml \
    --config_detection configs/postprocessors/relu/relu_cifar100_densenet121_ablation.yml \
    --root_dir ./results_ablation/relu/cifar100_densenet121_n_cal/ \
    --seed 1 \
    --gpu_id 1 \
    --latent_dir ./latent/ablation/cifar100_densenet121_n_cal/ 

