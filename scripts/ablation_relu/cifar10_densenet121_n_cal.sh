#!/usr/bin/env bash

python -m code.detection_ablation \
    --config_dataset configs/datasets/cifar10/cifar10_ablation.yml \
    --config_model configs/models/cifar10_densenet121.yml \
    --config_detection configs/postprocessors/relu/relu_cifar10_densenet121_ablation.yml \
    --root_dir ./results_ablation/relu/cifar10_densenet121_n_cal/ \
    --seed 1 \
    --gpu_id 1 \
    --latent_dir ./latent/ablation/cifar10_densenet121_n_cal/ 

