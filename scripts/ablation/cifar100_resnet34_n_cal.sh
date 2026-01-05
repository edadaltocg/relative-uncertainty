#!/usr/bin/env bash
export CUDA_LAUNCH_BLOCKING=1
python -m code.detection_ablation \
    --config_dataset configs/datasets/cifar100/cifar100_ablation.yml \
    --config_model configs/models/cifar100_resnet34.yml \
    --config_detection configs/postprocessors/clustering/clustering_cifar100_resnet34_ablation.yml \
    --root_dir ./results_ablation/clustering/cifar100_resnet34_n_cal/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/cifar100_resnet34_n_cal/ 

