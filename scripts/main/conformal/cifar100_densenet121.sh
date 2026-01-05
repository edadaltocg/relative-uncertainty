#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar100/cifar100_n_cal-5000_seed-split-9.yml \
    --config_model configs/models/cifar100_densenet121.yml \
    --config_detection configs/postprocessors/conformal/cifar100_densenet121.yml \
    --root_dir ./results_main/conformal/cifar100_densenet121/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/cifar100_densenet121_n_cal/ \
    --mode evaluation

