#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar10/cifar10_n_cal-5000.yml \
    --config_model configs/models/cifar10_densenet121.yml \
    --config_detection configs/postprocessors/msp/cifar10_densenet121_hyperparams_search.yml \
    --root_dir ./results_main/msp/cifar10_densenet121/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/cifar10_densenet121_n_cal/ \

