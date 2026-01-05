#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar100/cifar100_n_cal-5000.yml \
    --config_model configs/models/cifar100_resnet34.yml \
    --config_detection configs/postprocessors/doctor/cifar100_resnet34_hyperparams_search.yml \
    --root_dir ./results_main/doctor/cifar100_resnet34/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/ablation/cifar100_resnet34_n_cal/ \

