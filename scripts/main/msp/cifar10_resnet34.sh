#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar10/cifar10_n-cal-5000_seed-split-1.yml \
    --config_model configs/models/cifar10_resnet34.yml \
    --config_detection configs/postprocessors/msp/cifar10_resnet34_hyperparams_search.yml \
    --root_dir ./results_main/msp/cifar10_resnet34/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/cifar10_resnet34/ \
    --metric fpr \
    --mode evaluation

