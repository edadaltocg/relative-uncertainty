#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar10/cifar10_n-cal-5000_seed-split-9.yml \
    --config_model configs/models/cifar10_resnet34.yml \
    --config_detection configs/postprocessors/clustering/cifar10_resnet34/unif-width.yml \
    --root_dir ./results_hyperparams/partition_unif-width/cifar10_resnet34/n_cal-5000/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/cifar10_resnet34/ \
    --metric fpr \
    --quantizer_metric fpr \
    --mode evaluation





