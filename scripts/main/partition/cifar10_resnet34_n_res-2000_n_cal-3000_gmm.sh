#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar10/cifar10_n_res-2000_n-cal-3000.yml \
    --config_model configs/models/cifar10_resnet34.yml \
    --config_detection configs/postprocessors/clustering/cifar10_resnet34_gmm.yml \
    --root_dir ./results_main/partition_gmm_0.5_split/cifar10_resnet34_res-2000_cal-3000 \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/cifar10_resnet34/ \
    --metric roc_auc \
    --mode search




