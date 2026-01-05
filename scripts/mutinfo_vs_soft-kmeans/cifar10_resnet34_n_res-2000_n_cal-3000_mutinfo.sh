#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar10/cifar10_n_res-2000_n-cal-3000.yml \
    --config_model configs/models/cifar10_resnet34.yml \
    --config_detection configs/postprocessors/clustering/mutinfo_vs_soft-kmeans/fixed_cov_random_init/cifar10_resnet34_mutinfo_opt.yml \
    --root_dir ./results_mutinfo_vs_soft-kmeans/fixed_cov_random_init/cifar10_resnet34_n_res-2000_n_cal-3000/ \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/cifar10_resnet34/ \
    --metric fpr \
    --mode evaluation






