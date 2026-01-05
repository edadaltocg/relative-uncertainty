#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar10/cifar10_n-res-3000_n-cal-2000_seed-split-9.yml \
    --config_model configs/models/cifar10_densenet121.yml \
    --config_detection configs/postprocessors/clustering/cifar10_densenet121/kmeans_random_paper.yml \
    --root_dir ./results_main/partition_kmeans_paper/cifar10_densenet121/n_res-3000_n_cal-2000/procedure-1_paper_hyperparams_random \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/cifar10_densenet121_n-res-3000_n-cal-2000/ \
    --metric fpr \
    --quantizer_metric fpr \
    --mode search_res



