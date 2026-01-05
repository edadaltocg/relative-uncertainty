#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar100/cifar100_n_res-3000_n-cal-2000_seed-split-9.yml \
    --config_model configs/models/cifar100_densenet121.yml \
    --config_detection configs/postprocessors/clustering/cifar100_densenet121/gmm_plus_paper.yml \
    --root_dir ./results_main/partition_gmm_paper/cifar100_densenet121/n_res-3000_n_cal-2000/procedure-1_paper_hyperparams_plus \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/cifar100_densenet121_n-res-3000_n-cal-2000/ \
    --metric fpr \
    --quantizer_metric fpr \
    --mode search_res




