#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/cifar10/cifar10_n_res-2000_n-cal-3000.yml \
    --config_model configs/models/cifar10_densenet121.yml \
    --config_detection configs/postprocessors/clustering/cifar10_densenet121/gmm_random_cv_paper.yml \
    --root_dir ./results_main/partition_gmm_paper/cifar10_densenet121/n_res-2000_n_cal-3000/procedure-2_paper_hyperparams_random \
    --seed 1 \
    --gpu_id 0 \
    --latent_dir ./latent/cifar10_densenet121/ \
    --metric fpr \
    --quantizer_metric likelihood \
    --mode search_cv




