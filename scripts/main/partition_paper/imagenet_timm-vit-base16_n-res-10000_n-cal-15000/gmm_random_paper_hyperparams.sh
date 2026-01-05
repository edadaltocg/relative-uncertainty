#!/usr/bin/env bash

python -m code.detection_main \
    --config_dataset configs/datasets/imagenet/imagenet_n_res-10000_seed-split-9.yml \
    --config_model configs/models/imagenet_timm-vit-base16.yml \
    --config_detection configs/postprocessors/clustering/imagenet_timm-vit-base16/gmm_random_paper.yml \
    --root_dir ./results_main/partition_gmm_paper/imagenet_timm-vit-base16/n_res-10000_n_cal-15000/calibrate_paper_hyperparams_random \
    --seed 1 \
    --gpu_id 1 \
    --latent_dir ./latent/imagenet_timm-vit-base16/ \
    --metric fpr \
    --quantizer_metric fpr \
    --mode search_res




