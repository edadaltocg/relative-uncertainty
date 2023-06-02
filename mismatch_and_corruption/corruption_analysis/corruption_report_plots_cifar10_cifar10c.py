import os
import torch
import argparse
import logging
import pandas as pn
from tools import data_tools
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    config = data_tools.read_config(args.config_file_path)
    model_name = config['model_name']
    match_dataset_name = 'cifar10'
    mismatch_dataset_name = 'cifar10c'
    model_seed = 1
    data_path = 'data'
    device_id = 0
    batch_size = 1000
    rs = [10, 5, 3, 2]
    seeds = [1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10]
    temperatures = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 100.0, 1000.0]
    magnitudes = [0.0, 0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.0006, 0.0008,
                0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.0022, 0.0024,
                0.0026, 0.0028, 0.003, 0.0032, 0.0034, 0.0036, 0.0038, 0.004]

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    # std out handler
    std_out_handler = logging.StreamHandler()
    std_out_handler.setFormatter(formatter)
    logger.addHandler(std_out_handler)

    lbd = config['lbd']
    lr = 0.1
    epochs = 100
    corruptions = [
        "brightness",
                   "contrast",
                   "defocus_blur",
                   "elastic_transform",
                   "fog",
                   "frost",
                   "gaussian_blur",
                   "gaussian_noise",
                   "glass_blur",
                   "impulse_noise",
                   "jpeg_compression",
                   "motion_blur",
                   "pixelate",
                   "saturate",
                   "shot_noise",
                   "snow",
                   "spatter",
                   "speckle_noise",
                   "zoom_blur"
                   ]

    intensities = [
        # 1,
        # 2,
        # 3,
        4,
        # 5
        ]

    os.chdir('corruption_analysis')

    for corruption in corruptions:
        for intensity in intensities:
            logger.info(f'corruption: {corruption}, intensity: {intensity}')

            r_dict_doctor = {}
            for r in rs:
                seed_dict = {}
                for seed in seeds:
                    tmp_dict = {}
                    max_auc = -float('inf')
                    fpr_95_tpr_at_max_auc = None
                    tpr_at_max_auc = None
                    temperature_at_max_auc = None
                    magnitude_at_max_auc = None

                    for temperature in temperatures:
                        for magnitude in magnitudes:
                            source_folder = f'doctor/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/{corruption}_{intensity}'
                            dest_folder = f'{source_folder}/results/r_{r}/seed_{seed}'

                            final_dest_folder = f'{dest_folder}/magnitude_{magnitude}/temperature_{temperature}'
                            doctor_val_fprs = torch.load(
                                f'{final_dest_folder}/doctor_val_fprs.pt')
                            doctor_val_tprs = torch.load(
                                f'{final_dest_folder}/doctor_val_tprs.pt')
                            doctor_val_thresholds = torch.load(
                                f'{final_dest_folder}/doctor_val_thresholds.pt')
                            doctor_val_fpr = torch.load(
                                f'{final_dest_folder}/doctor_val_fpr.pt')
                            doctor_val_tpr = torch.load(
                                f'{final_dest_folder}/doctor_val_tpr.pt')
                            doctor_val_threshold = torch.load(
                                f'{final_dest_folder}/doctor_val_threshold.pt')
                            doctor_val_auc = torch.load(
                                f'{final_dest_folder}/doctor_val_auc.pt')

                            if doctor_val_auc > max_auc:
                                max_auc = doctor_val_auc
                                fpr_95_tpr_at_max_auc = doctor_val_fpr
                                tpr_at_max_auc = doctor_val_tpr
                                temperature_at_max_auc = temperature
                                magnitude_at_max_auc = magnitude
                            elif doctor_val_auc == max_auc:
                                if doctor_val_fpr < fpr_95_tpr_at_max_auc:
                                    fpr_95_tpr_at_max_auc = doctor_val_fpr
                                    tpr_at_max_auc = doctor_val_tpr
                                    temperature_at_max_auc = temperature
                                    magnitude_at_max_auc = magnitude
                    tmp_dict['max_auc'] = max_auc
                    tmp_dict['fpr_95_tpr_at_max_auc'] = fpr_95_tpr_at_max_auc
                    tmp_dict['tpr_at_max_auc'] = tpr_at_max_auc
                    tmp_dict['temperature_at_max_auc'] = temperature_at_max_auc
                    tmp_dict['magnitude_at_max_auc'] = magnitude_at_max_auc
                    seed_dict[seed] = tmp_dict
                r_dict_doctor[r] = seed_dict

            # print r_dict_doctor element by element
            # for r in rs:
            #     for seed in seeds:
            #         print(f'r: {r}, seed: {seed}')
            #         print(r_dict_doctor[r][seed])
            # create dataframe with columns: r, seed, max_auc, fpr_95_tpr_at_max_auc
            df_doctor = pn.DataFrame(columns=['r', 'seed', 'auc', 'fpr_95_tpr'])

            for r in rs:
                for seed in seeds:
                    selected_temperature = r_dict_doctor[r][seed]['temperature_at_max_auc']
                    selected_magnitude = r_dict_doctor[r][seed]['magnitude_at_max_auc']

                    source_folder = f'doctor/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/{corruption}_{intensity}'
                    dest_folder = f'{source_folder}/results/r_{r}/seed_{seed}'

                    final_dest_folder = f'{dest_folder}/magnitude_{selected_magnitude}/temperature_{selected_temperature}'
                    doctor_test_fprs = torch.load(
                        f'{final_dest_folder}/doctor_test_fprs.pt')
                    doctor_test_tprs = torch.load(
                        f'{final_dest_folder}/doctor_test_tprs.pt')
                    doctor_test_thresholds = torch.load(
                        f'{final_dest_folder}/doctor_test_thresholds.pt')
                    doctor_test_fpr = torch.load(
                        f'{final_dest_folder}/doctor_test_fpr.pt')
                    doctor_test_tpr = torch.load(
                        f'{final_dest_folder}/doctor_test_tpr.pt')
                    doctor_test_threshold = torch.load(
                        f'{final_dest_folder}/doctor_test_threshold.pt')
                    doctor_test_auc = torch.load(
                        f'{final_dest_folder}/doctor_test_auc.pt')

                    # for each r and seed put doctor_test_auc and doctor_test_fpr in a row of df_doctor using pandas concat
                    df_doctor = pn.concat(
                        [df_doctor,
                        pn.DataFrame([[1/float(r), seed, doctor_test_auc, doctor_test_fpr]],
                                    columns=['r', 'seed', 'auc', 'fpr_95_tpr'])],
                        ignore_index=True)

            # print(df_doctor)

            r_dict_D = {}
            for r in rs:
                seed_dict = {}
                for seed in seeds:
                    tmp_dict = {}
                    max_auc = -float('inf')
                    fpr_95_tpr_at_max_auc = None
                    tpr_at_max_auc = None
                    temperature_at_max_auc = None
                    magnitude_at_max_auc = None

                    for temperature in temperatures:
                        for magnitude in magnitudes:
                            source_folder = f'd_matrix/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}'
                            dest_folder = f'{source_folder}/seed_{seed}/r_{r}/lr_{lr}/epochs_{epochs}/lbd_{lbd}/{corruption}_{intensity}'

                            final_dest_folder = f'{dest_folder}/magnitude_{magnitude}/temperature_{temperature}'
                            D_val_fprs = torch.load(
                                f'{final_dest_folder}/D_fprs_val.pt')
                            D_val_tprs = torch.load(
                                f'{final_dest_folder}/D_tprs_val.pt')
                            D_val_thresholds = torch.load(
                                f'{final_dest_folder}/D_thresholds_val.pt')
                            D_val_fpr = torch.load(
                                f'{final_dest_folder}/D_fpr_val.pt')
                            D_val_tpr = torch.load(
                                f'{final_dest_folder}/D_tpr_val.pt')
                            D_val_threshold = torch.load(
                                f'{final_dest_folder}/D_threshold_val.pt')
                            D_val_auc = torch.load(
                                f'{final_dest_folder}/D_auc_val.pt')

                            if D_val_auc > max_auc:
                                max_auc = D_val_auc
                                fpr_95_tpr_at_max_auc = D_val_fpr
                                tpr_at_max_auc = D_val_tpr
                                temperature_at_max_auc = temperature
                                magnitude_at_max_auc = magnitude
                            elif D_val_auc == max_auc:
                                if D_val_fpr < fpr_95_tpr_at_max_auc:
                                    fpr_95_tpr_at_max_auc = D_val_fpr
                                    tpr_at_max_auc = D_val_tpr
                                    temperature_at_max_auc = temperature
                                    magnitude_at_max_auc = magnitude
                    tmp_dict['max_auc'] = max_auc
                    tmp_dict['fpr_95_tpr_at_max_auc'] = fpr_95_tpr_at_max_auc
                    tmp_dict['tpr_at_max_auc'] = tpr_at_max_auc
                    tmp_dict['temperature_at_max_auc'] = temperature_at_max_auc
                    tmp_dict['magnitude_at_max_auc'] = magnitude_at_max_auc
                    seed_dict[seed] = tmp_dict
                r_dict_D[r] = seed_dict

            # print r_dict_D element by element
            # for r in rs:
            #     for seed in seeds:
            #         print(f'r: {r}, seed: {seed}')
            #         print(r_dict_D[r][seed])

            # create dataframe with columns: r, seed, max_auc, fpr_95_tpr_at_max_auc
            df_D = pn.DataFrame(columns=['r', 'seed', 'auc', 'fpr_95_tpr'])

            for r in rs:
                for seed in seeds:
                    selected_temperature = r_dict_D[r][seed]['temperature_at_max_auc']
                    selected_magnitude = r_dict_D[r][seed]['magnitude_at_max_auc']

                    source_folder = f'd_matrix/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}'
                    dest_folder = f'{source_folder}/seed_{seed}/r_{r}/lr_{lr}/epochs_{epochs}/lbd_{lbd}/{corruption}_{intensity}'

                    final_dest_folder = f'{dest_folder}/magnitude_{selected_magnitude}/temperature_{selected_temperature}'
                    D_test_fprs = torch.load(
                        f'{final_dest_folder}/D_fprs_test.pt')
                    D_test_tprs = torch.load(
                        f'{final_dest_folder}/D_tprs_test.pt')
                    D_test_thresholds = torch.load(
                        f'{final_dest_folder}/D_thresholds_test.pt')
                    D_test_fpr = torch.load(
                        f'{final_dest_folder}/D_fpr_test.pt')
                    D_test_tpr = torch.load(
                        f'{final_dest_folder}/D_tpr_test.pt')
                    D_test_threshold = torch.load(
                        f'{final_dest_folder}/D_threshold_test.pt')
                    D_test_auc = torch.load(
                        f'{final_dest_folder}/D_auc_test.pt')

                    df_D = pn.concat(
                        [df_D,
                        pn.DataFrame([[1/float(r), seed, D_test_auc, D_test_fpr]],
                                    columns=['r', 'seed', 'auc', 'fpr_95_tpr'])],
                        ignore_index=True)

            # print(df_D)

            # create a figure with two plots side by side: one for the AUC and one for the FPR
            # in the one on the left plot the mean AUC over the seed and the area between the mean AUC and the min and max AUC
            # the x axis is labeled with the r values
            # in the one on the right plot the mean FPR over the seed and the area between the mean FPR and the min and max FPR
            # the x axis is labeled with the r values

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            # plot AUC
            df_doctor_mean = df_doctor.groupby('r').mean(numeric_only=True)
            # df_doctor_median = df_doctor.groupby('r').median(numeric_only=True)
            df_doctor_std = df_doctor.groupby('r').std(numeric_only=True)
            df_doctor_min = df_doctor.groupby('r').min(numeric_only=True)
            df_doctor_max = df_doctor.groupby('r').max(numeric_only=True)

            df_D_mean = df_D.groupby('r').mean(numeric_only=True)
            # df_D_median = df_D.groupby('r').median(numeric_only=True)
            df_D_std = df_D.groupby('r').std(numeric_only=True)
            df_D_min = df_D.groupby('r').min(numeric_only=True)
            df_D_max = df_D.groupby('r').max(numeric_only=True)

            ax1.plot(df_doctor_mean.index, df_doctor_mean['auc'], color='blue')
            ax1.plot(df_D_mean.index, df_D_mean['auc'], color='red')
            ax1.fill_between(df_doctor_mean.index,
                            df_doctor_mean['auc'] - df_doctor_std['auc'],
                            df_doctor_mean['auc'] + df_doctor_std['auc'],
                            color='blue',
                            alpha=0.2)
            ax1.fill_between(df_D_mean.index,
                            df_D_mean['auc'] - df_D_std['auc'],
                            df_D_mean['auc'] + df_D_std['auc'],
                            color='red',
                            alpha=0.2)

            ax1.set_xlabel('r')
            ax1.set_ylabel('AUC')

            # plot FPR
            ax2.plot(df_doctor_mean.index, df_doctor_mean['fpr_95_tpr'], color='blue')
            ax2.plot(df_D_mean.index, df_D_mean['fpr_95_tpr'], color='red')
            ax2.fill_between(df_doctor_mean.index,
                            df_doctor_mean['fpr_95_tpr'] - df_doctor_std['fpr_95_tpr'],
                            df_doctor_mean['fpr_95_tpr'] + df_doctor_std['fpr_95_tpr'],
                            color='blue',
                            alpha=0.2)
            ax2.fill_between(df_D_mean.index,
                            df_D_mean['fpr_95_tpr'] - df_D_std['fpr_95_tpr'],
                            df_D_mean['fpr_95_tpr'] + df_D_std['fpr_95_tpr'],
                            color='red',
                            alpha=0.2)

            ax2.set_xlabel('r')
            ax2.set_ylabel('FPR')

            ax1.set_xticks([round(1/float(rs[i]), 2) for i in range(len(rs))])
            ax2.set_xticks([round(1/float(rs[i]), 2) for i in range(len(rs))])

            plt.tight_layout()
            plt.legend(['Doctor', 'D'])

            #create a folder for the plots
            if not os.path.exists(f'plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/corruption_{corruption}_{intensity}/lbda_{lbd}'):
                os.makedirs(f'plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/corruption_{corruption}_{intensity}/lbda_{lbd}')
            plt.savefig(f'plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/corruption_{corruption}_{intensity}/lbda_{lbd}/r_{rs[0]}_{rs[-1]}_step_{rs[1]-rs[0]}.png')