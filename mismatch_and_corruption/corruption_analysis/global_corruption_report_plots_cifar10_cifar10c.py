import os
import torch
import logging
import argparse
import numpy as np
import pandas as pn
from tqdm import tqdm
import plotly.express as px
from tools import data_tools
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import itertools

models = ["densenet121_custom", "resnet34_custom"]
types = ["", "_mixup", "_regmixup", "_lognorm"]
lbd_indexes = [0, 1, 2, 3]

configs = list(itertools.product(models, types, lbd_indexes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_idx", type=int)
    # parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    # config = data_tools.read_config(args.config_file_path)
    # model_name = config['model_name']
    model_name, tp, lbd_index = configs[int(args.config_idx)]
    model_name = model_name + tp
    match_dataset_name = 'cifar10'
    mismatch_dataset_name = 'cifar10c'
    model_seed = 1
    # data_path = 'data'
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
    # standard output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    os.chdir('corruption_analysis')

    # lbds = [config['lbd']]
    all_lbds = [0.5, 0.6, 0.8, 1.0]
    lbds = [all_lbds[lbd_index]]
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
        # 4,
        5
    ]
    global_dictionary_doctor = {}
    for corruption in tqdm(corruptions, "Corruption"):
        for intensity in intensities:
            logger.info(f"Corruption: {corruption}, Intensity: {intensity}")

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
                            source_folder = f'corruption_analysis/doctor/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/{corruption}_{intensity}'
                            source_folder = os.path.join(os.environ.get("SCRATCH", "."), source_folder)
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

            global_dictionary_doctor[f"{corruption}_{intensity}"] = r_dict_doctor
    if not os.path.exists(f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}'):
        os.makedirs(f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}')
    torch.save(global_dictionary_doctor, '/'.join((f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}', 'global_dictionary_doctor.pt')))
    # for key, value in global_dictionary_doctor.items():
    #     print(key)
    #     print(value)

    angles_dict_doctor = {}
    auc_dict_doctor = {}
    fpr_95_tpr_dict_doctor = {}

    for r in tqdm(rs, "Rs"):
        auc_dict_doctor[r] = {}
        fpr_95_tpr_dict_doctor[r] = {}
        angles_dict_doctor[r] = []
        for corruption in corruptions:
            for intensity in intensities:
                auc_dict_doctor[r][f"{corruption}_{intensity}"] = None
                fpr_95_tpr_dict_doctor[r][f"{corruption}_{intensity}"] = None
                angles_dict_doctor[r].append(f"{corruption}_{intensity}")
                auc_list = []
                fpr_95_tpr_list = []
                for seed in seeds:
                    selected_temperature = global_dictionary_doctor[f"{corruption}_{intensity}"][r][seed]['temperature_at_max_auc']
                    selected_magnitude = global_dictionary_doctor[f"{corruption}_{intensity}"][r][seed]['magnitude_at_max_auc']

                    source_folder = f'corruption_analysis/doctor/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/{corruption}_{intensity}'
                    source_folder = os.path.join(os.environ.get("SCRATCH", "."), source_folder)
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

                    auc_list.append(doctor_test_auc)
                    fpr_95_tpr_list.append(doctor_test_fpr)

                auc_dict_doctor[r][f"{corruption}_{intensity}"] = np.mean(auc_list)
                fpr_95_tpr_dict_doctor[r][f"{corruption}_{intensity}"] = np.mean(fpr_95_tpr_list)

    torch.save(angles_dict_doctor, '/'.join((f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}', 'angles_dict_doctor.pt')))
    torch.save(auc_dict_doctor, '/'.join((f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}', 'auc_dict_doctor.pt')))
    torch.save(fpr_95_tpr_dict_doctor, '/'.join((f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}', 'fpr_95_tpr_dict_doctor.pt')))

    global_dictionary_d_matrix = {}
    for corruption in tqdm(corruptions, "Corruptions"):
        for intensity in intensities:
            logger.info(f"Corruption: {corruption}, Intensity: {intensity}")

            r_dict_d_matrix = {}
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
                            for lbd in lbds:
                                source_folder = f'corruption_analysis/d_matrix/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/seed_{seed}'
                                source_folder = os.path.join(os.environ.get("SCRATCH", "."), source_folder)
                                dest_folder = f'{source_folder}/r_{r}/lr_{lr}/epochs_{epochs}/lbd_{lbd}'

                                final_dest_folder = f'{dest_folder}/{corruption}_{intensity}/magnitude_{magnitude}/temperature_{temperature}'
                                d_matrix_val_fprs = torch.load(
                                    f'{final_dest_folder}/D_fprs_val.pt')
                                d_matrix_val_tprs = torch.load(
                                    f'{final_dest_folder}/D_tprs_val.pt')
                                d_matrix_val_thresholds = torch.load(
                                    f'{final_dest_folder}/D_thresholds_val.pt')
                                d_matrix_val_fpr = torch.load(
                                    f'{final_dest_folder}/D_fpr_val.pt')
                                d_matrix_val_tpr = torch.load(
                                    f'{final_dest_folder}/D_tpr_val.pt')
                                d_matrix_val_threshold = torch.load(
                                    f'{final_dest_folder}/D_threshold_val.pt')
                                d_matrix_val_auc = torch.load(
                                    f'{final_dest_folder}/D_auc_val.pt')

                                if d_matrix_val_auc > max_auc:
                                    max_auc = d_matrix_val_auc
                                    fpr_95_tpr_at_max_auc = d_matrix_val_fpr
                                    tpr_at_max_auc = d_matrix_val_tpr
                                    temperature_at_max_auc = temperature
                                    magnitude_at_max_auc = magnitude
                                elif d_matrix_val_auc == max_auc:
                                    if d_matrix_val_fpr < fpr_95_tpr_at_max_auc:
                                        fpr_95_tpr_at_max_auc = d_matrix_val_fpr
                                        tpr_at_max_auc = d_matrix_val_tpr
                                        temperature_at_max_auc = temperature
                                        magnitude_at_max_auc = magnitude
                    tmp_dict['max_auc'] = max_auc
                    tmp_dict['fpr_95_tpr_at_max_auc'] = fpr_95_tpr_at_max_auc
                    tmp_dict['tpr_at_max_auc'] = tpr_at_max_auc
                    tmp_dict['temperature_at_max_auc'] = temperature_at_max_auc
                    tmp_dict['magnitude_at_max_auc'] = magnitude_at_max_auc
                    seed_dict[seed] = tmp_dict
                r_dict_d_matrix[r] = seed_dict

            global_dictionary_d_matrix[f"{corruption}_{intensity}"] = r_dict_d_matrix
    torch.save(global_dictionary_d_matrix, '/'.join((f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}', 'global_dictionary_d_matrix.pt')))
    # for key, value in global_dictionary_d_matrix.items():
    #     print(key)
    #     print(value)

    angles_dict_d_matrix = {}
    auc_dict_d_matrix = {}
    fpr_95_tpr_dict_d_matrix = {}

    for r in tqdm(rs, "Rs"):
        auc_dict_d_matrix[r] = {}
        fpr_95_tpr_dict_d_matrix[r] = {}
        angles_dict_d_matrix[r] = []
        for corruption in corruptions:
            for intensity in intensities:
                auc_dict_d_matrix[r][f"{corruption}_{intensity}"] = None
                fpr_95_tpr_dict_d_matrix[r][f"{corruption}_{intensity}"] = None
                angles_dict_d_matrix[r].append(f"{corruption}_{intensity}")
                auc_list = []
                fpr_95_tpr_list = []
                for seed in seeds:
                    selected_temperature = global_dictionary_d_matrix[f"{corruption}_{intensity}"][r][seed]['temperature_at_max_auc']
                    selected_magnitude = global_dictionary_d_matrix[f"{corruption}_{intensity}"][r][seed]['magnitude_at_max_auc']

                    ROOT = os.environ.get("SCRATCH", ".")
                    source_folder = f'd_matrix/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/seed_{seed}'
                    source_folder = os.path.join(ROOT,"corruption_analysis", source_folder)
                    dest_folder = f'{source_folder}/r_{r}/lr_{lr}/epochs_{epochs}/lbd_{lbd}'

                    final_dest_folder =  f'{dest_folder}/{corruption}_{intensity}/magnitude_{selected_magnitude}/temperature_{selected_temperature}'
                    d_matrix_test_fprs = torch.load(
                        f'{final_dest_folder}/D_fprs_test.pt')
                    d_matrix_test_tprs = torch.load(
                        f'{final_dest_folder}/D_tprs_test.pt')
                    d_matrix_test_thresholds = torch.load(
                        f'{final_dest_folder}/D_thresholds_test.pt')
                    d_matrix_test_fpr = torch.load(
                        f'{final_dest_folder}/D_fpr_test.pt')
                    d_matrix_test_tpr = torch.load(
                        f'{final_dest_folder}/D_tpr_test.pt')
                    d_matrix_test_threshold = torch.load(
                        f'{final_dest_folder}/D_threshold_test.pt')
                    d_matrix_test_auc = torch.load(
                        f'{final_dest_folder}/D_auc_test.pt')

                    auc_list.append(d_matrix_test_auc)
                    fpr_95_tpr_list.append(d_matrix_test_fpr)

                auc_dict_d_matrix[r][f"{corruption}_{intensity}"] = np.mean(auc_list)
                fpr_95_tpr_dict_d_matrix[r][f"{corruption}_{intensity}"] = np.mean(fpr_95_tpr_list)

    torch.save(angles_dict_d_matrix, '/'.join((f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}', 'angles_dict_d_matrix.pt')))
    torch.save(auc_dict_d_matrix, '/'.join((f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}', 'auc_dict_d_matrix.pt')))
    torch.save(fpr_95_tpr_dict_d_matrix, '/'.join((f'./polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbds[0]}', 'fpr_95_tpr_dict_d_matrix.pt')))

    # for any element in the list rs
    for r in rs:
        angles = angles_dict_doctor[r]
        aucs_doctor = []
        fpr_95_tpr_doctor = []
        aucs_d_matrix = []
        fpr_95_tpr_d_matrix = []
        for angle in angles:
            aucs_doctor.append(auc_dict_doctor[r][angle])
            aucs_d_matrix.append(auc_dict_d_matrix[r][angle])
            fpr_95_tpr_doctor.append(fpr_95_tpr_dict_doctor[r][angle])
            fpr_95_tpr_d_matrix.append(fpr_95_tpr_dict_d_matrix[r][angle])

        df_aucs_doctor = pn.DataFrame(dict(
            r=aucs_doctor,
            theta=angles))
        # print(df_aucs)
        df_aucs_d_matrix = pn.DataFrame(dict(
            r=aucs_d_matrix,
            theta=angles))

        df_fpr_95_tpr_doctor = pn.DataFrame(dict(
            r=fpr_95_tpr_doctor,
            theta=angles))

        df_fpr_95_tpr_d_matrix = pn.DataFrame(dict(
            r=fpr_95_tpr_d_matrix,
            theta=angles))

        ###################################

        fig = go.Figure()
        fig.update_layout(
            title={
                'text': "AUC",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        fig.add_trace(go.Scatterpolar(
            r=aucs_doctor,
            theta=angles,
            fill='toself',
            name='Doctor'
        ))
        fig.add_trace(go.Scatterpolar(
            r=aucs_d_matrix,
            theta=angles,
            fill='toself',
            name='D_matrix'
        ))

        fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )),
        showlegend=True
        )



        if not os.path.exists(f'polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbd}'):
            os.makedirs(f'polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbd}')
        fig.write_image(f"polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbd}/polar_plot_r_{r}_auc.png")



        fig = go.Figure()
        fig.update_layout(
            title={
                'text': "FPR at 95% TPR",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        fig.add_trace(go.Scatterpolar(
            r=fpr_95_tpr_doctor,
            theta=angles,
            fill='toself',
            name='Doctor'
        ))
        fig.add_trace(go.Scatterpolar(
            r=fpr_95_tpr_d_matrix,
            theta=angles,
            fill='toself',
            name='D_matrix'
        ))

        fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )),
        showlegend=True
        )

        if not os.path.exists(f'polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbd}'):
            os.makedirs(f'polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbd}')
        fig.write_image(f"polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbd}/polar_plot_r_{r}_fpr.png")

        ###################################

        # fig = px.line_polar(df_aucs_doctor, r='r', theta='theta', line_close=True)
        # fig.update_traces(fill='toself')
        # fig.show()

        # fig = px.line_polar(df_fpr_95_tpr, r='r', theta='theta', line_close=True)
        # fig.update_traces(fill='toself')
        # fig.show()

        # fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'polar'}] * 2] * 1)

        # fig.add_trace(
        #     go.Scatterpolar(theta=angles, r=aucs_doctor, name='AUC doctor', fill='toself'),
        #     row=1, col=1
        # )

        # fig.add_trace(
        #     go.Scatterpolar(theta=angles, r=aucs_d_matrix, fill='toself', name='AUC d_matrix'),
        #     row=1, col=1
        # )

        # fig.add_trace(
        #     go.Scatterpolar(theta=angles, r=fpr_95_tpr_doctor, name='fpr doctor', fill='toself'),
        #     row=1, col=2
        # )

        # fig.add_trace(
        #     go.Scatterpolar(theta=angles, r=fpr_95_tpr_d_matrix, fill='toself', name='fpr d_matrix'),
        #     row=1, col=2
        # )
        # # fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0, 1])),showlegend=True)
        # # fig.update_layout(polar=dict(radialaxis=dict(visible=True,)),showlegend=True)
        # fig.update_layout(
        #     polar=dict(radialaxis=dict(range=[0, 1])),
        #     polar2=dict(radialaxis=dict(range=[0, 1])),
        #     showlegend=False
        # )


        # fig.show()
        # create a folder to save the plots if it does not exist
        # if not os.path.exists(f'polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbd}'):
        #     os.makedirs(f'polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbd}')
        # fig.write_image(f"polar_plots/{match_dataset_name}_to_{mismatch_dataset_name}/{model_name}/model_seed_{model_seed}/lbd_{lbd}/polar_plot_r_{r}.png")