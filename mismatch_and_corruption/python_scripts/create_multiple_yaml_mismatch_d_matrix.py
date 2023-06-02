import os
import yaml
import copy
from tools import data_tools
import itertools
# model_names
architectures = [
"densenet121_custom",
"resnet34_custom",
]

# training types
training_types = [
"",
"_lognorm",
"_mixup",
"_regmixup",
]
if __name__ == "__main__":

     for arch, tp in itertools.product(architectures, training_types):
        model_name = arch + tp
        config_file_path = f'mismatch_analysis/d_matrix/mismatch_d_matrix_config.yaml'
        config = data_tools.read_config(config_file_path)
        config["model_name"] = model_name
        # write the model in it

        list_files = []
        config_copy = copy.deepcopy(config)

        count = 0

        for r in config['rs']:
            for lmbd in config['lbds']:
                for seed in config['seeds']:
                    config_copy['batch_size'] = 512
                    config_copy['rs'] = [r]
                    config_copy['lbds'] = [lmbd]
                    config_copy['batch_size'] = 512
                    config_copy['seeds'] = [seed]

                    for key, value in config_copy.items():
                        print(key, value)

                    # create a yaml file with the new config
                    dest_folder = 'mismatch_analysis/d_matrix/sbatch_array_yaml_files_eval'
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    file_name = f'{dest_folder}/file_{count}_{model_name}.yaml'
                    count += 1
                    list_files.append(file_name.split('/')[-1])
                    file = open(file_name, "w")
                    yaml.dump(config_copy, file)
                    file.close()
                    print(f"YAML {r} {lmbd} {seed} file saved.")
        # print elements of list_files as a string separated by spaces without brackets and quotes
        print(' '.join(map(str, list_files)))

        #####

        list_files = []
        config_copy = copy.deepcopy(config)

        count = 0

        for seed in config['seeds']:
            for r in config['rs']:
                for lmbd in config['lbds']:
                    config_copy['batch_size'] = 512
                    config_copy['seeds'] = [seed]
                    config_copy['rs'] = [r]
                    config_copy['lbds'] = [lmbd]
                    for key, value in config_copy.items():
                        print(key, value)

                    # create a yaml file with the new config
                    dest_folder = 'mismatch_analysis/d_matrix/sbatch_array_yaml_files_train'
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    file_name = f'{dest_folder}/file_{count}_{model_name}.yaml'
                    count += 1
                    list_files.append(file_name.split('/')[-1])
                    file = open(file_name, "w")
                    yaml.dump(config_copy, file)
                    file.close()
                    print(f"YAML {seed} {r} {lmbd} file saved.")

        # print elements of list_files as a string separated by spaces without brackets and quotes
        print(' '.join(map(str, list_files)))
