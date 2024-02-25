import yaml
import os

# Load YAML file
dir_path = os.path.join('.', 'config')
model_list = ["resnet18"]
dataset_list = ["cifar100"]
threshold_list = [0.95]
factor_list = [1.1, 1.01, 1.001, 1.0001]
seed_list = [43, 44]
for dataset in dataset_list:
    for model in model_list:
        for threshold in threshold_list:
            for factor in factor_list:
                for seed in seed_list:
                    name = 'cifar100_resnet18_sam_adaptive_rho005_simple_095_11_bs128_seed42'
                    with open(f'./config/{name}.yaml', 'r') as file:
                        config = yaml.safe_load(file)
                    name = name.replace("resnet18", model)
                    name = name.replace("095", str(threshold).replace('.', ''))
                    name = name.replace("11", str(factor).replace('.', ''))
                    name = name.replace("cifar100", dataset)
                    name = name.replace("42", str(seed))

                    if dataset == "fashion_mnist":
                        _model = model + '_mnist'
                    else: _model = model
                    # Change seed value
                    config['trainer']['seed'] = seed
                    
                    config['data']['name'] = dataset
                    config['model']['architecture'] = _model
                    config['wandb']['name'] = config['wandb']['name'].replace("resnet18", model).replace("095", str(threshold).replace('.', '')).replace("11", str(factor).replace('.', ''))
                    
                    config['trainer']['threshold'] = threshold
                    config['trainer']['factor'] = factor
                    
                    config['wandb']['project'] = config['wandb']['project'].replace("CIFAR10", dataset.upper())

                    # Save updated content to a new file
                    with open(f'./config/{name}.yaml', 'w') as file:
                        yaml.dump(config, file, default_flow_style=False)
