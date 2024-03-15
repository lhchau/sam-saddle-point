import yaml
import os

# Load YAML file
dir_path = os.path.join('.', 'config')
betas = [ [0.8, 0.9], [0.95, 0.9], [0.8, 0.95], [0.8, 0.99], [0.95, 0.95], [0.95, 0.99],]
for beta in betas:
    name = 'cifar100_resnet18_samaf_rho005_betas0909_bs128_seed42'
    with open(f'./config/{name}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['model']['betas'] = beta
    config['model']['rho'] = 0.2
    name = name.replace('0909', str(beta[0]).replace('.', '') + str(beta[1]).replace('.', '')).replace('rho005', 'rho02')
    # Save updated content to a new file
    with open(f'./config/{name}.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
