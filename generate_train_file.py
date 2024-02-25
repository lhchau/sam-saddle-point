import os

text = "python dnsam/train_sam.py --experiment=cifar100_resnet18_sam_rho1_bs128_seed42\n"

dir_path = os.path.join('.', 'config')
f = open('train_eccv.sh', '+w')
for file in os.listdir(dir_path):
    if not file.endswith('.yaml') or 'wrn28_10' not in file:
        continue
    f.write(text.replace('cifar100_resnet18_sam_rho1_bs128_seed42', file[:-5]))

f.close()
    