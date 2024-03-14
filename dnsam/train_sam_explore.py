'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime

import os
import argparse
import wandb
import dynamic_yaml
import pprint

from dnsam.models import *
from dnsam.utils import *
from dnsam.data import *
from dnsam.scheduler import *
from dnsam.optimizer import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

################################
#### 0. SETUP CONFIGURATION
################################
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--experiment', default='example', type=str, help='path to YAML config file')
args = parser.parse_args()

yaml_filepath = os.path.join(".", "config", f"{args.experiment}.yaml")
with open(yaml_filepath, "r") as yamlfile:
    cfg = dynamic_yaml.load(yamlfile)
    print("==> Read YAML config file successfully ...")
seed = cfg['trainer'].get('seed', 42)
initialize(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

EPOCHS = cfg['trainer']['epochs'] 

print('==> Initialize wandb..')
name = cfg['wandb']['name']
wandb.init(project=cfg['wandb']['project'], name=cfg['wandb']['name'])
wandb.define_metric("epoch")
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("val/*", step_metric="epoch")
metrics = {}

old_sam_grads, old_sgd_grads = None, None
################################
#### 1. BUILD THE DATASET
################################
data_name = cfg['data']['name']
data_dict = get_dataloader(
    dataset=cfg['data']['name'],
    batch_size=cfg['data']['batch_size'], 
    num_workers=cfg['data']['num_workers'], 
    split=cfg['data']['split']
    )
train_dataloader, val_dataloader, test_dataloader, num_classes = data_dict['train_dataloader'], data_dict['val_dataloader'], data_dict['test_dataloader'], data_dict['num_classes']

################################
#### 2. BUILD THE NEURAL NETWORK
################################
net = get_model(cfg, num_classes=num_classes)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

total_params = sum(p.numel() for p in net.parameters())
print(f'==> Number of parameters in {cfg["model"]["architecture"]}: {total_params}')

################################
#### 3.a OPTIMIZING MODEL PARAMETERS
################################
criterion = nn.CrossEntropyLoss()
sch = cfg['trainer'].get('sch', None)
optimizer = get_optimizer(net, cfg)
scheduler = get_scheduler(optimizer, cfg)

################################
#### 3.b Training 
################################
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        enable_running_stats(net)  # <- this is the important line
        outputs = net(inputs)
        first_loss = criterion(outputs, targets)
        first_loss.backward()
        optimizer.first_step(zero_grad=False)
        
        sgd_grads = get_gradients(optimizer)
        optimizer.zero_grad()

        disable_running_stats(net)  # <- this is the important line
        criterion(net(inputs), targets).backward()
        optimizer.second_step(zero_grad=False)
        
        sam_grads = get_gradients(optimizer)
        optimizer.zero_grad()
        
        global old_sgd_grads, old_sam_grads
        if old_sgd_grads is None: 
            old_sgd_grads, old_sam_grads = sgd_grads, sam_grads
        else:
            dis1 = get_grad_norm([grad - old_grad for grad, old_grad in zip(sgd_grads, old_sgd_grads)])
            dis2 = get_grad_norm([grad - old_grad for grad, old_grad in zip(sam_grads, old_sam_grads)])
            wandb.log({'norm_dis_sgd': dis1, 'norm_dis_sam': dis2})
        
        with torch.no_grad():
            train_loss += first_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            train_loss_mean = train_loss/(batch_idx+1)
            acc = 100.*correct/total
            progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss_mean, acc, correct, total))
        
    metrics['train/loss'] = train_loss_mean
    metrics['train/acc'] = acc
    metrics['epoch'] = epoch

def val(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            val_loss_mean = val_loss/(batch_idx+1)
            acc = 100.*correct/total
            progress_bar(batch_idx, len(val_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (val_loss_mean, acc, correct, total))
        
    # Save checkpoint.
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'loss': val_loss,
            'epoch': epoch
        }
        if not os.path.isdir(f'checkpoint/{data_name}_{name}_{current_time}'):
            os.mkdir(f'checkpoint/{data_name}_{name}_{current_time}')
        torch.save(state, f'./checkpoint/{data_name}_{name}_{current_time}/ckpt_best.pth')
        best_acc = acc
    
    metrics['val/loss'] = val_loss_mean
    metrics['val/acc'] = acc
    metrics['val/best_acc'] = best_acc
    
def test():
    # Load checkpoint.
    print('==> Resuming from best checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{data_name}_{name}_{current_time}/ckpt_best.pth')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    wandb.log({
        'test/loss': test_loss/(len(test_dataloader)+1),
        'test/acc': 100.*correct/total,
        })

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+EPOCHS):
        train(epoch)
        val(epoch)
        wandb.log(metrics)
        scheduler.step()
    test()
    
        

