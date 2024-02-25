'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime

import os
import argparse
import yaml
import pprint

from torch.utils.tensorboard import SummaryWriter

from dnsam.models import *
from dnsam.utils import *
from dnsam.data import *
from dnsam.scheduler import *

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--experiment', default='example', type=str, help='path to YAML config file')
args = parser.parse_args()

yaml_filepath = os.path.join(".", "config", f"{args.experiment}.yaml")
with open(yaml_filepath, "r") as yamlfile:
    cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("==> Read YAML config file successfully ...")
    pprint.pprint(cfg)
seed = cfg['trainer'].get('seed', 42)
initialize(seed)

name = cfg['name']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc, start_epoch = 0, 0  # best test accuracy, start from epoch 0 or last checkpoint epoch
EPOCHS = cfg['trainer']['epochs'] 

# Initialize TensorBoard
print('==> Initialize TensorBoard..')
writer = SummaryWriter('runs/' + name + '_' + current_time)

# Data
data_name = cfg['data']['name']
data_dict = get_dataloader(dataset=cfg['data']['name'], batch_size=cfg['data']['batch_size'], num_workers=cfg['data']['num_workers'], split=cfg['data']['split'])

print(f"==> Loading dataset: {data_name}")
train_dataloader, val_dataloader, test_dataloader, num_classes = data_dict['train_dataloader'], data_dict['val_dataloader'], data_dict['test_dataloader'], data_dict['num_classes']

# Model
print(f'==> Loading model: {cfg["model"]["architecture"]}')
net = get_model(cfg, num_classes=num_classes)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

total_params = sum(p.numel() for p in net.parameters())
print(f'==> Number of parameters in {cfg["model"]["architecture"]}: {total_params}')

# layer_names = [name for name, _ in net.named_parameters()]
# print(layer_names)
# raise exit
idx_inc_rho = 16
adaptive_rho = [0.05 for name, _ in net.named_parameters()][:-idx_inc_rho]
adaptive_rho = adaptive_rho + [1 for i in range(idx_inc_rho)]

criterion = nn.CrossEntropyLoss()

sch = cfg['trainer'].get('sch', None)
print(f"==> Loading optimizer: {cfg['model']['name']}")
print(f"==> Loading scheduler: {sch}")

base_optimizer = optim.SGD
optimizer = get_optimizer(net, base_optimizer, cfg, adaptive_rho)
scheduler = get_scheduler(optimizer, cfg)

warmup_flag = cfg['trainer'].get('warmup', None)
if warmup_flag is not None:
    rho_scheduler = RhoScheduler(optimizer, cfg['model']['rho'], warmup_epochs=cfg['trainer']['warmup'], total_epochs=EPOCHS)
    rho_scheduler.step(-1)
    
# Training
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
        optimizer.zero_grad()

        disable_running_stats(net)  # <- this is the important line
        criterion(net(inputs), targets).backward()
        optimizer.second_step(zero_grad=False)
            
        train_loss += first_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_loss_mean = train_loss/(batch_idx+1)
        acc = 100.*correct/total
        progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss_mean, acc, correct, total))
        
    # log weights & gradients
    if (epoch+1) % 20 == 0 or epoch == 0: 
        for name, param in net.named_parameters():
            writer.add_histogram('weights/' + name, param, epoch)
            writer.add_histogram('gradients/' + name, param.grad, epoch)
        
    writer.add_scalar('Train/loss', train_loss_mean, epoch)
    writer.add_scalar('Train/acc', acc, epoch)

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
    if (epoch+1) % 100 == 0:
        print(f'Saving epoch {epoch+1}..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'loss': val_loss,
            'epoch': epoch
        }
        if not os.path.isdir(f'checkpoint/{data_name}_{name}_{current_time}'):
            os.mkdir(f'checkpoint/{data_name}_{name}_{current_time}')
        torch.save(state, f'./checkpoint/{data_name}_{name}_{current_time}/{epoch+1}.pth')
        
    writer.add_scalar('Val/loss', val_loss_mean, epoch)
    writer.add_scalar('Val/acc', acc, epoch)
    writer.add_scalar('Val/best_acc', best_acc, epoch)
    
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

    writer.add_scalar('Test/loss', test_loss/(len(test_dataloader)+1))
    writer.add_scalar('Test/acc', 100.*correct/total)

if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+EPOCHS):
        train(epoch)
        val(epoch)
        scheduler.step()
        if warmup_flag is not None:
            rho_scheduler.step(epoch)
    test()
    
        

