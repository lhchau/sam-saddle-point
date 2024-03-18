import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import datetime

import os
import argparse
import dynamic_yaml
import pprint
from torch.utils.tensorboard import SummaryWriter

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

# Initialize TensorBoard
print('==> Initialize TensorBoard..')
name = cfg['wandb']['name']
writer = SummaryWriter('runs/' + name + '_' + current_time)

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
step = 0
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
        
        sgd_grads = get_gradients(optimizer)
        optimizer.first_step(zero_grad=False)
        optimizer.zero_grad()

        disable_running_stats(net)  # <- this is the important line
        criterion(net(inputs), targets).backward()
        
        sam_grads = get_gradients(optimizer)
        optimizer.second_step(zero_grad=False)
        optimizer.zero_grad()
        
        global step
        writer.add_scalar('similarity', np.mean([cosine_similarity(sgd_grad, sam_grad) for sgd_grad, sam_grad in zip(sgd_grads, sam_grads)]), step)
        writer.add_scalar('step_norm_before_hess', optimizer.step_norm_before_hess, step)
        writer.add_scalar('step_norm', optimizer.step_norm, step)
        step += 1
        
        with torch.no_grad():
            train_loss += first_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            train_loss_mean = train_loss/(batch_idx+1)
            acc = 100.*correct/total
            progress_bar(batch_idx, len(train_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss_mean, acc, correct, total))
        
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
        writer.add_scalar('learning_rate', scheduler.get_last_lr()[-1]) 
        scheduler.step()
    test()