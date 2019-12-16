#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from capsnet import CapsNet
from dataloader import Dataset

random.seed(datetime.now())
torch.manual_seed(999)
torch.cuda.manual_seed(999)
torch.set_num_threads(16)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=bool, default=False, help='debug mode')
parser.add_argument('--pretrain', type=bool, default=False, help='load pretrain model')
parser.add_argument('--show', type=bool, default=False, help='show img')
parser.add_argument('--lr', type=float, default=3e-4, help='adam: learning rate')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='interval between model checkpoints')
parser.add_argument('--load_model', type=bool, default=True, help='whether to load model')
parser.add_argument('--model_num', type=int, default=0, help='number of model to load')
parser.add_argument('--epochs_num', type=int, default=1000, help='train epochs number')
parser.add_argument('--name', type=str, default='001/', help='model name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

opt = parser.parse_args()

def load_model():
    global capsule_net
    capsule_net.load_state_dict(torch.load(model_path +'%d/capsule_net.pth' % (opt.model_num), map_location=device))
    print('\033[1;32m [SUCCESS]:', 'successfully load model', str(opt.model_num)+'/capsule_net.pth', '\033[0m')
    
def save_model(step):
    global capsule_net
    os.makedirs(model_path + str(step), exist_ok=True)
    torch.save(capsule_net.state_dict(), model_path +'%d/capsule_net.pth' % (step))
    
class Config:
    def __init__(self, dataset='mnist'):
        self.batch_size = 64
        if dataset == 'mnist':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28

        elif dataset == 'cifar10':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'your own dataset':
            pass


def train():
    global capsule_net, dataset, optimizer, total_step
    train_loader = dataset.train_loader
    capsule_net.train()

    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        total_step += 1
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data).to(device), Variable(target).to(device)

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        if torch.isnan(loss):
            print('\033[1;31m [ERROR]:', 'loss is nan in step:', total_step, '\033[0m')
        loss.backward()
        torch.nn.utils.clip_grad_value_(capsule_net.parameters(), clip_value=20)
        optimizer.step()
        
        correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        
        logger.add_scalar('train/loss', loss.item()/opt.batch_size, total_step)
        logger.add_scalar('train/acc', correct/opt.batch_size, total_step)
            
        if total_step % 5 == 0:
            test()
            
        if total_step % 100 == 0:
            try:
                for name, param in capsule_net.named_parameters():
                    logger.add_histogram(name, param.clone().cpu().data.numpy(), total_step)
            except ValueError:
                print('\033[1;31m [ERROR]:', 'bad model parameters in step:', total_step, '\033[0m')
                
        if total_step % opt.checkpoint_interval == 0:
            save_model(total_step)

def test():
    global capsule_net, dataset, total_step
    test_loader = dataset.test_loader
    capsule_net.eval()
    
    dataiter = iter(test_loader)
    data, target = next(dataiter)
    target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
    data, target = Variable(data), Variable(target)

    data, target = data.to(device), target.to(device)

    output, reconstructions, masked = capsule_net(data)
    loss = capsule_net.loss(data, output, target, reconstructions)

    correct = sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                   np.argmax(target.data.cpu().numpy(), 1))
        
    logger.add_scalar('valid/loss', loss.item()/opt.batch_size, total_step)
    logger.add_scalar('valid/acc', correct/opt.batch_size, total_step)
        
    capsule_net.train()


if __name__ == '__main__':
    model_path = 'saved_models/'+opt.name
    log_path = 'log/'+opt.name
    os.makedirs(model_path, exist_ok=True)
            
    logger = SummaryWriter(log_dir=log_path)
        
    dataset_name = 'mnist' #'cifar10'
    config = Config(dataset_name)
    dataset = Dataset(dataset_name, opt.batch_size)

    capsule_net = CapsNet(config)
    if opt.load_model:
        try:
            load_model()
        except:
            print('\033[1;35m [WARNING]:', 'No model to load !', '\033[0m')
    capsule_net = capsule_net.to(device)
            
    optimizer = torch.optim.Adam(capsule_net.parameters(), lr=opt.lr, weight_decay=5e-5)
    
    total_step = 0
    for e in range(opt.epochs_num):
        train()