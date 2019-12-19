#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path as path
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from dataloader import MyDataset
from torch.utils.data import DataLoader
from stats import Statistics
from utils import *
from constants import *
from model import CapsNet
from options import create_options
from tqdm import tqdm

def get_alpha(epoch):
    # WARNING: Does not support alpha value saving when continuning training from a saved model
    if opts.anneal_alpha == "none":
        alpha = opts.alpha
    if opts.anneal_alpha == "1":
        alpha = opts.alpha * float(np.tanh(epoch/DEFAULT_ANNEAL_TEMPERATURE - np.pi) + 1) / 2
    if opts.anneal_alpha == "2":
        alpha = opts.alpha * float(np.tanh(epoch/(2 * DEFAULT_ANNEAL_TEMPERATURE)))
    return alpha

def onehot(tensor, num_classes=10):
    return torch.eye(num_classes).cuda().index_select(dim=0, index=tensor) # One-hot encode 

def transform_data(data,target,use_gpu, num_classes=10):
    data, target = Variable(data), Variable(target)
    if use_gpu:
        data, target = data.cuda(), target.cuda()
    target = onehot(target, num_classes=num_classes)
    return data, target

class GPUParallell(nn.DataParallel):
  def __init__(self, capsnet, device_ids):
    super(GPUParallell, self).__init__(capsnet, device_ids=device_ids)
    self.capsnet = capsnet
    self.num_classes = capsnet.num_classes
    
  def loss(self, images,labels, capsule_output,  reconstruction): 
    return self.capsnet.loss(images, labels, capsule_output, reconstruction)
  
  def forward(self, x, target=None):
    return self.capsnet(x, target)

def get_network(opts):
    if opts.dataset == "yqdataset":
        capsnet = CapsNet(reconstruction_type=opts.decoder,
                          imsize=128,
                          num_classes=7,
                          routing_iterations = opts.routing_iterations,
                          primary_caps_gridsize=8,
                          img_channels=3, 
                          batchnorm=opts.batch_norm,
                          num_primary_capsules=64,
                          loss=opts.loss_type,
                          leaky_routing=opts.leaky_routing)
    if opts.use_gpu:
        capsnet.cuda()
    if opts.gpu_ids:
        capsnet = GPUParallell(capsnet, opts.gpu_ids)
        print("Training on GPU IDS:", opts.gpu_ids)
    return capsnet

def load_model(opts, capsnet): 
    model_path = path.join(SAVE_DIR, opts.filepath)
    if path.isfile(model_path):
        print("Saved model found")
        capsnet.load_state_dict(torch.load(model_path))
    else:
        print("Saved model not found; Model initialized.")
        initialize_weights(capsnet)
    
def main(opts):
    capsnet = get_network(opts)

    optimizer = torch.optim.Adam(capsnet.parameters(), lr=opts.learning_rate)

    """ Load saved model"""
    load_model(opts, capsnet)

    ds = MyDataset(file_path='animation.txt')
    dataloader = DataLoader(dataset=ds,
                            batch_size=opts.batch_size,
                            shuffle=True,
                            num_workers=16)
    
    ds_valid = MyDataset(file_path='animation_valid.txt', if_valid=True)
    dataloader_valid = DataLoader(dataset=ds_valid,
                              batch_size=opts.batch_size,
                              shuffle=True,
                              num_workers=16)
    
    train_loader = dataloader
    test_loader = dataloader_valid
    
    stats = Statistics(LOG_DIR, opts.model)
    
    for epoch in range(opts.epochs):
        capsnet.train()
        
        # Annealing alpha
        alpha = get_alpha(epoch)

        for batch, (data, target) in tqdm(list(enumerate(train_loader)), ascii=True, desc="Epoch{:3d}".format(epoch)):
            optimizer.zero_grad()
            data, target = transform_data(data, target, opts.use_gpu, num_classes=capsnet.num_classes)
            capsule_output, reconstructions, _ = capsnet(data, target)
            predictions = torch.norm(capsule_output.squeeze(), dim=2)
            data = denormalize(data)
            loss, rec_loss, marg_loss = capsnet.loss(data, target, capsule_output, reconstructions, alpha)
            loss.backward()
            optimizer.step()
            stats.track_train(loss.data.detach().item(), rec_loss.detach().item(), marg_loss.detach().item(), target.detach(), predictions.detach())
        
        # Evaluate on test set
        capsnet.eval()
        for batch_id, (data, target) in tqdm(list(enumerate(test_loader)), ascii=True, desc="Test {:3d}".format(epoch)):
            data, target = transform_data(data, target, opts.use_gpu, num_classes=capsnet.num_classes)
            capsule_output, reconstructions, predictions = capsnet(data)
            data = denormalize(data)
            loss, rec_loss, marg_loss = capsnet.loss(data, target, capsule_output, reconstructions, alpha)
            stats.track_test(loss.data.detach().item(),rec_loss.detach().item(), marg_loss.detach().item(), target.detach(), predictions.detach())

        stats.save_stats(epoch)

        # Save reconstruction image from testing set
        if opts.save_images:
            #data, target = iter(test_loader).next()
            data, target = iter(train_loader).next()
            data, _ = transform_data(data, target, opts.use_gpu)
            _, reconstructions, _ = capsnet(data)
            filename = "reconstruction_epoch_{}.png".format(epoch)
            #if opts.dataset == 'cifar10':
            #    save_images_cifar10(IMAGES_SAVE_DIR, filename, data, reconstructions)
            #else:
            #    save_images(IMAGES_SAVE_DIR, filename, data, reconstructions, imsize=capsnet.imsize)
            save_images(IMAGES_SAVE_DIR, filename, data, reconstructions, imsize=capsnet.imsize)

        # Save model
        model_path = get_path(SAVE_DIR, "model{}.pt".format(epoch))
        torch.save(capsnet.state_dict(), model_path)
        capsnet.train()


if __name__ == '__main__':
    opts = create_options()
    main(opts)
