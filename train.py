""" train.py - Training script for SSCNN
    Felix J. Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import MinkowskiEngine as ME
import awkward as ak

import glob
import re
import os

from ic_ssnet import SparseIceCubeResNet
from ic_dataset import SparseIceCubeDataset
from ic_dataset import ic_data_prep
from utils import LogCoshLoss, ic_collate_fn, AngularDistanceLoss, CombinedAngleEnergyLoss

import yaml
import csv

with open("train.cfg", 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

# load data from files
t_photons_data, t_nu_data = ic_data_prep(cfg['train_data_file'])
v_photons_data, v_nu_data = ic_data_prep(cfg['valid_data_file'])

if cfg['train_event_list'] != '':
    event_list = np.loadtxt(cfg['train_event_list']).astype(np.int32)
    t_photons_data = t_photons_data[event_list]
    t_nu_data = t_nu_data[event_list]

if cfg['valid_event_list'] != '':
    event_list = np.loadtxt(cfg['valid_event_list']).astype(np.int32)
    v_photons_data = v_photons_data[event_list]
    v_nu_data = v_nu_data[event_list]

if cfg['model'] == 'angular_reco':
    num_outputs = 3
elif cfg['model'] == 'energy_reco':
    num_outputs = 1
elif cfg ['model'] == 'both':
    num_outputs = 4
else:
    print("Unknown model type! Use angular_reco, energy_reco, or both.")
    exit()

# initialize network
net = SparseIceCubeResNet(1, num_outputs, 
                            reps=cfg['reps'], 
                            depth=cfg['depth'], 
                            first_num_filters=cfg['num_filters'], 
                            stride=cfg['stride'], 
                            dropout=0., 
                            D=4).to(torch.device(cfg['device']))

train_dataset = SparseIceCubeDataset(t_photons_data, t_nu_data, cfg['first_hit'])
valid_dataset = SparseIceCubeDataset(v_photons_data, v_nu_data, cfg['first_hit'])

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                        batch_size = cfg['batch_size'], 
                                        shuffle=True,
                                        collate_fn=ic_collate_fn,
                                        num_workers=len(os.sched_getaffinity(0)),
                                        pin_memory=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
                                        batch_size = cfg['batch_size'], 
                                        shuffle=False,
                                        collate_fn=ic_collate_fn,
                                        num_workers=len(os.sched_getaffinity(0)),
                                        pin_memory=True)

print(len(train_dataset))
optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

if cfg['model'] == 'angular_reco':
    criterion = AngularDistanceLoss
if cfg['model'] == 'energy_reco':
    criterion = LogCoshLoss
if cfg['model'] == 'both':
    criterion = CombinedAngleEnergyLoss

tot_iter = 0
tot_time = 0
accum_iter = 0
accum_loss = 0

epoch_start = 0
if cfg['model_weights'] != "":
    checkpoint = torch.load(cfg['model_weights'], map_location=torch.device(cfg['device']))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']
    tot_iter = checkpoint['global_step']

# clear logs file and make new ckpts dir if needed
with open(cfg['logs_file'], 'w+') as f:
    writer = csv.writer(f)
    # writer.writerow(['iter', 'epoch', 'train_loss'])

if not os.path.exists(cfg['ckpt_dir']):
    os.makedirs(cfg['ckpt_dir'])

# Training loop
for epoch in range(epoch_start, cfg['epochs']):
    train_iter = iter(train_dataloader)

    # Training
    net.train()
    for i, data in enumerate(train_iter):

        coords, feats, labels = data
        
        inputs = ME.SparseTensor(feats.float().reshape(coords.shape[0], -1), coords, device=torch.device(cfg['device']),
                                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, requires_grad=True)
        
        out, inds = net(inputs)
        optimizer.zero_grad()

        if cfg['model'] == 'angular_reco':
            labels = labels[:,1:].reshape(-1, 3).float().to(torch.device(cfg['device']))
        if cfg['model'] == 'energy_reco':
            labels = labels[:,0].reshape(-1, 1).float().to(torch.device(cfg['device']))
        if cfg['model'] == 'both':
            labels = labels.reshape(-1, 4).float().to(torch.device(cfg['device']))

        preds = out.F
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        with open(cfg['logs_file'], 'a+') as logs_file:
            writer = csv.writer(logs_file)
            writer.writerow([tot_iter, (tot_iter * coords.shape[0]) / len(train_dataset), loss.item()])

        accum_loss += loss.item()
        accum_iter += 1
        tot_iter += 1

        if tot_iter % 100 == 0 or tot_iter == 1:
            print(
                f'Iter: {tot_iter}, Epoch: {epoch}, Loss: {accum_loss / accum_iter}'
            )
            accum_loss, accum_iter = 0, 0
            # solve memory issues
            torch.cuda.empty_cache()
        
        # validation run after every 500 iterations
        if tot_iter % 500 == 0:
            with torch.no_grad():
                valid_iter = iter(valid_dataloader)
                net.eval()
                valid_loss = 0
                valid_iters = 0
                for i, vdata in enumerate(valid_iter):
                    vcoords, vfeats, vlabels = vdata
                    vinputs = ME.SparseTensor(vfeats.float().reshape(vcoords.shape[0], -1), vcoords, 
                        device=torch.device(cfg['device']), requires_grad=True)

                    vout, inds = net(vinputs)
                    if cfg['model'] == 'angular_reco':
                        vlabels = vlabels[:,1:].reshape(-1, 3).float().to(torch.device(cfg['device']))
                    if cfg['model'] == 'energy_reco':
                        vlabels = vlabels[:,0].reshape(-1, 1).float().to(torch.device(cfg['device']))
                    if cfg['model'] == 'both':
                        vlabels = vlabels.reshape(-1, 4).float().to(torch.device(cfg['device']))
                        
                    vpreds = vout.F
                    valid_loss += criterion(vpreds, vlabels)
                    valid_iters += 1
                valid_loss = valid_loss / valid_iters
                print("validation loss: ", valid_loss)
            net.train()
    scheduler.step()
    
    # save every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'global_step': tot_iter,
                    }, cfg['ckpt_dir'] + 'epoch_' + str(epoch) + '.ckpt')
