import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME

import awkward as ak
import h5py

import glob
import re
import os

from ic_ssnet import SparseIceCubeNet
from ic_dataset import SparseIceCubeDataset
from ic_dataset import ic_data_prep
from ic_collate import ic_collate_fn

import yaml
import csv

with open("train.cfg", 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

# load data from files
data_files = sorted(glob.glob(cfg['data_dir'] + ("*.parquet")))[20:]

photons_data, nu_data = ic_data_prep(cfg, data_files)

# initialize network
net = SparseIceCubeNet(1, 1, expand=cfg['expand'], D=4).to(torch.device(cfg['device']))

dataset = SparseIceCubeDataset(photons_data, nu_data)
data_train, data_valid = torch.utils.data.random_split(dataset, [len(dataset) - cfg['validation_set_size'], cfg['validation_set_size']])
train_dataloader = torch.utils.data.DataLoader(data_train, 
                                         batch_size = cfg['batch_size'], 
                                         shuffle=True,
                                         collate_fn=ic_collate_fn)
valid_dataloader = torch.utils.data.DataLoader(data_valid, 
                                         batch_size = cfg['batch_size'], 
                                         shuffle=False,
                                         collate_fn=ic_collate_fn)


optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=1e-5)
criterion = nn.MSELoss()

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

if cfg['expand']:
    algorithm = ME.MinkowskiAlgorithm.MEMORY_EFFICIENT
else:
    algorithm = ME.MinkowskiAlgorithm.SPEED_OPTIMIZED

# Training loop
for epoch in range(epoch_start, cfg['epochs']):
    train_iter = iter(train_dataloader)

    # Training
    net.train()
    for i, data in enumerate(train_iter):
        coords, feats, labels = data
        
        inputs = ME.SparseTensor(feats.float().reshape(coords.shape[0], -1), coords, device=torch.device(cfg['device']),
                                 minkowski_algorithm=algorithm, requires_grad=True)
        

        out, inds = net(inputs)
        optimizer.zero_grad()
        
        labels = labels[:,0].reshape(-1, 1).float().to(torch.device(cfg['device']))
        preds = out.F
        loss = criterion(preds, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.)
        optimizer.step()

        with open(cfg['logs_file'], 'a+') as logs_file:
            writer = csv.writer(logs_file)
            writer.writerow([tot_iter, (tot_iter * coords.shape[0]) / len(data_train), loss.item()])

        accum_loss += loss.item()
        accum_iter += 1
        tot_iter += 1

        if tot_iter % 1 == 0 or tot_iter == 1:
            print(
                f'Iter: {tot_iter}, Epoch: {epoch}, Loss: {accum_loss / accum_iter}'
            )
            accum_loss, accum_iter = 0, 0
            
        # if tot_iter % 1000 == 0:
        #     with torch.no_grad():
        #         valid_iter = iter(valid_dataloader)
        #         net.eval()
        #         valid_loss = 0
        #         valid_iters = 0
        #         for i, vdata in enumerate(valid_iter):
        #             vcoords, vfeats, vlabels = vdata
        #             vinputs = ME.SparseTensor(vfeats.float().reshape(vcoords.shape[0], -1), vcoords, 
        #                    device=torch.device(cfg['device']), requires_grad=True)

        #             vout, inds = net(vinputs)
        #             vlabels = vlabels[:,0].reshape(-1, 1).float().to(torch.device(cfg['device']))
        #             vpreds = vout.F
        #             valid_loss += criterion(vpreds, vlabels)
        #             valid_iters += 1
        #         print("validation loss: ", valid_loss / valid_iters)
        #     net.train()

    torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': tot_iter,
                }, cfg['ckpt_dir'] + 'epoch_' + str(epoch) + '.ckpt')