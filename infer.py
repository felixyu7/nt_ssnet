import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./paper.mplstyle')
import MinkowskiEngine as ME

from ic_ssnet import SparseIceCubeNet, SparseIceCubeResNet
from ic_dataset import SparseIceCubeDataset
from ic_dataset import ic_data_prep
from utils import angle_between, ic_collate_fn, get_p_of_bins, make_double_plot, get_mean_of_bins

import yaml
import glob
import os

# todo

with open("inference.cfg", 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

# data_files = sorted(glob.glob(cfg['data_dir'] + ("*.parquet")))
photons_data, nu_data = ic_data_prep(cfg['data_file'])

if cfg['event_list'] != '':
    event_list = np.loadtxt(cfg['event_list']).astype(np.int32)
    photons_data = photons_data[event_list]
    nu_data = nu_data[event_list]

# initialize network
if cfg['model'] == 'angular_reco':
    num_outputs = 3
if cfg['model'] == 'energy_reco':
    num_outputs = 1
if cfg ['model'] == 'both':
    num_outputs = 4

if cfg['traditional']:
    net = SparseIceCubeNet(1, num_outputs, expand=True).to(torch.device(cfg['device']))
else:
    net = SparseIceCubeResNet(1, num_outputs, reps=cfg['reps'], depth=cfg['depth'], first_num_filters=cfg['num_filters'], stride=cfg['stride'], dropout=0., D=4).to(torch.device(cfg['device']))

test_dataset = SparseIceCubeDataset(photons_data, nu_data, cfg['first_hit'])
# train_dataset, test_dataset = random_split(dataset, [len(dataset) - 10000, 10000], generator=torch.Generator().manual_seed(42))
# test_dataset = train_dataset
test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                         batch_size = cfg['batch_size'], 
                                         shuffle=False,
                                         collate_fn=ic_collate_fn,
                                         num_workers=len(os.sched_getaffinity(0)),
                                         pin_memory=True)

if cfg['model_weights'] != "":
    checkpoint = torch.load(cfg['model_weights'], map_location=torch.device(cfg['device']))
    net.load_state_dict(checkpoint['model_state_dict'])
 
preds = np.empty((0, num_outputs))
truth = np.empty((0, num_outputs))
true_e = torch.Tensor([])

import time
times = []
num_hits = []

# for i in range(1000):
#     time_get_item = time.time()
#     item = test_dataset.__getitem__(i)
#     times.append(time.time() - time_get_item)

# import pdb; pdb.set_trace()

for epoch in range(1):
    test_iter = iter(test_dataloader)

    # eval
    with torch.no_grad():
        net.eval()
        total_time = time.time()
        for i, data in enumerate(test_iter):

            coords, feats, labels = data

            true_e = torch.hstack((true_e, (10**((labels[:,0] * 2) + 4))))

            if cfg['model'] == 'angular_reco':
                labels = labels[:,1:].reshape(-1, 3).float().to(torch.device(cfg['device']))
            if cfg['model'] == 'energy_reco':
                labels = labels[:,0].reshape(-1, 1).float().to(torch.device(cfg['device']))
            if cfg['model'] == 'both':
                labels = labels.reshape(-1, 4).float().to(torch.device(cfg['device']))

            start = time.time()
            inputs = ME.SparseTensor(feats.float().reshape(coords.shape[0], -1), coords, 
                                     minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=torch.device(cfg['device']))
            start = time.time()
            out, inds = net(inputs)
            times.append(time.time() - start)
            num_hits.append(coords.shape[0])
            preds = np.vstack((preds, out.F[inds].cpu().numpy()))
            truth = np.vstack((truth, labels.cpu().numpy()))
            
total_time = time.time() - total_time
print(np.array(times).mean() / cfg['batch_size'])
print(total_time / len(test_dataset))

with open("gpu_timing.txt", "a+") as myfile:
    myfile.write(str(np.array(times).mean() / cfg['batch_size']) + '\n')
    myfile.write(str(np.array(times).std() / cfg['batch_size']))

bins = np.logspace(2, 6, 20, endpoint=False)

if cfg['model'] == 'angular_reco':
    angle_diff = []
    for i in range(preds.shape[0]):
        angle_diff.append(angle_between(preds[i], truth[i]))

    print(np.array(angle_diff).mean())
    angle_diff = np.array(angle_diff)

    preds = preds / np.linalg.norm(preds, axis=1).reshape(-1, 1)

    m_ad = get_p_of_bins(angle_diff, true_e, bins, 50)
    p20_ad = get_p_of_bins(angle_diff, true_e, bins, 20)
    p80_ad = get_p_of_bins(angle_diff, true_e, bins, 80)
    mean_ad = get_mean_of_bins(angle_diff, true_e, bins)

    import pdb; pdb.set_trace()

elif cfg['model'] == 'energy_reco':
    preds = (preds.flatten() * 2) + 4
    truth = (truth.flatten() * 2) + 4
    diff = np.abs(preds - truth)
    m_diff = get_p_of_bins(diff, true_e, bins, 50)
    p20_diff = get_p_of_bins(diff, true_e, bins, 20)
    p80_diff = get_p_of_bins(diff, true_e, bins, 80)
    import pdb; pdb.set_trace()

elif cfg['model'] == 'both':
    preds_E = (preds[:,0] * 2) + 4
    preds_A = preds[:,1:]
    truth_E = (truth[:,0] * 2) + 4
    truth_A = truth[:,1:]

    angle_diff = []
    for i in range(preds.shape[0]):
        angle_diff.append(angle_between(preds_A[i], truth_A[i]))

    print(np.array(angle_diff).mean())
    angle_diff = np.array(angle_diff)

    m_ad = get_p_of_bins(angle_diff, true_e, bins, 50)
    p20_ad = get_p_of_bins(angle_diff, true_e, bins, 20)
    p80_ad = get_p_of_bins(angle_diff, true_e, bins, 80)
    mean_ad = get_mean_of_bins(angle_diff, true_e, bins)

    diff = np.abs(preds_E - truth_E)
    m_diff = get_p_of_bins(diff, true_e, bins, 50)
    p20_diff = get_p_of_bins(diff, true_e, bins, 20)
    p80_diff = get_p_of_bins(diff, true_e, bins, 80)

    res = {'m_ad': m_ad, 'p20_ad': p20_ad, 'p80_ad': p80_ad, 'mean_ad': mean_ad, 'm_diff': m_diff, 'p20_diff': p20_diff, 'p80_diff': p80_diff}
    import pdb; pdb.set_trace()