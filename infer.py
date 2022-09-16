import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import MinkowskiEngine as ME

from ic_ssnet import SparseIceCubeNet
from ic_dataset import SparseIceCubeDataset
from ic_dataset import ic_data_prep
from utils import angle_between, ic_collate_fn

import yaml
import glob
import os

# todo

with open("inference.cfg", 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

# data_files = sorted(glob.glob(cfg['data_dir'] + ("*.parquet")))
photons_data, nu_data = ic_data_prep(cfg['data_file'])

# initialize network
if cfg['pred_cartesian_direction']:
    net = SparseIceCubeNet(1, 3, expand=cfg['expand'], D=4).to(torch.device(cfg['device']))
else: 
    net = SparseIceCubeNet(1, 1, expand=cfg['expand'], D=4).to(torch.device(cfg['device']))    

test_dataset = SparseIceCubeDataset(photons_data[:10000], nu_data[:10000], cfg['pred_cartesian_direction'], cfg['first_hit'])
test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                         batch_size = cfg['batch_size'], 
                                         shuffle=False,
                                         collate_fn=ic_collate_fn,
                                         num_workers=len(os.sched_getaffinity(0)),
                                         pin_memory=True)

if cfg['model_weights'] != "":
    checkpoint = torch.load(cfg['model_weights'], map_location=torch.device(cfg['device']))
    net.load_state_dict(checkpoint['model_state_dict'])

preds = np.empty((0, 3))
truth = np.empty((0, 3))
true_e = torch.Tensor([])

if cfg['expand']:
    algorithm = ME.MinkowskiAlgorithm.MEMORY_EFFICIENT
else:
    algorithm = ME.MinkowskiAlgorithm.SPEED_OPTIMIZED

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

            start = time.time()
            inputs = ME.SparseTensor(feats.float().reshape(coords.shape[0], -1), coords, 
                                     minkowski_algorithm=algorithm, device=torch.device(cfg['device']))
            start = time.time()
            out, inds = net(inputs)
            times.append(time.time() - start)
            num_hits.append(coords.shape[0])
            pred = out.F[inds]
            preds = np.vstack((preds, pred.cpu().numpy()))
            truth = np.vstack((truth, labels[:,1:].cpu().numpy()))
            true_e = torch.hstack((true_e, labels[:,0]))
            
total_time = time.time() - total_time
print(np.array(times).mean() / cfg['batch_size'])
print(total_time / len(test_dataset))

angle_diff = []
for i in range(preds.shape[0]):
    angle_diff.append(angle_between(preds[i][1:], truth[i][1:]))

print(np.array(angle_diff).mean())

import matplotlib.pyplot as plt

preds = preds / np.linalg.norm(preds, axis=1).reshape(-1, 1)

# cos_zenith_pred = np.cos(preds)
# cos_zenith_truth = np.cos(truth)
cos_zenith_pred = preds[:,2]
cos_zenith_truth = truth[:,2]
diff = np.array(cos_zenith_pred - cos_zenith_truth)

plt.hist(diff, bins=100)
plt.xlim([-1.5, 1.5])
plt.xlabel('Reconstructed - Truth (cos(zenith))')
plt.ylabel('Num Events')
plt.savefig("./result.png")

print(np.sqrt(np.mean(diff**2)))
print(np.median(diff))

# import pdb; pdb.set_trace()