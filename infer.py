import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import MinkowskiEngine as ME

from ic_ssnet import SparseIceCubeNet
from ic_dataset import NewSparseIceCubeDataset
from ic_dataset import new_ic_data_prep
from ic_collate import ic_collate_fn

import yaml
import glob

# todo

with open("inference.cfg", 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

data_files = sorted(glob.glob(cfg['data_dir'] + ("*.parquet")))[0:40]

photons_data, nu_data = new_ic_data_prep(cfg, data_files)

# initialize network
net = SparseIceCubeNet(1, 1, expand=cfg['expand'], D=4).to(torch.device(cfg['device']))

test_dataset = NewSparseIceCubeDataset(photons_data, nu_data)
test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                         batch_size = cfg['batch_size'], 
                                         shuffle=False,
                                         collate_fn=ic_collate_fn)

if cfg['model_weights'] != "":
    checkpoint = torch.load(cfg['model_weights'], map_location=torch.device(cfg['device']))
    net.load_state_dict(checkpoint['model_state_dict'])

preds = torch.Tensor([])
truth = torch.Tensor([])
true_e = torch.Tensor([])

if cfg['expand']:
    algorithm = ME.MinkowskiAlgorithm.MEMORY_EFFICIENT
else:
    algorithm = ME.MinkowskiAlgorithm.SPEED_OPTIMIZED

import time
times = []
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
            out, inds = net(inputs)
            pred = out.F[inds][:,0]
            times.append(time.time() - start)
            preds = torch.hstack((preds, pred.cpu()))
            truth = torch.hstack((truth, labels[:,1]))
            true_e = torch.hstack((true_e, labels[:,0]))
            
total_time = time.time() - total_time
print(np.array(times).mean() / cfg['batch_size'])
print(total_time / len(test_dataset))

import matplotlib.pyplot as plt

cos_zenith_pred = np.cos(preds)
cos_zenith_truth = np.cos(truth)
cos_diff = np.array(cos_zenith_pred - cos_zenith_truth)

plt.hist(cos_diff, bins=100)
plt.xlim([-1.5, 1.5])
plt.xlabel('Reconstructed - Truth (cos(zenith))')
plt.ylabel('Num Events')
plt.savefig("./result.png")

print(np.sqrt(np.mean(cos_diff**2)))
print(np.median(cos_diff))