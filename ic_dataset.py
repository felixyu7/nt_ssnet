""" ic_dataset.py - Prometheus IceCube dataset processor for SSCNN
    Felix J. Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME

import awkward as ak

import glob
import re
import time

# sparse icecube dataset (Prometheus)
class SparseIceCubeDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        photons_data,
        nu_data,
        first_hit,
        training=True):

        self.data = photons_data
        self.nu_data = nu_data
        self.dataset_size = len(self.data)
        self.first_hit = first_hit
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        
        # [energy, zenith, azimuth]
        label = [self.nu_data[i][0], self.nu_data[i][1], self.nu_data[i][2]]

        xs = self.data[i].filtered.sensor_pos_x.to_numpy() * 4.566
        ys = self.data[i].filtered.sensor_pos_y.to_numpy() * 4.566
        zs = self.data[i].filtered.sensor_pos_z.to_numpy() * 4.566
        ts = self.data[i].filtered.t.to_numpy() - self.data[i].filtered.t.to_numpy().min()

        pos_t = np.array([
            xs,
            ys,
            zs,
            ts
        ]).T

        # only use first photon hit time per dom
        if self.first_hit:
            spos_t = pos_t[np.argsort(pos_t[:,-1])]
            _, indices, feats = np.unique(spos_t[:,:3], axis=0, return_index=True, return_counts=True)
            pos_t = spos_t[indices]
            pos_t = np.trunc(pos_t)
            # feats = (feats - feats.mean()) / (feats.std() + 1e-8)
        else:
            pos_t = np.trunc(pos_t)
            pos_t, feats = np.unique(pos_t, return_counts=True, axis=0)
            # feats = (feats - feats.mean()) / (feats.std() + 1e-8)

        feats = feats.reshape(-1, 1).astype(np.float64)

        x = np.cos(label[2]) * np.sin(label[1])
        y = np.sin(label[2]) * np.sin(label[1])
        z = np.cos(label[1])
        label = [label[0], x, y, z]

        return torch.from_numpy(pos_t), torch.from_numpy(feats).view(-1, 1), torch.from_numpy(np.array([label]))

def ic_data_prep(data_file):

    """ function to convert read files into inputs
        args: config dict, list of photon parquet files to use
        returns: awkward array of photon hit information, numpy array of true neutrino information
    """

    tsime = time.time()

    photons_data = ak.from_parquet(data_file, columns=["mc_truth", "filtered"])

    print("total time:", time.time() - tsime)

    # converting read data to inputs
    es = np.array(photons_data['mc_truth']['injection_energy'])
    zenith = np.array(photons_data['mc_truth']['injection_zenith'])
    azimuth = np.array(photons_data['mc_truth']['injection_azimuth'])

    # energy transforms/normalization
    es_norm = (np.log(1 + es) - np.log(1 + es).mean()) / np.log(1 + es).std()

    nu_data = np.dstack((((np.log10(es) - 4) / 2), zenith, azimuth)).reshape(-1, 3)

    return photons_data, nu_data