import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME

import awkward as ak
import h5py

import glob
import re
import time

# sparse icecube dataset (Prometheus)
class SparseIceCubeDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        photons_data,
        nu_data,
        training=True):

        self.data = photons_data
        self.nu_data = nu_data
        self.dataset_size = len(self.data)
        

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        
        t = []
        pos = []
        
        primary_truth = self.nu_data
        
        # [energy, zenith, azimuth]
        label = [primary_truth[i][0], primary_truth[i][1], primary_truth[i][2]]
        
        if (-1 not in self.data[i]['primary_lepton_1']['t']):
            for x, y, z, time in zip(self.data[i]['primary_lepton_1']['sensor_pos_x'], self.data[i]['primary_lepton_1']['sensor_pos_y'], self.data[i]['primary_lepton_1']['sensor_pos_z'], self.data[i]['primary_lepton_1']['t']):
                t.append(time)
                pos.append([x, y, z])
                
        if (-1 not in self.data[i]['primary_hadron_1']['t']):  
            for x, y, z, time in zip(self.data[i]['primary_hadron_1']['sensor_pos_x'], self.data[i]['primary_hadron_1']['sensor_pos_y'], self.data[i]['primary_hadron_1']['sensor_pos_z'], self.data[i]['primary_hadron_1']['t']):
                t.append(time)
                pos.append([x, y, z])

        # 3 ns for light to travel 1 m
        pos = np.array(pos) * 3.
        t = np.array(t).reshape(-1, 1)
        pos_t = np.hstack((pos, t))
        pos_t = np.trunc(pos_t)
        
        pos_t, feats = np.unique(pos_t, return_counts=True, axis=0)
        feats = feats.reshape(-1, 1).astype(np.float64)
        
        return torch.Tensor(pos_t), torch.Tensor(feats).view(-1, 1), torch.Tensor([label])

# function to convert read files into inputs
# args: config dict, list of photon parquet files to use
# returns: awkward array of photon hit information, numpy array of true neutrino information
def ic_data_prep(cfg, data_files):
    total_time = time.time()

    photons_data = ak.Array([])
    file_loop = time.time()

    for photon_file in data_files:
        pd = ak.from_parquet(photon_file)
        photons_data = ak.concatenate((photons_data, pd), axis=0)

    print("fileloop time:", time.time() - file_loop)

    # remove null events
    keep_indices = []

    for i in range(len(photons_data)):
        if ((len(photons_data[i]['primary_lepton_1']['t']) + len(photons_data[i]['primary_hadron_1']['t'])) > cfg['hits_min']) and ((len(photons_data[i]['primary_lepton_1']['t']) + len(photons_data[i]['primary_hadron_1']['t'])) < cfg['hits_max']):
            keep_indices.append(i)
            
    photons_data = photons_data[keep_indices]
    print("total time:", time.time() - total_time)

    # converting read data to inputs
    es = []
    zenith = []
    azimuth = []
    for i in range(len(photons_data)):
        es.append(photons_data[i]['mc_truth']['injection_energy'])
        zenith.append(photons_data[i]['mc_truth']['injection_zenith'])
        azimuth.append(photons_data[i]['mc_truth']['injection_azimuth'])
    es = np.array(es)
    zenith = np.array(zenith)
    azimuth = np.array(azimuth)

    # energy transforms/normalization
    es_norm = (np.log(1 + es) - np.log(1 + es).mean()) / np.log(1 + es).std()
    es_tev = es / 1000.

    nu_data = np.dstack((es_norm, zenith, azimuth)).reshape(-1, 3)

    return photons_data, nu_data