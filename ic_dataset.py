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
        pred_cartesian_direction,
        first_hit,
        training=True):

        self.data = photons_data
        self.nu_data = nu_data
        self.dataset_size = len(self.data)
        self.pred_cartesian_direction = pred_cartesian_direction
        self.first_hit = first_hit
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        
        # [energy, zenith, azimuth]
        label = [self.nu_data[i][0], self.nu_data[i][1], self.nu_data[i][2]]

        if self.data[i]['primary_lepton_1']['t'][0] != -1:
            xs = ak.to_numpy(self.data[i]['primary_lepton_1']['sensor_pos_x']).reshape(-1, 1)
            ys = ak.to_numpy(self.data[i]['primary_lepton_1']['sensor_pos_y']).reshape(-1, 1)
            zs = ak.to_numpy(self.data[i]['primary_lepton_1']['sensor_pos_z']).reshape(-1, 1)
            ts = ak.to_numpy(self.data[i]['primary_lepton_1']['t']).reshape(-1, 1)
            if self.data[i]['primary_hadron_1']['t'][0] != -1:
                xs = np.concatenate((xs, ak.to_numpy(self.data[i]['primary_hadron_1']['sensor_pos_x']).reshape(-1, 1)))
                ys = np.concatenate((ys, ak.to_numpy(self.data[i]['primary_hadron_1']['sensor_pos_y']).reshape(-1, 1)))
                zs = np.concatenate((zs, ak.to_numpy(self.data[i]['primary_hadron_1']['sensor_pos_z']).reshape(-1, 1)))
                ts = np.concatenate((ts, ak.to_numpy(self.data[i]['primary_hadron_1']['t']).reshape(-1, 1)))
        else:
            xs = ak.to_numpy(self.data[i]['primary_hadron_1']['sensor_pos_x']).reshape(-1, 1)
            ys = ak.to_numpy(self.data[i]['primary_hadron_1']['sensor_pos_y']).reshape(-1, 1)
            zs = ak.to_numpy(self.data[i]['primary_hadron_1']['sensor_pos_z']).reshape(-1, 1)
            ts = ak.to_numpy(self.data[i]['primary_hadron_1']['t']).reshape(-1, 1)

        # 4.566 ns for light to travel 1 m (NEED DOUBLE-CHECK)
        # pos_t = np.hstack((ak.to_numpy(self.data[i]['total']['sensor_pos_x']).reshape(-1, 1) * 4.566, 
        #                 ak.to_numpy(self.data[i]['total']['sensor_pos_y']).reshape(-1, 1) * 4.566, 
        #                 ak.to_numpy(self.data[i]['total']['sensor_pos_z']).reshape(-1, 1) * 4.566, 
        #                 ak.to_numpy(self.data[i]['total']['t']).reshape(-1, 1)))

        # xs = (xs + 570.9) * 4.566
        # ys = (ys + 521.08) * 4.566
        # zs = (zs + 2460.89) * 4.566

        xs *= 4.566
        ys *= 4.566
        zs *= 4.566

        pos_t = np.hstack((xs, ys, zs, ts))

        # only use first photon hit time per dom
        if self.first_hit:
            spos_t = pos_t[np.argsort(pos_t[:,-1])]
            _, indices, feats = np.unique(spos_t[:,:3], axis=0, return_index=True, return_counts=True)
            pos_t = spos_t[indices]
            pos_t = np.trunc(pos_t)
        else:
            pos_t, feats = np.unique(pos_t, return_counts=True, axis=0)

        feats = feats.reshape(-1, 1).astype(np.float64)

        if self.pred_cartesian_direction:
            x = np.cos(label[2]) * np.sin(label[1])
            y = np.sin(label[2]) * np.sin(label[1])
            z = np.cos(label[1])
            label = [label[0], x, y, z]
        
        return torch.from_numpy(pos_t), torch.from_numpy(feats).view(-1, 1), torch.from_numpy(np.array([label]))

# function to convert read files into inputs
# args: config dict, list of photon parquet files to use
# returns: awkward array of photon hit information, numpy array of true neutrino information
def ic_data_prep(data_file):
    total_time = time.time()

    # photons_data = ak.Array([])

    photons_data = ak.from_parquet(data_file, columns=["mc_truth", "primary_lepton_1", "primary_hadron_1"])

    # for photon_file in data_files:
    #     pd = ak.from_parquet(photon_file)
    #     photons_data = ak.concatenate((photons_data, pd), axis=0)

    # remove null events
    # keep_indices = []

    # for i in range(len(photons_data)):
    #     if ((len(photons_data[i]['primary_lepton_1']['t']) + len(photons_data[i]['primary_hadron_1']['t'])) > cfg['hits_min']) and ((len(photons_data[i]['primary_lepton_1']['t']) + len(photons_data[i]['primary_hadron_1']['t'])) < cfg['hits_max']):
    #         keep_indices.append(i)
            
    # photons_data = photons_data[keep_indices]
    print("total time:", time.time() - total_time)

    # converting read data to inputs
    es = np.array(photons_data['mc_truth']['injection_energy'])
    zenith = np.array(photons_data['mc_truth']['injection_zenith'])
    azimuth = np.array(photons_data['mc_truth']['injection_azimuth'])

    # energy transforms/normalization
    es_norm = (np.log(1 + es) - np.log(1 + es).mean()) / np.log(1 + es).std()

    nu_data = np.dstack((es, zenith, azimuth)).reshape(-1, 3)

    return photons_data, nu_data