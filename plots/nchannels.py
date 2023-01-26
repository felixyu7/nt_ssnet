import awkward as ak
import glob
import numpy as np

# from smt import passed_SMT

# data_files = sorted(glob.glob("/n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim_round3/data/" + ("*_smt.parquet")))
# data_files.extend(sorted(glob.glob("/n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim_round3/data/" + ("*_smt.parquet"))))
# data_files.extend(sorted(glob.glob("/n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim_round4/data/" + ("*_smt.parquet"))))

# photons_data_total = ak.Array([])

# for file in data_files:
#     data = ak.from_parquet(file)
#     photons_data_total = ak.concatenate((photons_data_total, data), axis=0)

# ak.to_parquet(photons_data_total, "/n/home10/felixyu/ic_ssnet/data/ic_ssnet_final_r3.parquet")

data_files = ["/n/home10/felixyu/ic_ssnet/data/ic_ssnet_final_total.parquet"]

hits = []
# es = []

for photon_file in data_files:
    photons_data = ak.from_parquet(photon_file)
    # keep_indices = []
 
    for i in range(len(photons_data)):

        xs = photons_data[i].filtered.sensor_pos_x.to_numpy()
        ys = photons_data[i].filtered.sensor_pos_y.to_numpy()
        zs = photons_data[i].filtered.sensor_pos_z.to_numpy()
        ts = photons_data[i].filtered.t.to_numpy()

        xs *= 4.566
        ys *= 4.566
        zs *= 4.566
        ts = ts - ts.min()

        pos_t = np.array([
            xs,
            ys,
            zs,
            ts
        ]).T

        # only use first photon hit time per dom
        spos_t = pos_t[np.argsort(pos_t[:,-1])]
        _, indices, feats = np.unique(spos_t[:,:3], axis=0, return_index=True, return_counts=True)
        pos_t = spos_t[indices]
        pos_t = np.trunc(pos_t)

        hits.append(pos_t.shape[0])
        # es.append(photons_data[i]['mc_truth']['injection_energy'])


    # keep_indices = []
    # # for i in range(len(photons_data)):
    #     # if len(photons_data[i]['primary_lepton_1']['t']) + len(photons_data[i]['primary_hadron_1']['t']) > 10:
    #     #     keep_indices.append(i)
    # for event in photons_data:
    #     keep_indices.append(passed_SMT(event))
    
    # photons_data = photons_data[keep_indices]
    # photons_data_total = ak.concatenate((photons_data_total, photons_data), axis=0)

hits = np.array(hits)
np.save("./data/hits.npy", hits)

# data1.total = data1["total", [x for x in ak.fields(data1.total) if x != "string_id"]]
# ak.to_parquet(photons_data_total, "/n/holyscratch01/arguelles_delgado_lab/Everyone/felixyu/ic_ssnet_sim_smt_round2.parquet")