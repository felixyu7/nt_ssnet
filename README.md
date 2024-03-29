# nt_ssnet: A Sparse Submanifold CNN for Neutrino Telescopes

This repo contains the scripts and code for the Sparse Submanifold CNN (SSCNN) project for neutrino telescope data. The code in this repo are setup to train on and predict the direction and/or energy of the neutrino. The working environment used to implement, run and test this code are as follows:

- Python 3.7.5
- PyTorch 1.10.1
- CUDA 11.1 (if using gpu)
- MinkowskiEngine 0.5.4
- Numpy 1.21.4
- Awkward 1.8.0
- yaml 5.4.1

Training and inference runs can be configured with the files (train.cfg, inference.cfg), and running ```python train.py```, ```python infer.py```.

A brief description of the configuration file options is listed below:

- device: 'cuda' or 'cpu', this option determines which hardware to run the network on (cuda for gpu, cpu for cpu)
- model: 'angular_reco', 'energy_reco', 'both', this option determines what the final output will be (just the angle, just the energy, or both)
- reps: int, the number of blocks each level of ResNet will have
- depth: int, the number of depth levels ResNet will have
- num_filters: int, the number of filters at the first level of ResNet
- stride: int, the downsampling factor at each level of ResNet
- train_data_file: str, the path to train data files, in .parquet format, as output from Prometheus simulation software
- valid_data_file: str, the path to validation data files, in .parquet format, as output from Prometheus simulation software
- batch_size: int, batch size used during training/inference
- epochs: int, number of epochs to train
- lr: float, initial learning rate
- model_weights: str, path to a checkpoint file to load in previously trained weights
- first_hit: bool, if true, preprocess data such that only the exact timing of the first hit on each optical module will be used. If false, will use all data points
- ckpt_dir: str, path to save checkpoint files to
- logs_file: str, path to save log files to

## Beansi: Noise module

Beansi (Ban-shee) applies correlated and uncorrelated noise to events. The process for this is outlined in https://inspirehep.net/files/147e9132d1d0245895dc407c4dd7505f. At this point I am setting the uncorrelated noise rate to 30 Hz. In reality it varies between 20 and 40 Hz depending on the depth of the DOM, but the correlated noise rate is ~250 Hz so it is the dominant source. I don't think that doing a uniform 30 Hz should cause issues

Minimal working example:
```
[In]: import awkward as ak
 ...:  a = ak.from_parquet("/n/home12/jlazar/hebe/examples/output/MuMinus_Hadrons_seed_10121995_meta_data.parquet")
 ...: event = a[6]
 ...: event.fields
 
[Out]: ['event_id', 'mc_truth', 'primary_lepton_1', 'primary_hadron_1', 'total']

[In]: from beansi import add_noise
 ...: infos = []
 ...: with open(""./beansi/icecube-geo"") as geofile:
 ...:     for i, line in enumerate(geofile.readlines()):
 ...:         if i >= 4:
 ...:             line = line.replace("\n", "").split("\t")
 ...:             line = tuple([float(x) for x in line])
 ...:             infos.append(line)
 ...: noisey_event = add_noise(event, infos)
 ...: noisey_event.fields
 
[Out]: ['event_id', 'mc_truth', 'primary_lepton_1', 'primary_hadron_1', 'noise', 'total']

[In]: len(event.total.t), len(noisey_event.total.t)

[Out]: (362, 398)
```

## Simple Multiplicity Trigger

(I hope) I follow the simple multiplicty trigger procedure outlined in https://arxiv.org/abs/1612.05093. I look for a given number of hard local coincidences (HLCs) in a certain time window. By default it is 8 HLCs in 5 microseconds. Once this condition is met, I continue to record all HLCs until no more are found within the defined time window. The trigger time window goes from the time of the first HLC minus 4 microseconds to the time of the last HLC plus 6 microseconds. I think this is what the paper defines, but who knows.

```
[In]: import awkward as ak
 ...: a = ak.from_parquet("/n/home12/jlazar/hebe/examples/output/MuMinus_Hadrons_seed_10121995_meta_data.parquet")
 ...: event = a[6]
 
[In]: from smt import passed_SMT
 ...: passed_SMT(event)
 
[Out]: True
 
[In]: passed_SMT(event, multiplicity=1e4)
 
[Out]: False
 
[In]: passed_SMT(event, return_trange=True)

[Out]: (True, 21610.240234, 36565.841797)

[In]: passed_SMT(event, return_trange=True, multiplicity=1e4)

[Out]: False, None, None
```


