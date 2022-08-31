import pickle
import numpy as np

from utils import Callable, is_floatable
from noise_box import uncorrelated_noise, correlated_noise

def add_noise(doms, specs):
    """
    """
    tmax = 0
    outfile = specs['hitsfile'].replace('.ppc', '_noisy.ppc')

    # Check if thermal rate has depth dependence or not
    if is_floatable(specs['thermal_rate']):
        depth_2_rate = Callable(float(specs['thermal_rate']))
    else: # load pickle
        with open(specs['thermal_rate'], 'rb') as pkl_f:
            depth_2_rate = pickle.load(pkl_f)

    with open(specs['hitsfile'], 'r') as fp:
        with open(outfile, 'w') as of:
            for line in fp:
                if 'EE' not in line:
                    of.write(line)
                    if 'HIT' in line:
                        tmax = max(tmax, float(line.split(' ')[3]))
                # The event is over so let's make some noise
                else:
                    of.write('CORRELATED\n')
                    for dom in doms:
                        nstr, nom, _ = dom
                        rate         = specs['radio_rate']
                        tt           = correlated_noise(rate, tmax, mu=specs['mu'], sigma=specs['sigma'], eta=specs['eta'], rand=specs['rand'])
                        tmax = max(tmax, max(tt))
                        for t in tt:
                            # Make a new line. All light is 400 nm cuz.....
                            l = f'HIT {nstr} {nom} {t} 400\n'
                            of.write(l)
                    of.write('THERMAL\n')
                    for dom in doms:
                        nstr, nom, z = dom
                        rate = depth_2_rate(z)
                        tt   = uncorrelated_noise(rate, tmax, rand=specs['rand'])
                        for t in tt:
                            # Make a new line. All light is 400 nm cuz.....
                            l = f'HIT {nstr} {nom} {t} 400\n'
                            of.write(l)
                    of.write(line)
    return outfile

if __name__=='__main__':
    import argparse
    import os, sys

    parser = argparse.ArgumentParser()

    parser.add_argument('-s',
                        type=int,
                        help='Seed for random number generator.'
                       )
    # Paths to places, etc.
    parser.add_argument('--hitsfile', 
                        type=str,
                        help='Path to hitsfile',
                        default=''
                       )
    parser.add_argument('--geofile', 
                        dest='geofile', 
                        default='../MCN/PPC/geo-f2k',
                        type=str,
                        help='Path to file with information about DOM geometry'
                       )
    parser.add_argument('--thermal_rate',
                        help='Path to file which has pickled spline for converting depth to rate or constant'
                       )

    # Parameter for correlated noise as described in https://inspirehep.net/files/147e9132d1d0245895dc407c4dd7505f
    parser.add_argument('--rate',
                        dest='rate',
                        type=float,
                        default=250,
                        help='Rate for radioactive events which give correlated noise.'
                       )
    parser.add_argument('--mu',
                        dest='mu',
                        type=float,
                        default=-6,
                        help='Mu parameter described in noise model paper'
                       )
    parser.add_argument('--sigma',
                        dest='sigma',
                        type=float,
                        default=2.7,
                        help='Sigma parameter described in noise model paper'
                       )
    parser.add_argument('--eta',
                        dest='eta',
                        type=float,
                        default=8.,
                        help='Eta parameter described in noise model paper'
                       )
    args = parser.parse_args()

    doms = []
    n = 134314
    with open(args.geofile, 'r') as gf:
        for line in gf:
            spl = line.replace('\n', '').split('\t')
            nstr = int(spl[5])
            nom  = int(spl[6])
            z    = float(spl[4])
            n = min(nom,n)
            if nom<=60: # Exclude icetop oms
                doms.append((nstr, nom, z))

    
    specs = {}
    specs['rand']         = np.random.RandomState(args.s)
    specs['hitsfile']     = args.hitsfile
    specs['thermal_rate'] = args.thermal_rate
    specs['radio_rate']   = args.rate
    specs['mu']           = args.mu
    specs['eta']          = args.eta
    specs['sigma']        = args.sigma
    
    add_noise(doms, specs)
