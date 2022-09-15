import numpy as np

def uncorrelated_noise(rate, delta_t, rand=None):
    if rand is None:
        rand = np.random.RandomState()
    rate /= 1e9 # Convert to gigahertz
    n_events = np.random.poisson(rate*delta_t)
    times = np.random.uniform(0, delta_t, n_events)
    return times

def correlated_noise(rate, delta_t, mu=-6, sigma=2.7, eta=8., rand=None):
    '''
    calculating correlated noise from radioactive decay following:\n
    https://inspirehep.net/files/147e9132d1d0245895dc407c4dd7505f

    params:
    ______
    rate (float) expected rate for radioactive decay [Hz]
    delta_t (float) Total duration of event [ns]
    mu (float) Parameter describing the mean delay of photons after radioactive decay
    sigma (float) Parameter describing the time spread of photons after radioactive decay
    eta (float) Poisson expectation of number of photons per radioactive decay

    returns
    _______
    times (array) Array of times at which correlated photons arrived
    '''

    if rand is None:
        rand = np.random.RandomState()
    mu += 9 # nano-ify
    # Times when radioactive decays happened
    decay_ts = uncorrelated_noise(rate, delta_t, rand=rand)
    photon_ts = np.array([], dtype=float)
    for time in decay_ts:
        n_photons = rand.poisson(eta)
        zz = rand.normal(size=n_photons)
        dtt = np.power(10., mu + sigma * zz)
        photon_ts = np.append(photon_ts, time+np.cumsum(dtt))
    return photon_ts
