import numpy as np
import noise_box
import awkward as ak


def construct_total_dict(event):

    # These are the keys which refer to the physical particles
    particle_fields = [
        f for f in event.fields
        if f not in "event_id mc_truth".split()
    ]
    # A set of the all the fields that we should expect
    fields = set(getattr(event, particle_fields[0]).fields)
    # Check to make sure all the particles have matching fields
    # TODO should we just make this a warning and not to the 
    # offending field
    for f in particle_fields:
        set_keys = set(getattr(event, f).fields)
        if not (
            fields.issubset(set_keys) and\
            set_keys.issubset(fields)
        ):
            raise ValueError("Particle keys are not compatible.")
    
    d = {}
    for field_k in fields:
        #nevents = len(fill_dict[particle_fields[0]][field_k])

        # Make an empty array that we will start stacking on
        total = np.array([])
        #total = np.array(
        #    [np.array([]) for _ in range(nevents)]
        #)
        # Iterate over all the particles, stacking on total each time
        for i, k in enumerate(particle_fields):
            # Don't need to do any special handling
            if i==0:
                current = getattr(getattr(event, k), field_k)
            # If this isn't the first one, we need to filter out [-1] entries
            # so that they don't crop up in the middle
            else:
                #current = [
                #    x if np.all(x!=-1) else [] for x in getattr(getattr(event, k), field_k)
                #]
                if np.all(getattr(getattr(event, k), field_k) == -1):
                    current = np.array([])
                else:
                    getattr(getattr(event, k), field_k)
            # Add the new stuff to the running total
            total = ak.concatenate(
                (total, current),
                #axis=1
            )
        # Throw it all in the dictionary :-)
        d[field_k] = total
    return d

def add_noise(
    event,
    detector_info,
    cor_rate = 250,
    uncor_rate = 20
):
    times = event.total.t
    # Patch for -1 bug
    times = times[times > 0]
    if len(times) > 0:
        delta_t = np.max(times) - np.min(times)
        noisy_event = event[[f for f in event.fields if f!="total"]]
        d = dict(
            sensor_pos_x = [],
            sensor_pos_y = [],
            sensor_pos_z = [],
            string_id = [],
            sensor_id = [],
            t = [],
        )
        ts = [
            np.append(
                noise_box.correlated_noise(cor_rate, delta_t),
                noise_box.uncorrelated_noise(uncor_rate, delta_t)
            ) for _ in detector_info
        ]
        for _, (info, t_) in enumerate(zip(detector_info, ts)):
            for tprime in t_:
                d["sensor_pos_x"].append(info[0])
                d["sensor_pos_y"].append(info[1])
                d["sensor_pos_z"].append(info[2])
                d["string_id"].append(info[3])
                d["sensor_id"].append(info[4])
                d["t"].append(tprime)
        noise = ak.Array(d)
        noisy_event = ak.with_field(noisy_event, noise, where="noise")
        total = construct_total_dict(noisy_event)
        noisy_event = ak.with_field(noisy_event, ak.Array(total), where="total")
    else:
        noisy_event = event
    return noisy_event
