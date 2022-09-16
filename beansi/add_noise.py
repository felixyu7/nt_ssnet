import numpy as np
import awkward as ak

from .noise_box import uncorrelated_noise, correlated_noise

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
    for field in fields:
        total = np.array([])
        for i, k in enumerate(particle_fields):
            # Don't need to do any special handling
            if np.all(getattr(getattr(event, k), field) == -1):
                current = np.array([])
            else:
                current = getattr(getattr(event, k), field)
            # Add the new stuff to the running total
            total = np.hstack(
                (total, current),
            )
            if len(total)==0:
                total = np.array([-1])
        # Throw it all in the dictionary :-)
        d[field] = total
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
        fields = event.total.fields
        delta_t = np.max(times) - np.min(times)
        noisy_event = event[[f for f in event.fields if f!="total"]]
        noise = {f:[] for f in fields}
        # TODO This is so jank, we need to fix this properly
        if "string_id" in noise.keys():
            string_id_key = "string_id"
        else:
            string_id_key= "sensor_string_id"
        ts = [
            np.append(
                correlated_noise(cor_rate, delta_t),
                uncorrelated_noise(uncor_rate, delta_t)
            ) for _ in detector_info
        ]
        for _, (info, t_) in enumerate(zip(detector_info, ts)):
            for tprime in t_:
                noise["sensor_pos_x"].append(info[0])
                noise["sensor_pos_y"].append(info[1])
                noise["sensor_pos_z"].append(info[2])
                noise[string_id_key].append(info[3])
                noise["sensor_id"].append(info[4])
                noise["t"].append(tprime + np.min(times))
        noisy_event = ak.with_field(noisy_event, noise, where="noise")
        total = construct_total_dict(noisy_event)
        noisy_event = ak.with_field(noisy_event, total, where="total")
    else:
        noisy_event = event
    return noisy_event
