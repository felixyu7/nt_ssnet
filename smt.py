import numpy as np

def SMT(
    str_id,
    sensor_id,
    t,
    multiplicity = 8,
    hlc_dt = 1000,
    time_window = 5e3
):
    str_sorter = np.argsort(str_id)
    # sort the hits by which string they on since only
    # OMs on the same string can produce a HLC
    str_id, sensor_id, t = (
        str_id[str_sorter], sensor_id[str_sorter], t[str_sorter]
    )
    # Split all the stuff by string
    str_id_split, splitter = consecutive(str_id, return_splitter=True)
    sensor_id_split = np.split(sensor_id, splitter)
    t_split = np.split(t, splitter)
    # Iterate over the hits on each string
    hlc_times = np.array([])
    for a, b in zip(
        sensor_id_split,
        t_split
    ):
        passed = False
        # Sort since HLCs only happen in one direction
        t_sorter = np.argsort(b)
        a, b = a[t_sorter], b[t_sorter]
        idx = 0
        for idx in range(len(a)):
            current_om, current_t = a[idx], b[idx]
            slc = slice(idx+1, None)
            # Check if the Oms are neighboring or next-to-neighboring
            is_neighbor = np.logical_and(
                a[slc]!=current_om,
                np.abs(a[slc]-current_om)<=2
            )
            is_time_coincident = (b[slc] - current_t) < hlc_dt
            is_hlc = np.logical_and(is_neighbor, is_time_coincident)
            hlc_times = np.append(hlc_times, b[slc][is_hlc])
            if (
                len(hlc_times) >= multiplicity and \
                np.max(hlc_times) - np.min(hlc_times) < time_window
            ):
                passed = True
                break
    # TODO do the complicated thing but I don't think it's necessary
    if return_trange:
        return passed, np.min(hlc_times), np.min(hlc_times) + time_window
    else:
        return passed

def consecutive(data, stepsize=0, return_splitter=False):
    splitter = np.where(np.diff(data) != stepsize)[0]+1
    if return_splitter:
        return np.split(data, splitter), splitter
    else:
        return np.split(data, splitter)

def passed_SMT(
    event,
    multiplicity = 8,
    hlc_dt = 1000,
    time_window = 5000
    return_trange = True
):

    if event.primary_lepton_1.t[0]==-1 and  event.primary_hadron_1.t[0]==-1:
        return False
    elif event.primary_lepton_1.t[0]==-1:
        str_id = event.primary_hadron_1.string_id
        sensor_id = event.primary_hadron_1.sensor_id
        t = event.primary_hadron_1.t
    elif event.primary_hadron_1.t[0]==-1:
        str_id = event.primary_lepton_1.string_id
        sensor_id = event.primary_lepton_1.sensor_id
        t = event.primary_lepton_1.t
    else:
        str_id = np.hstack(
            event.primary_lepton_1.string_id,
            event.primary_hadron_1.string_id
        )
        sensor_id = np.hstack(
            event.primary_lepton_1.sensor_id,
            event.primary_hadron_1.sensor_id
        )
        t = np.hstack(
            event.primary_lepton_1.t,
            event.primary_hadron_1.t
        )

     return SMT(
        str_id,
        sensor_id,
        t,
        multiplicity = multiplicity,
        hlc_dt = hlc_dt,
        time_window = time_window,
        return_trange = return_trange
    )

