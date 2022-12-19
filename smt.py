import numpy as np

from collections.abc import Iterable

class TimesNotOrderedError(Exception):
    """Rasied if hit times are not in ascending order"""
    def __init__(self):
        self.message = "Input times are not in ascending order"
        super().__init__(self.message)

class IncompatibleLengthsError(Exception):

    def __init__(self, id_type, nts, nids):
        self.message = f"Number of {id_type} incompatible with number of times."
        self.message += f"Expected {nts} but got {nids}"
        super().__init__(self.message)

def has_HLC(
    string_ids: Iterable[int],
    sensor_ids: Iterable[int],
    times: Iterable[float],
    hlc_dt: float = 5000.0
):
    has_hlc = False
    rstring_ids = string_ids[::-1]
    rsensor_ids = sensor_ids[::-1]
    rtimes = times[::-1]
    for idx, (time, sensor_id, string_id) in enumerate(zip(rtimes, rsensor_ids, rstring_ids)):
        slc = slice(idx+1, None, None)
        is_neighbor = np.logical_and(
            0 != np.abs(sensor_ids[slc] - sensor_id),
            np.abs(sensor_ids[slc] - sensor_id) <= 2
        )
        is_samestring = string_ids[slc] == string_id
        can_hlc = np.logical_and(is_neighbor, is_samestring)
        did_hlc = np.abs(times[slc][can_hlc] - time) <= hlc_dt
        hlc_times = times[slc][can_hlc][did_hlc]
        if len(hlc_times) > 0:
            has_hlc = True
            break
    if has_hlc:
        return has_hlc, time
    else:
        return has_hlc, np.max(times)


def SMT(
    string_ids: Iterable[int],
    sensor_ids: Iterable[int],
    times: Iterable[float],
    multiplicity:int = 8,
    hlc_dt:float = 5000.0,
) -> bool:
    """
    Function to determine if event passed SMT-N
    """
    # Make sure lengths are compatible
    ntimes = len(times)
    if len(sensor_ids) != ntimes:
        raise IncompatibleLengthsError("sensor_ids", ntimes, len(sensor_ids))
    if len(string_ids) != ntimes:
        raise IncompatibleLengthsError("string_ids", ntimes, len(string_ids))

    # Make sure times are ordered
    if np.any(np.diff(times) < 0):
        raise TimesNotOrderedError

    times = np.array(times)
    sensor_ids = np.array(sensor_ids)
    string_ids = np.array(string_ids)
    n_hlc = 0
    hlc_times = np.array([])
    did_smt = False
    for idx, (time, sensor_id, string_id) in enumerate(zip(times, sensor_ids, string_ids)):
        slc = slice(idx+1, None, None)
        is_neighbor = np.logical_and(
            0 != np.abs(sensor_ids[slc] - sensor_id),
            np.abs(sensor_ids[slc] - sensor_id) <= 2
        )
        is_samestring = string_ids[slc] == string_id
        can_hlc = np.logical_and(is_neighbor, is_samestring)
        did_hlc = times[slc][can_hlc] - time <= hlc_dt
        n_hlc += np.count_nonzero(did_hlc)
        hlc_times = np.append(hlc_times, times[slc][can_hlc][did_hlc])
        if n_hlc >= multiplicity:
            did_smt = True
            break
    if not did_smt:
        return False, None, None
    # Find when the trigger should stop recording
    has_hlc = True
    min_time = np.max(hlc_times)
    while has_hlc:
        max_time = min_time + hlc_dt
        mask = np.logical_and(
            min_time < times,
            times < max_time
        )
        if len(string_ids[mask])==0:
            has_hlc = False
            min_time = min_time + hlc_dt
        else:
            has_hlc, min_time = has_HLC(
            string_ids[mask],
            sensor_ids[mask],
            times[mask],
            hlc_dt=hlc_dt
        )
    return did_smt, np.min(hlc_times), max(np.min(hlc_times)+hlc_dt, min_time)
    

def passed_SMT(
    event: ak.Record,
    multiplicity: int = 8,
    hlc_dt: float = 5000.0,
    field: str = "total"
) -> (bool, float, float):

    # sorry bout it
    whatever = getattr(event, field)
    if "string_id" in whatever.fields:
        string_ids = np.array(
            [x for x in whatever.string_id if x != -1]
        )
    else:
        string_ids = np.array(
            [x for x in whatever.sensor_string_id if x != -1]
        )
    times = np.array([x for x in whatever.t if x != -1])
    sensor_ids = np.array([x for x in whatever.sensor_id if x != -1])

    sorter = np.argsort(times)
    return SMT(
        string_ids[sorter],
        sensor_ids[sorter],
        times[sorter],
        multiplicity = multiplicity,
        hlc_dt = hlc_dt,
    )
