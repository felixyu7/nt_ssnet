import numpy as np

from collections.abc import Iterable

class TimesNotOrderedError(Exception):

    def __init__(self):
        self.message = "Input times are not in ascending order"
        super().__init__(self.message)

class IncompatibleLengthsError(Exception):

    def __init__(self, id_type, nts, nids):
        self.message = f"Number of {id_type} incompatible with number of times."
        self.message += f"Expected {nts} but got {nids}"
        super().__init__(self.message)

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
    for idx, (time, sensor_id, string_id) in enumerate(zip(times, sensor_ids, string_ids)):
        slc = slice(idx+1, None, None)
        is_neighbor = np.logical_and(
            0 < np.abs(sensor_ids[slc] - sensor_id),
            np.abs(sensor_ids[slc] - sensor_id) <= 2
        )
        is_samestring = string_ids[slc] == string_id
        can_hlc = np.logical_and(is_neighbor, is_samestring)
        n_hlc += np.count_nonzero(times[slc][can_hlc] - time)
        if n_hlc >= multiplicity:
            break
    return n_hlc >= multiplicity

def passed_SMT(
    event,
    multiplicity: int = 8,
    hlc_dt:float = 5000.0,
):
    if "string_id" in event.total.fields:
        string_ids = np.array(
            [x for x in event.total.string_id if x != -1]
        )
    else:
        string_ids = np.array(
            [x for x in event.total.sensor_string_id if x != -1]
        )
    times = np.array([x for x in event.total.t if x != -1])
    sensor_ids = np.array([x for x in event.total.sensor_id if x != -1])

    sorter = np.argsort(times)
    return SMT(
        string_ids[sorter],
        sensor_ids[sorter],
        times[sorter],
        multiplicity = multiplicity,
        hlc_dt = hlc_dt,
    )
