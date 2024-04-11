import h5py

import numpy as np
import scipy.spatial.distance as distance
import pickle_functions as pkl

from typing import Union, Sequence, Optional


def load_sync(path):
    """
    Loads an hdf5 sync dataset.

    Parameters
    ----------
    path : str
        Path to hdf5 file.

    """
    dfile = h5py.File(
        path, 'r')
    return dfile


def get_meta_data(sync_file):
    """
    Returns the metadata for the sync file. 
    
    """ 
    meta_data = eval(sync_file['meta'][()])
    return meta_data


def get_line_labels(sync_file):
    """
    Returns the line labels for the sync file. 
    
    """ 
    meta_data = get_meta_data(sync_file)
    line_labels = meta_data['line_labels']
    return line_labels


def get_times(sync_file):
    """
    Returns the times for the sync file. 
    
    """ 
    times = process_times(sync_file)
    return times


def process_times(sync_file):
    """
    Preprocesses the time array to account for rollovers.
        This is only relevant for event-based sampling.

    """
    times = sync_file['data'][()][:, 0:1].astype(np.int64)

    intervals = np.ediff1d(times, to_begin=0)
    rollovers = np.where(intervals < 0)[0]

    for i in rollovers:
        times[i:] += 4294967296

    return times


def line_to_bit(sync_file, line):
    """
    Returns the bit for a specified line.  Either line name and number is
        accepted.

    Parameters
    ----------
    line : str
        Line name for which to return corresponding bit.

    """
    line_labels = get_line_labels(sync_file)

    if type(line) is int:
        return line
    elif type(line) is str:
        return line_labels.index(line)
    else:
        raise TypeError("Incorrect line type.  Try a str or int.")


def get_edges(
    sync_file: h5py.File,
    kind: str,
    keys: Union[str, Sequence[str]],
    units: str = "seconds",
    permissive: bool = False
) -> Optional[np.ndarray]:
    """ Utility function for extracting edge times from a line

    Parameters
    ----------
    kind : One of "rising", "falling", or "all". Should this method return
        timestamps for rising, falling or both edges on the appropriate
        line
    keys : These will be checked in sequence. Timestamps will be returned
        for the first which is present in the line labels
    units : one of "seconds", "samples", or "indices". The returned
        "time"stamps will be given in these units.
    raise_missing : If True and no matching line is found, a KeyError will
        be raised

    Returns
    -------
    An array of edge times. If raise_missing is False and none of the keys
        were found, returns None.

    Raises
    ------
    KeyError : none of the provided keys were found among this dataset's
        line labels

    """

    if isinstance(keys, str):
        keys = [keys]

    for line in keys:
        if kind == 'falling':
            fn = get_falling_edges(sync_file, line, units)
        elif kind == 'rising':
            fn = get_rising_edges(sync_file, line, units)
        elif kind == 'all':
            return np.sort(np.concatenate([
                get_edges('rising', keys, units),
                get_edges('falling', keys, units)
            ]))
        
        try:
            return fn(sync_file, line, units)
        except ValueError:
            continue

    if not permissive:
        raise KeyError(
            f"none of {keys} were found in this dataset's line labels")


def get_bit_changes(bit):
    """
    Returns the first derivative of a specific bit.
        Data points are 1 on rising edges and 255 on falling edges.

    Parameters
    ----------
    bit : int
        Bit for which to return changes.

    """
    bit_array = get_bit(bit)
    return np.ediff1d(bit_array, to_begin=0)


def get_bit(uint_array, bit):
    """
    Returns a bool array for a specific bit in a uint ndarray.

    Parameters
    ----------
    uint_array : (numpy.ndarray)
        The array to extract bits from.
    bit : (int)
        The bit to extract.

    """
    return np.bitwise_and(uint_array, 2 ** bit).astype(bool).astype(np.uint8)


def get_sample_freq(meta_data):
    try:
        return float(meta_data['ni_daq']['sample_freq'])
    except KeyError:
        return float(meta_data['ni_daq']['counter_output_freq'])


def get_all_times(sync_file, meta_data, units='samples'):
    """
    Returns all counter values.

    Parameters
    ----------
    units : str
        Return times in 'samples' or 'seconds'

    """
    if meta_data['ni_daq']['counter_bits'] == 32:
        times = sync_file['data'][()][:, 0]
    else:
        times = times
    units = units.lower()
    if units == 'samples':
        return times
    elif units in ['seconds', 'sec', 'secs']:
        freq = get_sample_freq(meta_data)
        return times / freq
    else:
        raise ValueError("Only 'samples' or 'seconds' are valid units.")


def get_falling_edges(sync_file, line, units='samples'):
    """
    Returns the counter values for the falling edges for a specific bit
        or line.

    Parameters
    ----------
    line : str
        Line for which to return edges.

    """
    meta_data  = get_meta_data(sync_file)
    bit = line_to_bit(sync_file, line)
    changes = get_bit_changes(bit)
    return get_all_times(sync_file, meta_data, units)[np.where(changes == 255)]


def get_rising_edges(sync_file, line, units='samples'):
    """
    Returns the counter values for the rizing edges for a specific bit or
        line.

    Parameters
    ----------
    line : str
        Line for which to return edges.

    """
    meta_data  = get_meta_data(sync_file)
    bit = line_to_bit(sync_file, line)
    changes = get_bit_changes(bit)
    return get_all_times(sync_file, meta_data, units)[np.where(changes == 1)]


def trimmed_stats(data, pctiles=(10, 90)):
    low = np.percentile(data, pctiles[0])
    high = np.percentile(data, pctiles[1])

    trimmed = data[np.logical_and(
        data <= high,
        data >= low
    )]

    return np.mean(trimmed), np.std(trimmed)


def trim_discontiguous_vsyncs(vs_times, photodiode_cycle=60):
    vs_times = np.array(vs_times)

    breaks = np.where(np.diff(vs_times) > (1/photodiode_cycle)*100)[0]

    if len(breaks) > 0:
        chunk_sizes = np.diff(np.concatenate((np.array([0, ]),
                                                breaks,
                                                np.array([len(vs_times), ]))))
        largest_chunk = np.argmax(chunk_sizes)

        if largest_chunk == 0:
            return vs_times[:np.min(breaks+1)]
        elif largest_chunk == len(breaks):
            return vs_times[np.max(breaks+1):]
        else:
            return vs_times[breaks[largest_chunk-1]:breaks[largest_chunk]]
    else:
        return vs_times


def assign_to_last(starts, ends, frame_duration, irregularity):
    ends[-1] += frame_duration * np.sign(irregularity)
    return starts, ends


def remove_zero_frames(frame_times):
    deltas = np.diff(frame_times)

    small_deltas = np.where(deltas < 0.01)[0]
    big_deltas = np.where((deltas > 0.018) * (deltas < 0.1))[0]

    def find_match(big_deltas, value):
        try:
            return big_deltas[np.max(np.where((big_deltas < value))[0])] - value
        except ValueError:
            return None

    paired_deltas = [find_match(big_deltas, A) for A in small_deltas]

    ft = np.copy(deltas)

    for idx, d in enumerate(small_deltas):
        if paired_deltas[idx] is not None:
            if paired_deltas[idx] > -100:
                ft[d+paired_deltas[idx]] = np.median(deltas)
                ft[d] = np.median(deltas)

    t = np.concatenate(([np.min(frame_times)],
                        np.cumsum(ft) + np.min(frame_times)))

    return t



def compute_frame_times(photodiode_times,
                        frame_duration,
                        num_frames,
                        cycle,
                        irregular_interval_policy=assign_to_last):

    indices = np.arange(num_frames)
    starts = np.zeros(num_frames, dtype=float)
    ends = np.zeros(num_frames, dtype=float)

    num_intervals = len(photodiode_times) - 1
    for start_index, (start_time, end_time) in \
            enumerate(zip(photodiode_times[:-1], photodiode_times[1:])):

        interval_duration = end_time - start_time
        irregularity = \
            int(np.around((interval_duration) / frame_duration)) - cycle

        local_frame_duration = interval_duration / (cycle + irregularity)
        durations = \
            np.zeros(cycle +
                     (start_index == num_intervals - 1)) + local_frame_duration

        current_ends = np.cumsum(durations) + start_time
        current_starts = current_ends - durations

        while irregularity != 0:
            current_starts, current_ends = irregular_interval_policy(
                start_index,
                current_starts,
                current_ends,
                local_frame_duration,
                irregularity, cycle
            )
            irregularity += -1 * np.sign(irregularity)

        early_frame = start_index * cycle
        late_frame = \
            (start_index + 1) * cycle + (start_index == num_intervals - 1)

        remaining = starts[early_frame: late_frame].size
        starts[early_frame: late_frame] = current_starts[:remaining]
        ends[early_frame: late_frame] = current_ends[:remaining]

    return indices, starts, ends


def separate_vsyncs_and_photodiode_times(vs_times,
                                         pd_times,
                                         photodiode_cycle=60):

    vs_times = np.array(vs_times)
    pd_times = np.array(pd_times)

    breaks = np.where(np.diff(vs_times) > (1/photodiode_cycle)*100)[0]

    shift = 2.0
    break_times = [-shift]
    break_times.extend(vs_times[breaks].tolist())
    break_times.extend([np.inf])

    vs_times_out = []
    pd_times_out = []

    for indx, b in enumerate(break_times[:-1]):

        pd_in_range = np.where((pd_times > break_times[indx] + shift) *
                               (pd_times <= break_times[indx+1] + shift))[0]
        vs_in_range = np.where((vs_times > break_times[indx]) *
                               (vs_times <= break_times[indx+1]))[0]

        vs_times_out.append(vs_times[vs_in_range])
        pd_times_out.append(pd_times[pd_in_range])

    return vs_times_out, pd_times_out


def flag_unexpected_edges(pd_times, ndevs=10):
    pd_diff = np.diff(pd_times)
    diff_mean, diff_std = trimmed_stats(pd_diff)

    expected_duration_mask = np.ones(pd_diff.size)
    expected_duration_mask[np.logical_or(
        pd_diff < diff_mean - ndevs * diff_std,
        pd_diff > diff_mean + ndevs * diff_std
    )] = 0
    expected_duration_mask[1:] = np.logical_and(expected_duration_mask[:-1],
                                                expected_duration_mask[1:])
    expected_duration_mask = np.concatenate([expected_duration_mask,
                                            [expected_duration_mask[-1]]])

    return expected_duration_mask


def fix_unexpected_edges(pd_times, ndevs=10, cycle=60, max_frame_offset=4):
    pd_times = np.array(pd_times)
    expected_duration_mask = flag_unexpected_edges(pd_times, ndevs=ndevs)
    diff_mean, diff_std = trimmed_stats(np.diff(pd_times))
    frame_interval = diff_mean / cycle

    bad_edges = np.where(expected_duration_mask == 0)[0]
    bad_blocks = np.sort(np.unique(np.concatenate([
        [0],
        np.where(np.diff(bad_edges) > 1)[0] + 1,
        [len(bad_edges)]
    ])))

    output_edges = []
    for low, high in zip(bad_blocks[:-1], bad_blocks[1:]):
        current_bad_edge_indices = bad_edges[low: high-1]
        current_bad_edges = pd_times[current_bad_edge_indices]
        low_bound = pd_times[current_bad_edge_indices[0]]
        high_bound = pd_times[current_bad_edge_indices[-1] + 1]

        edges_missing = int(np.around((high_bound - low_bound) / diff_mean))
        expected = np.linspace(low_bound, high_bound, edges_missing + 1)

        distances = distance.cdist(current_bad_edges[:, None],
                                   expected[:, None])
        distances = np.around(distances / frame_interval).astype(int)

        min_offsets = np.amin(distances, axis=0)
        min_offset_indices = np.argmin(distances, axis=0)
        output_edges = np.concatenate([
            output_edges,
            expected[min_offsets > max_frame_offset],
            current_bad_edges[min_offset_indices[min_offsets <=
                              max_frame_offset]]
        ])

    return np.sort(np.concatenate([output_edges,
                                   pd_times[expected_duration_mask > 0]]))