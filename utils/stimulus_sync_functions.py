import numpy as np
import scipy.spatial.distance as distance
import pickle_functions as pkl
from typing import Union, Sequence, Optional


def seconds_to_frames(seconds, stim_file):
    pkl_file = pkl.read_pickle(stim_file)
    return (np.array(seconds) + pkl.get_pre_blank_sec(pkl_file)) * pkl.get_fps(pkl_file)

def get_edges(
    self,
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
    if kind == 'falling':
        fn = self.get_falling_edges
    elif kind == 'rising':
        fn = self.get_rising_edges
    elif kind == 'all':
        return np.sort(np.concatenate([
            self.get_edges('rising', keys, units),
            self.get_edges('falling', keys, units)
        ]))

    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        try:
            return fn(key, units)
        except ValueError:
            continue

    if not permissive:
        raise KeyError(
            f"none of {keys} were found in this dataset's line labels")


def get_falling_edges(self, line, units='samples'):
    """
    Returns the counter values for the falling edges for a specific bit
        or line.

    Parameters
    ----------
    line : str
        Line for which to return edges.

    """
    bit = self._line_to_bit(line)
    changes = self.get_bit_changes(bit)
    return self.get_all_times(units)[np.where(changes == 255)]


def get_rising_edges(self, line, units='samples'):
    """
    Returns the counter values for the rizing edges for a specific bit or
        line.

    Parameters
    ----------
    line : str
        Line for which to return edges.

    """
    bit = self._line_to_bit(line)
    changes = self.get_bit_changes(bit)
    return self.get_all_times(units)[np.where(changes == 1)]


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