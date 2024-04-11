import numpy as np
import scipy.spatial.distance as distance


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