import numpy as np

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