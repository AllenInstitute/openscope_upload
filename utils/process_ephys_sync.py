import numpy as np
import stimulus_sync_functions as stimulus_sync
import pickle_functions as pkl 
import functools

FRAME_KEYS = ('frames', 'stim_vsync', 'vsync_stim')
PHOTODIODE_KEYS = ('photodiode', 'stim_photodiode')
OPTOGENETIC_STIMULATION_KEYS = ("LED_sync", "opto_trial")
EYE_TRACKING_KEYS = ("eye_frame_received",  # Expected eye tracking
                                        # line label after 3/27/2020
                # clocks eye tracking frame pulses (port 0, line 9)
                "cam2_exposure",
                # previous line label for eye tracking
                # (prior to ~ Oct. 2018)
                "eyetracking",
                "eye_cam_exposing",
                "eye_tracking")  # An undocumented, but possible eye tracking line label  # NOQA E114
BEHAVIOR_TRACKING_KEYS = ("beh_frame_received",  # Expected behavior line label after 3/27/2020  # NOQA E127
                                            # clocks behavior tracking frame # NOQA E127
                                            # pulses (port 0, line 8)
                    "cam1_exposure",
                    "behavior_monitoring")



def extract_frame_times_from_photodiode(
    photodiode_cycle=60,
    frame_keys=FRAME_KEYS,
    photodiode_keys=PHOTODIODE_KEYS,
    trim_discontiguous_frame_times=True):

    photodiode_times = stimulus_sync.get_edges('all', photodiode_keys)
    vsync_times = stimulus_sync.get_edges('falling', frame_keys)

    if trim_discontiguous_frame_times:
        vsync_times = stimulus_sync.trim_discontiguous_vsyncs(vsync_times)

    vsync_times_chunked, pd_times_chunked = \
        stimulus_sync.separate_vsyncs_and_photodiode_times(
            vsync_times,
            photodiode_times,
            photodiode_cycle)

    frame_start_times = np.zeros((0,))

    for i in range(len(vsync_times_chunked)):

        photodiode_times = stimulus_sync.trim_border_pulses(
            pd_times_chunked[i],
            vsync_times_chunked[i])
        photodiode_times = stimulus_sync.correct_on_off_effects(
            photodiode_times)
        photodiode_times = stimulus_sync.fix_unexpected_edges(
            photodiode_times,
            cycle=photodiode_cycle)

        frame_duration = stimulus_sync.estimate_frame_duration(
            photodiode_times,
            cycle=photodiode_cycle)
        irregular_interval_policy = functools.partial(
            stimulus_sync.allocate_by_vsync,
            np.diff(vsync_times_chunked[i]))
        frame_indices, frame_starts, frame_end_times = \
            stimulus_sync.compute_frame_times(
                photodiode_times,
                frame_duration,
                len(vsync_times_chunked[i]),
                cycle=photodiode_cycle,
                irregular_interval_policy=irregular_interval_policy
                )

        frame_start_times = np.concatenate((frame_start_times,
                                            frame_starts))

    frame_start_times = stimulus_sync.remove_zero_frames(frame_start_times)

    return frame_start_times


def build_stimulus_table(
        stimulus_pkl_path,
        sync_h5_path,
        frame_time_strategy,
        minimum_spontaneous_activity_duration,
        extract_const_params_from_repr,
        drop_const_params,
        maximum_expected_spontanous_activity_duration,
        stimulus_name_map,
        column_name_map,
        output_stimulus_table_path,
        output_frame_times_path,
        fail_on_negative_duration,
        **kwargs
):
    stim_file = pkl.read_pkl(stimulus_pkl_path)
    frame_times = stimulus_sync.extract_frame_times(
        strategy=frame_time_strategy,
        trim_discontiguous_frame_times=kwargs.get(
            'trim_discontiguous_frame_times',
            True)
        )
    minimum_spontaneous_activity_duration = (
            minimum_spontaneous_activity_duration / pkl.get_fps(stim_file)
    )

    