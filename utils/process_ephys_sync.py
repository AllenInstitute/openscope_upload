import functools

import numpy as np
import sync_functions as sync
import pickle_functions as pkl 
import stimulus_functions as stim
import naming_functions as names


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
    sync_file = sync.read_sync(sync_h5_path)

    frame_times = stim.extract_frame_times_from_photodiode(
        sync_file
        )
    minimum_spontaneous_activity_duration = (
            minimum_spontaneous_activity_duration / pkl.get_fps(stim_file)
    )

    stimulus_tabler = functools.partial(
        stim.build_stimuluswise_table,
        seconds_to_frames=stim.seconds_to_frames,
        extract_const_params_from_repr=extract_const_params_from_repr,
        drop_const_params=drop_const_params,
    )

    spon_tabler = functools.partial(
        stim.make_spontaneous_activity_tables,
        duration_threshold=minimum_spontaneous_activity_duration,
    )

    stim_table_sweeps = stim.create_stim_table(
        stim_file, stim_file.stimuli, stimulus_tabler, spon_tabler
    )

    stim_table_seconds= stim.convert_frames_to_seconds(
        stim_table_sweeps, frame_times, stim_file.frames_per_second, True
    )

    stim_table_seconds = names.collapse_columns(stim_table_seconds)
    stim_table_seconds = names.drop_empty_columns(stim_table_seconds)
    stim_table_seconds = names.standardize_movie_numbers(
        stim_table_seconds)
    stim_table_seconds = names.add_number_to_shuffled_movie(
        stim_table_seconds)
    stim_table_seconds = names.map_stimulus_names(
        stim_table_seconds, stimulus_name_map
    )

    stim_table_final = names.map_column_names(stim_table_seconds,
                                                        column_name_map,
                                                        ignore_case=False)

    stim_table_final.to_csv(output_stimulus_table_path, index=False)