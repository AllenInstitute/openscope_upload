import functools

import numpy as np
import pandas as pd
import utils.sync_functions as sync
import utils.pickle_functions as pkl 
import utils.stimulus_functions as stim
import utils.naming_functions as names


def build_optogenetics_table(
        opto_pkl_path,
        sync_h5_path,
        condition_map,
        output_opto_table_path,
        keys=stim.OPTOGENETIC_STIMULATION_KEYS
    ): 
    opto_file = pkl.load_pkl(opto_pkl_path)
    sync_file = sync.load_sync(sync_h5_path)

    start_times = sync.extract_led_times(sync_file,
                                         keys
                                        )

    conditions = [str(item) for item in opto_file['opto_conditions']]
    levels = opto_file['opto_levels']
    assert len(conditions) == len(levels)
    if len(start_times) > len(conditions):
        raise ValueError(
            f"there are {len(start_times) - len(conditions)} extra "
            f"optotagging sync times!")
    optotagging_table = pd.DataFrame({
        'start_time': start_times,
        'condition': conditions,
        'level': levels
    })
    optotagging_table = optotagging_table.sort_values(by='start_time', axis=0)

    stop_times = []
    names = []
    conditions = []
    for ii, row in optotagging_table.iterrows():
        condition = condition_map[row["condition"]]
        stop_times.append(row["start_time"] + condition["duration"])
        names.append(condition["name"])
        conditions.append(condition["condition"])

    optotagging_table["stop_time"] = stop_times
    optotagging_table["stimulus_name"] = names
    optotagging_table["condition"] = conditions
    optotagging_table["duration"] = \
        optotagging_table["stop_time"] - optotagging_table["start_time"]

    optotagging_table.to_csv(output_opto_table_path, index=False)
    return {'output_opto_table_path': output_opto_table_path}


def build_stimulus_table(
        stimulus_pkl_path,
        sync_h5_path,
        output_stimulus_table_path,
        minimum_spontaneous_activity_duration=0.0,
        extract_const_params_from_repr=False,
        drop_const_params=stim.DROP_PARAMS,
        stimulus_name_map=names.default_stimulus_renames,
        column_name_map=names.default_column_renames,
):
    stim_file = pkl.load_pkl(stimulus_pkl_path)
    sync_file = sync.load_sync(sync_h5_path)

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
        stim_file, pkl.get_stimuli(stim_file), stimulus_tabler, spon_tabler
    )

    stim_table_seconds= stim.convert_frames_to_seconds(
        stim_table_sweeps, frame_times, pkl.get_fps(stim_file), True
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
    return output_stimulus_table_path


if __name__ == "__main__":
    stimulus_pkl_path = r"/allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/1305804722/1298465622_692072_20230921.stim.pkl"
    sync_h5_path = r"/allen/programs/mindscope/production/openscope/prod0/specimen_1292228830/ecephys_session_1298465622/1304435871/1298465622_692072_20230921.sync"
    minimum_spontaneous_activity_duration = 0.0
    extract_const_params_from_repr = False
    drop_const_params = stim.DROP_PARAMS
    stimulus_name_map = names.default_stimulus_renames
    column_name_map = names.default_column_renames
    opto_path = r'/allen/programs/mindscope/production/openscope/prod0/specimen_1172968426/ecephys_session_1182865981/1194443659/1182865981_625545_20220608.opto.pkl'
    opto_sync = r'/allen/programs/mindscope/production/openscope/prod0/specimen_1172968426/ecephys_session_1182865981/1194443659/1182865981_625545_20220608.sync'
    opto_name_map = OPTOGENETIC_STIMULATION_KEYS = ("LED_sync", "opto_trial")
    output_stimulus_table_path = r"/allen/programs/mindscope/workgroups/openscope/ahad/output.csv"
    output_opto_table_path = r"/allen/programs/mindscope/workgroups/openscope/ahad/opto_output.csv"

    conditions = {
        "0": {
            "duration": .01,
            "name": "1Hz_10ms",
            "condition": "10 ms pulse at 1 Hz"
        },
        "1": {
            "duration": .002,
            "name": "1Hz_2ms",
            "condition": "2 ms pulse at 1 Hz"
        },
        "2": {
            "duration": 1.0,
            "name": "5Hz_2ms",
            "condition": "2 ms pulses at 5 Hz"
        },
        "3": {
            "duration": 1.0,
            "name": "10Hz_2ms",
            "condition": "2 ms pulses at 10 Hz'"
        },
        "4": {
            "duration": 1.0,
            "name": "20Hz_2ms",
            "condition": "2 ms pulses at 20 Hz"
        },
        "5": {
            "duration": 1.0,
            "name": "30Hz_2ms",
            "condition": "2 ms pulses at 30 Hz"
        },
        "6": {
            "duration": 1.0,
            "name": "40Hz_2ms",
            "condition": "2 ms pulses at 40 Hz"
        },
        "7": {
            "duration": 1.0,
            "name": "50Hz_2ms",
            "condition": "2 ms pulses at 50 Hz"
        },
        "8": {
            "duration": 1.0,
            "name": "60Hz_2ms",
            "condition": "2 ms pulses at 60 Hz"
        },
        "9": {
            "duration": 1.0,
            "name": "80Hz_2ms",
            "condition": "2 ms pulses at 80 Hz"
        },
        "10": {
            "duration": 1.0,
            "name": "square_1s",
            "condition": "1 second square pulse: continuously on for 1s"
        },
        "11": {
            "duration": 1.0,
            "name": "cosine_1s",
            "condition": "cosine pulse"
        },
    }
    '''
    build_stimulus_table(
        stimulus_pkl_path,
        sync_h5_path,
        minimum_spontaneous_activity_duration,
        extract_const_params_from_repr,
        drop_const_params,
        stimulus_name_map,
        column_name_map,
        output_stimulus_table_path,
    )
    '''
    build_optogenetics_table(
        opto_path,
        opto_sync,
        opto_name_map,
        conditions,
        output_opto_table_path
    )
