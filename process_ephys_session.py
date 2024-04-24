import aind_data_schema
import aind_data_schema.core.session as session_schema
import argparse
import datetime
import io
import json
import npc_ephys
import npc_mvr
import np_session
import npc_session
import npc_sessions
import npc_sync
import numpy as np
import pandas as pd
import pathlib as pl
from aind_data_schema.models.modalities import Modality as schema_modalities
from aind_data_schema.models.coordinates import Coordinates3d as schema_coordinates
from utils import process_ephys_sync as stim_utils
from utils import pickle_functions as pkl_utils

# defaults
DEFAULT_OPTO_CONDITIONS = {
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


def get_available_probes(platform_json):
    # available_probes = {'A','B','C','D','E','F'}
    # for recording_dir in recording_dirs:
    #     invalid_probes = set(probe_letter for probe_letter in available_probes if not valid_probe_folder(probe_letter, recording_dir))
    #     print('invalid probes:',invalid_probes)
    #     available_probes  -= invalid_probes
    insertion_notes = platform_json['InsertionNotes']
    available_probes = [letter for letter in 'ABCDEF' if insertion_notes == {} or not insertion_notes[f'Probe{letter}']["FailedToInsert"]]
    print('available probes:',available_probes)
    return tuple(available_probes)


def manipulator_coords(probe_name, newscale_coords):
    probe_row = newscale_coords.query(f"electrode_group == '{probe_name}'")
    return schema_coordinates(
        x=probe_row['x'].item(),
        y=probe_row['y'].item(),
        z=probe_row['z'].item(),
        unit='micrometer',
    ) 


def ephys_modules(available_probes, motor_locs_path):
    newscale_coords = npc_sessions.get_newscale_coordinates(motor_locs_path)
    
    ephys_modules = []
    for probe_letter in available_probes:
        probe_name = f'probe{probe_letter}'
        probe_module = session_schema.EphysModule(
            assembly_name=probe_name.upper(),
            arc_angle=0.0,
            module_angle=0.0,
            rotation_angle=0.0,
            primary_targeted_structure='none',
            ephys_probes=[session_schema.EphysProbeConfig(name=probe_name.upper())],
            manipulator_coordinates=manipulator_coords(probe_name, newscale_coords)
        )
        ephys_modules.append(probe_module)
    return ephys_modules


def sync_data(sync_path):
    return npc_sync.SyncDataset(io.BytesIO(sync_path.read_bytes()))


def session_start_end_times(sync_path):    
    start_timestamp = sync_data(sync_path).start_time
    end_timestamp = sync_data(sync_path).stop_time
    return (start_timestamp, end_timestamp)


def ephys_stream(sync_path, recording_dirs, motor_locs_path, platform_json) -> session_schema.Stream:
    available_probes = get_available_probes(platform_json)

    start_time, _ = session_start_end_times(sync_path)
    epmods = ephys_modules(available_probes, motor_locs_path)

    times = npc_ephys.get_ephys_timing_on_sync(sync=sync_path, recording_dirs=recording_dirs)
    ephys_timing_data = tuple(
        timing for timing in times if \
            (p := npc_session.extract_probe_letter(timing.device.name)) is None or p in available_probes
    )

    stream_first_time = min(timing.start_time for timing in ephys_timing_data)
    stream_last_time = max(timing.stop_time for timing in ephys_timing_data)

    return session_schema.Stream(
        stream_start_time=start_time + datetime.timedelta(seconds=stream_first_time),
        stream_end_time=start_time + datetime.timedelta(seconds=stream_last_time),
        ephys_modules=epmods,
        stick_microscopes=epmods, # cannot create ecephys modality without stick microscopes
        stream_modalities=[schema_modalities.ECEPHYS]
    )


def sync_stream(sync_path):
    sync_start, sync_end = session_start_end_times(sync_path)
    return session_schema.Stream(
            stream_start_time=sync_start,
            stream_end_time=sync_end,
            stream_modalities=[schema_modalities.BEHAVIOR],
            daq_names=['Sync']
    )

def video_stream(npexp_path, sync_path):
    session_start_time, _ = session_start_end_times(sync_path)
    video_frame_times = npc_mvr.mvr.get_video_frame_times(sync_path, npexp_path)

    stream_first_time = min(np.nanmin(timestamps) for timestamps in video_frame_times.values())
    stream_last_time = max(np.nanmax(timestamps) for timestamps in video_frame_times.values())

    return session_schema.Stream(
        stream_start_time=session_start_time + datetime.timedelta(seconds=stream_first_time),
        stream_end_time=session_start_time + datetime.timedelta(seconds=stream_last_time),
        camera_names=['Front camera', 'Side camera', 'Eye camera'],
        stream_modalities=[schema_modalities.BEHAVIOR_VIDEOS],
    )


def data_streams(npexp_path, sync_path, recording_dirs, motor_locs_path, platform_json) -> tuple[session_schema.Stream, ...]:
    data_streams = []
    data_streams.append(ephys_stream(sync_path, recording_dirs, motor_locs_path, platform_json))
    data_streams.append(sync_stream(sync_path))
    data_streams.append(video_stream(npexp_path, sync_path))
    return tuple(data_streams)


def epoch_from_opto_table(session, opto_table_path, sync_path):
    stim = aind_data_schema.core.session.StimulusModality
    session_start_time, _ = session_start_end_times(sync_path)

    script_obj = aind_data_schema.models.devices.Software(
        name=session.mtrain['regimen']['name'],
        version='1.0',
        url=session.mtrain['regimen']['script']
    )

    opto_table = pd.read_csv(opto_table_path)

    opto_params = {}
    for column in opto_table:
        if column in ('start_time', 'stop_time', 'stimulus_name'):
            continue
        param_set = set(opto_table[column].dropna())
        opto_params[column] = param_set

    params_obj = session_schema.VisualStimulation(
        stimulus_name="Optogenetic Stimulation",
        stimulus_parameters=opto_params,
        stimulus_template_name=[]
    )

    opto_epoch = session_schema.StimulusEpoch(
        stimulus_start_time=session_start_time + datetime.timedelta(seconds=opto_table.start_time.iloc[0]),
        stimulus_end_time=session_start_time + datetime.timedelta(seconds=opto_table.start_time.iloc[-1]),
        stimulus_name="Optogenetic Stimulation",
        software=[],
        script=script_obj,
        stimulus_modalities=[stim.OPTOGENETICS],
        stimulus_parameters=[params_obj],
    )

    return opto_epoch


def extract_stim_epochs(stim_table):
    epochs = []

    current_epoch = [None, 0.0, 0.0, {}, set()]
    epoch_start_idx = 0
    for i, row in stim_table.iterrows():
        if row['stimulus_name'] != current_epoch[0]:
            # end current epoch, summarize this epochs params and stim templates and start a new epoch
            for column in stim_table:
                if column in ('start_time', 'stop_time', 'stimulus_name'):
                    continue
                param_set = set(stim_table[column][epoch_start_idx:i].dropna())
                if column == 'stim_template':
                    current_epoch[4] = param_set
                else:
                    current_epoch[3][column] = param_set

            epochs.append(current_epoch)
            epoch_start_idx = i
            current_epoch = [row['stimulus_name'], row['start_time'], row['stop_time'], {}, set()]
        else:
            # otherwise, keep pushing epoch end time later
            current_epoch[2] = row['stop_time']

    # slice off dummy epoch from beginning
    return epochs[1:]


def epochs_from_stim_table(session, pkl_path, stim_table_path, sync_path):
    stim = aind_data_schema.core.session.StimulusModality
    session_start_time, _ = session_start_end_times(sync_path)

    software_obj = aind_data_schema.models.devices.Software(
        name='camstim',
        version=pkl_utils.load_pkl(pkl_path)['platform']['camstim'].split('+')[0],
        url='https://eng-gitlab.corp.alleninstitute.org/braintv/camstim'
    )

    script_obj = aind_data_schema.models.devices.Software(
        name=session.mtrain['regimen']['name'],
        version='1.0',
        url=session.mtrain['regimen']['script']
    )

    schema_epochs = []
    for epoch_name, epoch_start, epoch_end, stim_params, stim_template_names in extract_stim_epochs(pd.read_csv(stim_table_path)):
        params_obj = session_schema.VisualStimulation(
            stimulus_name=epoch_name,
            stimulus_parameters=stim_params,
            stimulus_template_name=stim_template_names
        )

        epoch_obj = session_schema.StimulusEpoch(
            stimulus_start_time=session_start_time + datetime.timedelta(seconds=epoch_start),
            stimulus_end_time=session_start_time + datetime.timedelta(seconds=epoch_end),
            stimulus_name=epoch_name,
            software=[software_obj],
            script=script_obj,
            stimulus_modalities=[stim.VISUAL],
            stimulus_parameters=[params_obj],
            # stimulus_device_names=get_device_names(epoch_name),
            # speaker_config=get_speaker_config(epoch_name),
            # reward_consumed_during_epoch=get_reward_consumed(epoch_name),
            # reward_consumed_unit="milliliter",
            # trials_total=get_num_trials(epoch_name),
            # trials_finished=get_num_trials(epoch_name),
            # trials_rewarded=get_num_trials_rewarded(epoch_name),
            # notes=nwb_epoch.notes.item()
        )
        schema_epochs.append(epoch_obj)

    return schema_epochs


def generate_session_json(session_id: str) -> None:
    session = np_session.Session(session_id)

    # sometimes data files are deleted on npexp, better to try files on lims
    try:
        recording_dirs = tuple(session.lims_path.rglob('**/Record Node*/experiment*/recording*'))
    except:
        recording_dirs = tuple(session.npexp_path.rglob('**/Record Node*/experiment*/recording*'))

    sync_path = tuple(session.npexp_path.glob('*.sync'))[0]
    session_start, session_end = session_start_end_times(sync_path)
    motor_locs_path = next(session.npexp_path.glob(f'{session.folder}.motor-locs.csv'))
 
    pkl_path = session.npexp_path / f'{session.folder}.stim.pkl'
    stim_table_path = session.npexp_path / f'{session.folder}_stim_epochs.csv' 
    opto_pkl_path = session.npexp_path / f'{session.folder}.opto.pkl'
    opto_table_path = session.npexp_path / f'{session.folder}_opto_epochs.csv' 
 
    platform_path = next(session.npexp_path.glob(f'{session.folder}_platform*.json'))
    platform_json = json.loads(platform_path.read_text())
    project_name = platform_json['project']
    experiment_info = json.loads(pl.Path(__file__).with_name('experiment_info.json').read_text())

    if not stim_table_path.exists():
        stim_utils.build_stimulus_table(pkl_path, sync_path, stim_table_path)
    print("getting stim epochs")
    stim_epochs = epochs_from_stim_table(session, pkl_path, stim_table_path, sync_path)

    if opto_pkl_path.exists() and not opto_table_path.exists():
        opto_conditions = experiment_info[project_name].get('opto_conditions', DEFAULT_OPTO_CONDITIONS)
        stim_utils.build_optogenetics_table(opto_pkl_path, sync_path, opto_conditions, opto_table_path)

    if opto_table_path.exists():
        stim_epochs.append(epoch_from_opto_table(session, opto_table_path, sync_path))

    session_json = session_schema.Session(
        experimenter_full_name=[platform_json['operatorID'].replace('.', ' ').title()],
        session_start_time=session_start,
        session_end_time=session_end,
        session_type=experiment_info.get('session_type', ''),
        iacuc_protocol=experiment_info.get('iacuc_protocol',''),
        rig_id=platform_json['rig_id'],
        subject_id=session.folder.split('_')[1],
        data_streams=data_streams(session.npexp_path, sync_path, recording_dirs, motor_locs_path, platform_json),
        stimulus_epochs=stim_epochs,
        mouse_platform_name=experiment_info.get('mouse_platform','Mouse Platform'),
        active_mouse_platform=experiment_info.get('active_mouse_platform', False),
        reward_consumed_unit='milliliter',
        notes='',
    )
    session_json.write_standard_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a session.json file for an ephys session')
    parser.add_argument('session_id', help='session ID (lims or np-exp foldername) or path to session folder')
    return parser.parse_args()


def main() -> None:
    generate_session_json(**vars(parse_args()))


if __name__ == '__main__':
    main()