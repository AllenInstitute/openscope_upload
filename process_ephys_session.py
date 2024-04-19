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
import re
from aind_data_schema.models.modalities import Modality as schema_modalities
from aind_data_schema.models.coordinates import Coordinates3d as schema_coordinates
from utils import process_ephys_sync as stim_utils
from utils import pickle_functions as pkl_utils


# def valid_probe_folder(probe_letter, recording_dir):
#     for probe_dir in  (recording_dir / 'continuous').glob(f'*Probe{probe_letter}*'):
#         data_files = ('continuous.dat', 'sample_numbers.npy', 'timestamps.npy')
#         for data_file in data_files:
#             if not (probe_dir / data_file).exists():
#                 return False
#     return True


def get_available_probes(platform_json):
    # available_probes = {'A','B','C','D','E','F'}
    # for recording_dir in recording_dirs:
    #     invalid_probes = set(probe_letter for probe_letter in available_probes if not valid_probe_folder(probe_letter, recording_dir))
    #     print("invalid probes:",invalid_probes)
    #     available_probes  -= invalid_probes
    available_probes = [letter for letter in 'ABCDEF' if not platform_json['InsertionNotes'][f'Probe{letter}']["FailedToInsert"]]
    print("available probes:",available_probes)
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
        probe_name = f"probe{probe_letter}"
        probe_module = session_schema.EphysModule(
            assembly_name=probe_name.upper(),
            arc_angle=0.0,
            module_angle=0.0,
            rotation_angle=0.0,
            primary_targeted_structure="none",
            ephys_probes=[session_schema.EphysProbeConfig(name=probe_name.upper())],
            manipulator_coordinates=manipulator_coords(probe_name, newscale_coords)
        )
        ephys_modules.append(probe_module)
    return ephys_modules


# def experimenter_name(log_path) -> str:
#     text = log_path.read_text()
#     matches = re.findall(r"User\(\'(.+)\'\)", text)
#     return [match.replace(".", " ").title() for match in matches]


def sync_data(sync_path):
    return npc_sync.SyncDataset(io.BytesIO(sync_path.read_bytes()))


def session_start_end_times(sync_path):    
    start_timestamp = sync_data(sync_path).start_time
    end_timestamp = sync_data(sync_path).stop_time
    return (start_timestamp, end_timestamp)


def session_type() -> str:
    return "OpenScope Vippo Project!"


def iacuc_protocol() -> str:
    return "2117"


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
            daq_names=["Sync"]
    )

def video_stream(npexp_path, sync_path):
    session_start_time, _ = session_start_end_times(sync_path)
    video_frame_times = npc_mvr.mvr.get_video_frame_times(sync_path, npexp_path)

    stream_first_time = min(np.nanmin(timestamps) for timestamps in video_frame_times.values())
    stream_last_time = max(np.nanmax(timestamps) for timestamps in video_frame_times.values())

    return session_schema.Stream(
        stream_start_time=session_start_time + datetime.timedelta(seconds=stream_first_time),
        stream_end_time=session_start_time + datetime.timedelta(seconds=stream_last_time),
        camera_names=["Front camera", "Side camera", "Eye camera"],
        stream_modalities=[schema_modalities.BEHAVIOR_VIDEOS],
    )


def data_streams(npexp_path, sync_path, recording_dirs, motor_locs_path, platform_json) -> tuple[session_schema.Stream, ...]:
    data_streams = []
    data_streams.append(ephys_stream(sync_path, recording_dirs, motor_locs_path, platform_json))
    data_streams.append(sync_stream(sync_path))
    data_streams.append(video_stream(npexp_path, sync_path))
    return tuple(data_streams)


def mouse_platform_name() -> str:
    return "Mouse Platform"


def active_mouse_platform() -> str:
    return False    


def extract_epochs(stim_table):
    epochs = []

    current_epoch = [None, 0.0, 0.0, {}, set()]
    epoch_start_idx = 0
    for i, row in stim_table.iterrows():
        if row["stimulus_name"] != current_epoch[0]:
            # end current epoch, summarize this epochs params and stim templates and start a new epoch
            for column in stim_table:
                if column in ("Start", "End", "stimulus_name"):
                    continue
                param_set = set(stim_table[column][epoch_start_idx:i].dropna())
                if column == "stim_template":
                    current_epoch[4] = param_set
                else:
                    current_epoch[3][column] = param_set

            epochs.append(current_epoch)
            epoch_start_idx = i
            current_epoch = [row["stimulus_name"], row["Start"], row["End"], {}, set()]
        else:
            # otherwise, keep pushing epoch end time later
            current_epoch[2] = row["End"]

    # slice off dummy epoch from beginning
    return epochs[1:]


def stim_epochs_from_table(session, pkl_path, stim_table_path, sync_path):
    stim = aind_data_schema.core.session.StimulusModality

    stim_table = pd.read_csv(stim_table_path)
    session_start_time, _ = session_start_end_times(sync_path)

    schema_epochs = []
    for epoch_name, epoch_start, epoch_end, stim_params, stim_template_names in extract_epochs(stim_table):
        schema_epochs.append(
            session_schema.StimulusEpoch(
                stimulus_start_time=session_start_time + datetime.timedelta(seconds=epoch_start),
                stimulus_end_time=session_start_time + datetime.timedelta(seconds=epoch_end),
                stimulus_name=epoch_name,
                software=[aind_data_schema.models.devices.Software(
                        name="camstim",
                        version=pkl_utils.load_pkl(pkl_path)['platform']['camstim'].split('+')[0],
                        url='https://eng-gitlab.corp.alleninstitute.org/braintv/camstim'
                    )],
                script=aind_data_schema.models.devices.Software(
                        name=session.mtrain['regimen']['name'],
                        version='1.0',
                        url=session.mtrain['regimen']['script']
                    ),
                stimulus_modalities=[stim.OPTOGENETICS] if "opto" in epoch_name.lower() else [stim.VISUAL],
                # stimulus_parameters=,
                # stimulus_device_names=get_device_names(epoch_name),
                # speaker_config=get_speaker_config(epoch_name),
                # reward_consumed_during_epoch=get_reward_consumed(epoch_name),
                # reward_consumed_unit="milliliter",
                # trials_total=get_num_trials(epoch_name),
                # trials_finished=get_num_trials(epoch_name),
                # trials_rewarded=get_num_trials_rewarded(epoch_name),
                # notes=nwb_epoch.notes.item(),
            )
        )

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
    platform_path = next(session.npexp_path.glob(f'{session.folder}_platform*.json'))
    # platform = np_session.components.platform_json.PlatformJson(platform_path)
    platform_json = json.loads(platform_path.read_text())

    if not stim_table_path.exists():
        stim_utils.build_stimulus_table(pkl_path, sync_path, stim_table_path)

    session_json = session_schema.Session(
        experimenter_full_name=[platform_json['operatorID'].replace('.', ' ').title()],
        session_start_time=session_start,
        session_end_time=session_end,
        session_type=session_type(),
        iacuc_protocol=iacuc_protocol(),
        rig_id=platform_json['rig_id'],
        subject_id=session.folder.split('_')[1],
        data_streams=data_streams(session.npexp_path, sync_path, recording_dirs, motor_locs_path, platform_json),
        stimulus_epochs=stim_epochs_from_table(session, pkl_path, stim_table_path, sync_path),
        mouse_platform_name=mouse_platform_name(),
        active_mouse_platform=active_mouse_platform(),
        reward_consumed_unit='milliliter',
        notes='',
    )
    session_json.write_standard_file() # writes subject.json
    

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a session.json file for an ephys session')
    parser.add_argument('session_id', help='session ID (lims or np-exp foldername) or path to session folder')
    return parser.parse_args()


def main() -> None:
    generate_session_json(**vars(parse_args()))


if __name__ == '__main__':
    main()