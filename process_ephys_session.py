import aind_data_schema
import aind_data_schema.core.session as session_schema
import argparse
import datetime
import io
import json
import npc_ephys
import np_session
import npc_session
import npc_sessions
import npc_sync
import pandas as pd
import re
from aind_data_schema.models.modalities import Modality as schema_modalities
from aind_data_schema.models.coordinates import Coordinates3d as schema_coordinates
from utils import process_ephys_sync as stim_utils

# def _aind_session_metadata(self) -> aind_data_schema.core.session.Session:
    # return aind_data_schema.core.session.Session(
    #     experimenter_full_name=self.experimenter
    #     or ["NSB trainer"],
    #     session_start_time=self.session_start_time,
    #     session_end_time=self.sync_data.stop_time if self.is_sync else (max(self.epochs.stop_time) if self.epochs.stop_time else None),
    #     session_type=self.session_description.replace(" without CCF-annotated units", ""),
    #     iacuc_protocol="2104",
    #     rig_id=self._aind_rig_id,
    #     subject_id=str(self.id.subject),
    #     data_streams=list(self._aind_data_streams),
    #     stimulus_epochs=list(self._aind_stimulus_epochs),
    #     mouse_platform_name="Mouse Platform",
    #     active_mouse_platform=False,
    #     reward_delivery=self._aind_reward_delivery if self.is_task else None,
    #     reward_consumed_total=(
    #         (np.nanmean(self.sam.rewardSize) * len(self.sam.rewardTimes))
    #         if self.is_task
    #         else None
    #     ),
    #     reward_consumed_unit='milliliter',
    #     notes=self.notes,
    # )

def valid_probe_folder(probe_letter, recording_dir):
    for probe_dir in  (recording_dir / 'continuous').glob(f'*Probe{probe_letter}*'):
        data_files = ('continuous.dat', 'sample_numbers.npy', 'timestamps.npy')
        for data_file in data_files:
            if not (probe_dir / data_file).exists():
                return False
    return True


def get_available_probes(recording_dirs):
    available_probes = {'A','B','C','D','E','F'}
    for recording_dir in recording_dirs:
        invalid_probes = set(probe_letter for probe_letter in available_probes if not valid_probe_folder(probe_letter, recording_dir))
        print("invalid probes:",invalid_probes)
        available_probes  -= invalid_probes
    print("available probes:",available_probes)
    return tuple(available_probes)


def manipulator_coords(probe_name, newscale_coords):
    probe_row = newscale_coords.query(f"electrode_group == '{probe_name}'")
    return schema_coordinates(
        x=probe_row['x'].item(),
        y=probe_row['y'].item(),
        z=probe_row['z'].item(),
        unit="micrometer",
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


def experimenter_name(log_path) -> str:
    text = log_path.read_text()
    matches = re.findall(r"User\(\'(.+)\'\)", text)
    return [match.replace(".", " ").title() for match in matches]


def sync_data(sync_path):
    return npc_sync.SyncDataset(io.BytesIO(sync_path.read_bytes()))


def session_start_end_times(sync_path, sync_messages_paths) -> tuple[int, int]:
    try:
        extract_timestamp = lambda line: float(line[line.index(':')+1:].strip())
        start_times = [extract_timestamp(sync_messages_path.read_text().split('\n')[0]) for sync_messages_path in sync_messages_paths]
        start_timestamp = datetime.datetime.fromtimestamp(min(start_times) / 1e3)
    except:
        raise ValueError("sync_messages.txt is formatted unexpectedly")
    
    end_timestamp = sync_data(sync_path).stop_time       
    return (start_timestamp, end_timestamp)


def session_type() -> str:
    return "OpenScope Vippo Project!"


def iacuc_protocol() -> str:
    return "2117"


# def rig_id() -> str:
#     return "Indetermined (rig.json does not exist)"


# def subject_id() -> int:
#     pass


def ephys_streams(sync_path, recording_dirs, sync_messages_paths, motor_locs_path) -> session_schema.Stream:
    available_probes = get_available_probes(recording_dirs)

    start_time, end_time = session_start_end_times(sync_path, sync_messages_paths)
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


def data_streams() -> tuple[session_schema.Stream, ...]:
    data_streams = []
    data_streams.append(ephys_streams())
    return tuple(data_streams)


def mouse_platform_name() -> str:
    return "Mouse Platform"


def active_mouse_platform() -> str:
    return False    


def extract_epochs(stim_table):
    epochs = []

    current_epoch = [None, 0.0, 0.0]
    for i, row in stim_table.iterrows():
        if row["stimulus_name"] != current_epoch[0]:
            # end current epoch, start a new epoch
            epochs.append(current_epoch)
            current_epoch = [row["stimulus_name"], row["Start"], row["End"]]
        else:
            # otherwise, keep pushing epoch end time later
            current_epoch[2] = row["End"]

    # slice off dummy epoch from beginning
    return epochs[1:]


def stim_epochs_from_table(stim_table_path, sync_path, sync_messages_paths):
    stim = aind_data_schema.core.session.StimulusModality

    stim_table = pd.read_csv(stim_table_path)
    session_start_time, _ = session_start_end_times(sync_path, sync_messages_paths)
    print(stim_table)

    epochs = extract_epochs(stim_table)
    schema_epochs = []
    for epoch_name, epoch_start, epoch_end in epochs:
        schema_epochs.append(
            session_schema.StimulusEpoch(
                stimulus_start_time=session_start_time + datetime.timedelta(seconds=epoch_start),
                stimulus_end_time=session_start_time + datetime.timedelta(seconds=epoch_end),
                stimulus_name=epoch_name,
                # software=[
                #     aind_data_schema.models.devices.Software(
                #         name='PsychoPy',
                #         version='2022.1.2',
                #         url='https://www.psychopy.org/',
                #     ),
                # ],
                # script=aind_data_schema.models.devices.Software(
                #         name='DynamicRoutingTask',
                #         version=self.source_script.split('DynamicRoutingTask/')[-1],
                #         url=self.source_script,
                #     ),
                stimulus_modalities=[stim.VISUAL] if "opto" not in epoch_name.lower() else [stim.VISUAL, stim.OPTOGENETICS],
                # stimulus_parameters=get_parameters(epoch_name),
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

    subject_id = session.folder.split('_')[1]
    sync_messages_paths = tuple(session.npexp_path.rglob("**/*/sync_messages.txt"))
    sync_path = tuple(session.npexp_path.glob('*.sync'))[0]
    recording_dirs = tuple(session.npexp_path.rglob('**/Record Node*/experiment*/recording*'))
    session_start, session_end = session_start_end_times(sync_path, sync_messages_paths)
    motor_locs_path = next(session.npexp_path.glob(f'{session.folder}.motor-locs.csv'))

    pkl_path = session.npexp_path / f"{session.folder}.stim.pkl"
    stim_table_path = session.npexp_path / f"{session.folder}_stim_epochs.csv" 

    if not stim_table_path.exists():
        stim_utils.build_stimulus_table(pkl_path, sync_path, stim_table_path)

    session_json = session_schema.Session(
        experimenter_full_name=experimenter_name(session.npexp_path / 'exp\logs\debug.log'),
        session_start_time=session_start,
        session_end_time=session_end,
        session_type=session_type(),
        iacuc_protocol=iacuc_protocol(),
        rig_id='TODO',
        subject_id=subject_id,
        data_streams=[ephys_streams(sync_path, recording_dirs, sync_messages_paths, motor_locs_path)],
        stimulus_epochs=stim_epochs_from_table(stim_table_path, sync_path, sync_messages_paths),
        mouse_platform_name=mouse_platform_name(),
        active_mouse_platform=active_mouse_platform(),
        # reward_delivery=,
        # reward_consumed_total,
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