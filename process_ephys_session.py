import aind_data_schema.core.session as session_schema
from aind_data_schema.models.modalities import Modality as schema_modalities
from aind_data_schema.models.coordinates import Coordinates3d as schema_coordinates
import argparse
import datetime
import json
import np_session
import pathlib
import re

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


def ephys_modules():
    ephys_modules = []

    available_probes = [] # TODO!!!
    for probe_name in available_probes:
        probe_module = session_schema.EphysModule(
            assembly_name=probe_name.upper(),
            arc_angle=0.0,
            module_angle=0.0,
            rotation_angle=0.0,
            primary_targeted_structure="none",
            ephys_probes=[session_schema.EphysProbeConfig(name=probe_name.upper())]
        )
        ephys_modules.append(probe_module)
    return ephys_modules


def experimenter_name(log_path) -> str:
    text = log_path.read_text()
    matches = re.findall(r"User\(\'(.+)\'\)", text)
    return [match.replace(".", " ").title() for match in matches]


def session_start_end_times(sync_messages_paths) -> tuple[int, int]:
    try:
        extract_timestamp = lambda line: float(line[line.index(':')+1:].strip())
        start_times = [extract_timestamp(sync_messages_path.read_text().split('\n')[0]) for sync_messages_path in sync_messages_paths]
        start_timestamp = datetime.datetime.fromtimestamp(min(start_times) / 1e3)
    except:
        raise ValueError("sync_messages.txt is formatted unexpectedly")
    print(start_timestamp)
    return (start_timestamp, -1)


def session_type() -> str:
    return "OpenScope Vippo Project!"


def iacuc_protocol() -> str:
    return "2117"


# def rig_id() -> str:
#     return "Indetermined (rig.json does not exist)"


# def subject_id() -> int:
#     pass

def ephys_streams() -> session_schema.Stream:
    start_time, end_time = session_start_end_times()
    epmods = ephys_modules()
    return session_schema.Stream(
        # stream_start_time=start_time + datetime.timedelta(seconds=min()),
        # stream_end_time=start_time + datetime.timedelta(seconds=max()),
        stream_start_time = 0,
        stream_end_time = -1
        ephys_modules = epmods
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


def generate_session_json(session_id: str) -> None:
    session = np_session.Session(session_id)
    print(session.folder)
    print(session.npexp_path)
    print(experimenter_name(session.npexp_path / 'exp\logs\debug.log'))
    print(session_start_end_times(session.npexp_path.rglob("**/*/sync_messages.txt")))
    print(session.rig)

    session_start, session_end = session_start_end_times(session.npexp_path.rglob("**/*/sync_messages.txt"))
    subject_id = session.folder.split("_")[1]

    session_json = session_schema.Session(
        experimenter_full_name=experimenter_name(session.npexp_path / 'exp\logs\debug.log'),
        session_start_time=session_start,
        session_end_time=session_end,
        session_type=session_type(),
        iacuc_protocol=iacuc_protocol(),
        rig_id="TODO",
        subject_id=subject_id,
        data_streams=[],
        stimulus_epochs=[],
        mouse_platform_name=mouse_platform_name(),
        active_mouse_platform=active_mouse_platform(),
        # reward_delivery=,
        # reward_consumed_total,
        reward_consumed_unit='milliliter',
        notes="",
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