import argparse
import datetime
import json
import np_codeocean
import np_session
import pandas as pd
import pathlib as pl

from aind_data_schema.core.data_description import Funding, DataDescription
from aind_data_schema_models.modalities import Modality as schema_modalities
from aind_data_schema_models.organizations import Organization
from aind_data_schema_models.pid_names import PIDName
from aind_data_schema_models.platforms import Platform
from aind_metadata_mapper.ephys.camstim_ephys_session import CamstimEphysSession

from utils import process_ephys_sync as stim_utils


# defaults
DEFAULT_OPTO_CONDITIONS = {
    '0': {
        'duration': .01,
        'name': '1Hz_10ms',
        'condition': '10 ms pulse at 1 Hz'
    },
    '1': {
        'duration': .002,
        'name': '1Hz_2ms',
        'condition': '2 ms pulse at 1 Hz'
    },
    '2': {
        'duration': 1.0,
        'name': '5Hz_2ms',
        'condition': '2 ms pulses at 5 Hz'
    },
    '3': {
        'duration': 1.0,
        'name': '10Hz_2ms',
        'condition': '2 ms pulses at 10 Hz'
    },
    '4': {
        'duration': 1.0,
        'name': '20Hz_2ms',
        'condition': '2 ms pulses at 20 Hz'
    },
    '5': {
        'duration': 1.0,
        'name': '30Hz_2ms',
        'condition': '2 ms pulses at 30 Hz'
    },
    '6': {
        'duration': 1.0,
        'name': '40Hz_2ms',
        'condition': '2 ms pulses at 40 Hz'
    },
    '7': {
        'duration': 1.0,
        'name': '50Hz_2ms',
        'condition': '2 ms pulses at 50 Hz'
    },
    '8': {
        'duration': 1.0,
        'name': '60Hz_2ms',
        'condition': '2 ms pulses at 60 Hz'
    },
    '9': {
        'duration': 1.0,
        'name': '80Hz_2ms',
        'condition': '2 ms pulses at 80 Hz'
    },
    '10': {
        'duration': 1.0,
        'name': 'square_1s',
        'condition': '1 second square pulse: continuously on for 1s'
    },
    '11': {
        'duration': 1.0,
        'name': 'cosine_1s',
        'condition': 'cosine pulse'
    },
}

organization_map = {
    'NINDS': Organization.NINDS
}

modality_map = {
    'ephys': schema_modalities.ECEPHYS,
    'behavior-videos': schema_modalities.BEHAVIOR_VIDEOS,
    'behavior': schema_modalities.BEHAVIOR,
    'ophys': schema_modalities.POPHYS
}

USER_EMAIL = "carter.peene@alleninstitute.org"


def generate_rig_json(session: np_session.Session, overwrite: bool = False):
    if (session.npexp_path / 'rig.json').exists():
        print('rig.json already exists')
        if not overwrite:
            return
        print('overwriting')

    platform_path = next(session.npexp_path.glob(f'{session.folder}_platform*.json'))
    platform_json = json.loads(platform_path.read_text())
    rig_id = platform_json['rig_id']

    rig_json_template = json.loads((pl.Path(__file__).parent / 'openscope_rig.json').read_text())
    rig_json_template['rig_id'] = rig_id
    rig_json_template['modification_date'] = str(datetime.date.today())

    with open(session.npexp_path / 'rig.json', 'w', encoding='utf-8') as f:
        json.dump(rig_json_template, f, ensure_ascii=False, indent=4)


def generate_data_description_json(project_name: str, session: np_session.Session, overwrite: bool = False) -> None:
    if (session.npexp_path / 'data_description.json').exists():
        print('data_description.json already exists')
        if not overwrite:
            return
        print('overwriting')
    
    subject_id = session.folder.split('_')[1]
    projects_info = pd.read_csv(pl.Path(__file__).parent / 'projects_info.csv', index_col='project_name')
    project_info = projects_info.loc[project_name]

    funding_schemas = [Funding(funder=Organization.AI)]
    for funding_source in project_info['funding_sources'].split(','):
        name, grant_num = funding_source.split(' ',maxsplit=1)
        funding_schemas.append(Funding(funder=organization_map[name], grant_number=grant_num))

    create_time = datetime.datetime.now()
    data_description = DataDescription(
        label=project_name,
        license=project_info['license'],
        platform=Platform.ECEPHYS,
        subject_id=subject_id,
        creation_time=create_time,
        institution=Organization.AIND,
        funding_source=funding_schemas,
        data_level='derived',
        investigators=[PIDName(name=name) for name in project_info['investigators'].split(',')],
        modality=[modality_map[mod] for mod in project_info['modalities'].split(',')]
    )
    data_description.write_standard_file(session.npexp_path)


def generate_session_json(session_id: str, session: np_session.Session, overwrite: bool = False) -> str:
    platform_path = next(session.npexp_path.glob(f'{session.folder}_platform*.json'))
    platform_json = json.loads(platform_path.read_text())
    project_name = platform_json['project']
    experiment_info = json.loads(pl.Path(__file__).with_name('experiment_info.json').read_text())

    if (session.npexp_path / 'session.json').exists():
        print('session.json already exists')
        if not overwrite:
            return project_name
        print('overwriting')

    session_settings = experiment_info[project_name]
    if overwrite:
        session_settings['overwrite_tables'] = True
    session_mapper = CamstimEphysSession(session_id, session_settings)
    session_mapper.generate_session_json()
    session_mapper.write_session_json()

    return project_name


def fetch_rig_json(session: np_session.Session):
    import requests
    rig_name = 'NP1'
    rig_endpoint = 'http://aind-metadata-service/rig'
    res = requests.get(f'{rig_endpoint}/{rig_name}')
    print(res)


def generate_jsons(session_ids: str, force: bool = False, no_upload: bool = False, overwrite: bool = False) -> None:
    for session_id in session_ids:
        session = np_session.Session(session_id)
        # fetch_rig_json(session)
        print(f'\ngenerating jsons for session {session_id}')
        session = np_session.Session(session_id)
        project_name = generate_session_json(session_id, session, overwrite=overwrite)
        generate_data_description_json(project_name, session, overwrite=overwrite)
        generate_rig_json(session, overwrite=overwrite)
        if not no_upload:
            np_codeocean.upload_session(session_id, force=force, hpc_upload_job_email=USER_EMAIL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a session.json file for an ephys session')
    parser.add_argument('session_ids', nargs='+', help='one or more session IDs (lims or np-exp foldername) or path to session folder')
    parser.add_argument('--no_upload', action='store_true', help='Don\'t run an upload job, just generate metadata files on npexp')
    parser.add_argument('--force', action='store_true', help="enable `force_cloud_sync` option, re-uploading and re-making raw asset even if data exists on S3")
    parser.add_argument('--overwrite', action='store_true', help='overwrite metadata files that already exist')
    return parser.parse_args()


def main() -> None:
    generate_jsons(**vars(parse_args()))


if __name__ == '__main__':
    main()