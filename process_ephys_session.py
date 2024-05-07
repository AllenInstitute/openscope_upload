import aind_data_schema
import aind_data_schema.core.session as session_schema
import aind_metadata_mapper
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
from aind_metadata_mapper.ephys.camstim_session import CamstimSession
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


def generate_data_description_json(project_name: str) -> None:
    projects_info = pd.read_csv('./projects_info.csv')
    print(projects_info)
    print(projects_info[project_name])


def generate_session_json(session_id: str) -> str:
    session = np_session.Session(session_id)

    sync_path = session.npexp_path / f'{session.folder}.sync' 
    pkl_path = session.npexp_path / f'{session.folder}.stim.pkl'
    stim_table_path = session.npexp_path / f'{session.folder}_stim_epochs.csv' 
    opto_pkl_path = session.npexp_path / f'{session.folder}.opto.pkl'
    opto_table_path = session.npexp_path / f'{session.folder}_opto_epochs.csv' 
 
    platform_path = next(session.npexp_path.glob(f'{session.folder}_platform*.json'))
    platform_json = json.loads(platform_path.read_text())
    project_name = platform_json['project']
    experiment_info = json.loads(pl.Path(__file__).with_name('experiment_info.json').read_text())
    
    if not stim_table_path.exists():
        print("building stim table")
        stim_utils.build_stimulus_table(pkl_path, sync_path, stim_table_path)
    if opto_pkl_path.exists() and not opto_table_path.exists():
        print("building opto table")
        opto_conditions = experiment_info[project_name].get('opto_conditions', DEFAULT_OPTO_CONDITIONS)
        stim_utils.build_optogenetics_table(opto_pkl_path, sync_path, opto_conditions, opto_table_path)

    session_settings = experiment_info[project_name]
    session_mapper = CamstimSession(session_id, session_settings)
    session_mapper.generate_session_json()
    session_mapper.write_session_json()

    return project_name


def generate_jsons(session_id: str) -> None:
    project_name = generate_session_json(session_id)
    generate_data_description_json(project_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a session.json file for an ephys session')
    parser.add_argument('session_id', help='session ID (lims or np-exp foldername) or path to session folder')
    return parser.parse_args()


def main() -> None:
    generate_session_json(**vars(parse_args()))


if __name__ == '__main__':
    main()