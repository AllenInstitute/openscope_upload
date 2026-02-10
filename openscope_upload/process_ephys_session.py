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

from aind_metadata_mapper.models import SessionSettings, JobSettings as GatherMetadataJobSettings
from aind_metadata_mapper.open_ephys.models import JobSettings as CamstimEphysSessionSettings


from aind_metadata_mapper.open_ephys.camstim_ephys_session import CamstimEphysSessionEtl

from codeocean.computation import RunParams, DataAssetsRunParam
from codeocean.data_asset import DataAssetParams
from aind_codeocean_pipeline_monitor.models import (
    PipelineMonitorSettings,
    CaptureSettings,
)
from aind_data_schema_models.data_name_patterns import DataLevel

# from aind_data_transfer_models.core import (
#     ModalityConfigs,
#     BasicUploadJobConfigs,
#     SubmitJobRequest,
#     CodeOceanPipelineMonitorConfigs,
# )
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


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

THIS_FILE_DIR = pl.Path(__file__).parent

def generate_rig_json(session: np_session.Session, overwrite: bool = False):
    if (session.npexp_path / 'rig.json').exists():
        print('rig.json already exists')
        if not overwrite:
            return
        print('overwriting')

    platform_path = next(session.npexp_path.glob(f'{session.folder}_platform*.json'))
    platform_json = json.loads(platform_path.read_text())
    rig_id = platform_json['rig_id']

    rig_json_template = json.loads((THIS_FILE_DIR.parent / 'data' / 'openscope_rig.json').read_text())
    rig_json_template['rig_id'] = rig_id
    rig_json_template['modification_date'] = str(datetime.date.today())

    with open(session.npexp_path / 'rig.json', 'w', encoding='utf-8') as f:
        json.dump(rig_json_template, f, ensure_ascii=False, indent=4)


# def generate_data_description_json(project_name: str, session: np_session.Session, overwrite: bool = False) -> None:
#     if (session.npexp_path / 'data_description.json').exists():
#         print('data_description.json already exists')
#         if not overwrite:
#             return
#         print('overwriting')
    
#     subject_id = session.folder.split('_')[1]
#     projects_info = pd.read_csv(pl.Path(__file__).parent / 'projects_info.csv', index_col='project_name')
#     project_info = projects_info.loc[project_name]

#     funding_schemas = [Funding(funder=Organization.AI)]
#     for funding_source in project_info['funding_sources'].split(','):
#         name, grant_num = funding_source.split(' ',maxsplit=1)
#         funding_schemas.append(Funding(funder=organization_map[name], grant_number=grant_num))

#     create_time = datetime.datetime.now()
#     data_description = DataDescription(
#         label=project_name,
#         license=project_info['license'],
#         platform=Platform.ECEPHYS,
#         subject_id=subject_id,
#         creation_time=create_time,
#         institution=Organization.AIND,
#         funding_source=funding_schemas,
#         data_level='derived',
#         investigators=[PIDName(name=name) for name in project_info['investigators'].split(',')],
#         modality=[modality_map[mod] for mod in project_info['modalities'].split(',')]
#     )
#     data_description.write_standard_file(session.npexp_path)

def generate_session_json(session_id: str, session: np_session.Session, overwrite: bool = False) -> str:
    platform_path = next(session.npexp_path.glob(f'{session.folder}_platform*.json'))
    platform_json = json.loads(platform_path.read_text())
    project_name = platform_json['project']

    print(project_name)
    projects_info = pd.read_csv(THIS_FILE_DIR / 'data' / 'projects_info.csv', index_col='project_name')

    print(projects_info)
    project_info = projects_info.loc[project_name]


    if (session.npexp_path / 'session.json').exists():
        print('session.json already exists')
        if not overwrite:
            return project_name
        print('overwriting')

    # session_settings = project_info[project_name]
    session_settings = project_info
    print("session settings",session_settings)
    if overwrite:
        session_settings['overwrite_tables'] = True
    session_settings["session_id"] = session_id
    session_mapper = CamstimEphysSessionEtl(session_settings)
    session_mapper.generate_session_json()
    session_mapper.write_session_json()

    return project_name


def fetch_rig_json(session: np_session.Session):
    import requests
    rig_name = 'NP1'
    rig_endpoint = 'http://aind-metadata-service/rig'
    res = requests.get(f'{rig_endpoint}/{rig_name}')
    print(res)


def get_co_configs():
    # return {
    #     schema_modalities.BEHAVIOR_VIDEOS.abbreviation: PipelineMonitorSettings(
    #             run_params=RunParams(
    #                 pipeline_id="7935d378-ce0e-4129-8774-81e9c8573bc2",
    #                 data_assets=[DataAssetsRunParam(id="", mount="")],
    #                 parameters=[],
    #             ),
    #             capture_settings=CaptureSettings(
    #                 process_name_suffix="eye-tracking",
    #                 target={"aws": {"bucket": "aind-open-data"}},
    #                 permissions={"everyone": "viewer"},
    #                 tags=[DataLevel.DERIVED.value, "eye_tracking"],
    #             ),
    #         ),
    #     schema_modalities.ECEPHYS.abbreviation: PipelineMonitorSettings(
    #             run_params=RunParams(
    #                 pipeline_id="daef0b82-2f12-4122-964d-efa5f608ad69",
    #                 data_assets=[DataAssetsRunParam(id="", mount="ecephys")],
    #                 parameters=[],
    #             ),
    #             capture_settings=CaptureSettings(
    #                 process_name_suffix="sorted",
    #                 target={"aws": {"bucket": "aind-open-data"}},
    #                 permissions={"everyone": "viewer"},
    #                 tags=[DataLevel.DERIVED.value, "ecephys_sorted"],
    #             ),
    #         )
    #     }

    # eye_tracking_settings_json = {
    #     "run_params": {
    #         "pipeline_id": "7935d378-ce0e-4129-8774-81e9c8573bc2",
    #         "data_assets": [{"id": "", "mount": ""}],
    #         "parameters": []
    #     },
    #     "capture_settings": {
    #         "process_name_suffix": "eye_tracking",
    #         "tags": ["derived", "eye_tracking"],
    #         "permissions": {"everyone": "viewer"},
    #         "target": {"aws": {"bucket": "aind-open-data"}},
    #         "custom_metadata": {"data level": "derived",},
    #     }
    # }
    spike_sorting_settings_json = {
        "run_params": {
            # "pipeline_id": "daef0b82-2f12-4122-964d-efa5f608ad69",
            "pipeline_id": "e16bc028-30b1-4aa2-89f9-a2cb27aaf844",
            "data_assets": [{"id": "", "mount": "ecephys"}],
            "parameters": []
        },
        "capture_settings": {
            "process_name_suffix": "sorted",
            "tags": ["derived", "ecephys_sorted"],
            "permissions": {"everyone": "viewer"},
            "target": {"aws": {"bucket": "aind-open-data"}},
            "custom_metadata": {"data level": "derived",},
        }
    }

    return {
        schema_modalities.ECEPHYS.abbreviation: PipelineMonitorSettings(**spike_sorting_settings_json)
    }


def generate_jsons(session_ids: list[str], force: bool = False, no_upload: bool = False, overwrite: bool = False, test_upload: bool = False) -> None:
    log = []
    for session_id in session_ids:
        # fetch_rig_json(session) # do this when slims is up and running
        print(f'\ngenerating jsons for session {session_id}')
        session = np_session.Session(session_id)
        print(session.npexp_path)
        # project_name = generate_session_json(session_id, session, overwrite=overwrite)
        # generate_data_description_json(project_name, session, overwrite=overwrite)
        generate_rig_json(session, overwrite=overwrite)

        print(session.folder)
        platform_path = next(session.npexp_path.glob(f'{session.folder}_platform*.json'))
        platform_json = json.loads(platform_path.read_text())
        project_name = platform_json['project']

        projects_info = pd.read_csv(THIS_FILE_DIR.parent / 'data' / 'projects_info.csv', index_col='project_name')
        project_info = projects_info.loc[project_name]

        openscope_session_settings = CamstimEphysSessionSettings(
            session_type="ecephys",
            project_name=project_name,
            iacuc_protocol=str(project_info["iacuc_protocol"]),
            description=project_info["description"],
            overwrite_tables=True,
            mtrain_server="http://mtrain:5000",
            session_id=session_id,
            input_source=session.npexp_path,
            output_directory=session.npexp_path
        )
        session_mapper = CamstimEphysSessionEtl(openscope_session_settings)
        session_mapper.run_job()
        

        codeocean_configs = get_co_configs()
        if not no_upload:
            # try:
            print(session_id)
            np_codeocean.upload_session(session_id, force=force, hpc_upload_job_email=USER_EMAIL, test=test_upload, codeocean_pipeline_settings=codeocean_configs)
            log.append(f"{session_id} upload succesfully triggered!")
            # except Exception as e:
            #     log.append(f"{session_id} upload failed with error: {e}")
        if no_upload:
            print("not uploading!")
            log.append(f"{session_id} upload skipped")

    print("="*64)
    for logstr in log:
        print(logstr)
    print("="*64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a session.json file for an ephys session')
    parser.add_argument('session_ids', nargs='+', help='one or more session IDs (lims or np-exp foldername) or path to session folder')
    parser.add_argument('--no_upload', action='store_true', help='Don\'t run an upload job, just generate metadata files on npexp')
    parser.add_argument('--force', action='store_true', help="enable `force_cloud_sync` option, re-uploading and re-making raw asset even if data exists on S3")
    parser.add_argument('--overwrite', action='store_true', help='overwrite metadata files that already exist')
    parser.add_argument('--test_upload', action='store_true', help='Run a test upload to the aind data transfer dev endpoint')
    return parser.parse_args()


def main() -> None:
    generate_jsons(**vars(parse_args()))


if __name__ == '__main__':
    main()
