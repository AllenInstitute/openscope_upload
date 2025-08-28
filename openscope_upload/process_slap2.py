import harp
import pathlib as pl
import matplotlib.pyplot as plt
import numpy as np
import requests, yaml, io
import scipy.io
import h5py
import argparse
import json
import np_codeocean
import np_session
import pandas as pd
import os
import traceback
import shutil
from scbc.slap2.experiment_summary import ExperimentSummary
import h5py
import numpy as np

from aind_data_schema_models.modalities import Modality as schema_modalities
from aind_data_schema_models.organizations import Organization
from aind_data_schema_models.platforms import Platform

from aind_metadata_mapper.slap2_harp.models import JobSettings as Slap2HarpJobSettings
from datetime import datetime

import requests
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.platforms import Platform

from aind_data_transfer_service.models.core import (
    SubmitJobRequestV2,
    Task,
    UploadJobConfigsV2,
)

from openscope_upload import harp_utils
from aind_metadata_mapper.slap2_harp.session import Slap2HarpSessionEtl



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

SLAP2_PATH = pl.Path(r"\\allen\aind\scratch\OpenScope\Slap2\Data")
SLAP2_RIG_JSON = pl.Path(r"\\allen\aind\scratch\OpenScope\Slap2\rig.json")


def add_rois_group(dmd_grp, expsum, dmd):
    dmd_idx = dmd-1
    rois_grp = dmd_grp.create_group("user_rois")

    user_rois = expsum.get_user_rois_info()
    if user_rois['masks'][dmd_idx]:
        masks = np.stack(user_rois['masks'][dmd_idx], axis=-1)
    else:
        masks = np.zeros(0)
    
    rois_grp.create_dataset("mask", data=masks)

    F_list = []
    Fsvd_list = []
    for trial in expsum.valid_trials[dmd_idx]:
        if len(expsum.get_user_rois_info()['masks'][dmd_idx]) != 0:
            F = expsum.get_user_roi_traces(dmd, trial)
            F_list.append(F)
            Fsvd = expsum.get_user_roi_traces(dmd, trial, trace_type='Fsvd')
            Fsvd_list.append(Fsvd)

    if F_list:
        F_concat = np.concatenate(F_list, axis=2)
        Fsvd_concat = np.concatenate(Fsvd_list, axis=2)
    else:
        F_concat = np.array([])
        Fsvd_concat = np.array([])
    rois_grp.create_dataset("F", data=F_concat)
    rois_grp.create_dataset("Fsvd", data=Fsvd_concat)


def get_concatenated_traces(exp, dmd, trace_type1, trace_type2):
    """
    Concatenate traces and timestamps for all valid trials for a given DMD and trace type.
    Returns (all_traces, all_timestamps)
    """
    dmd_idx = dmd-1
    all_traces = None
    slap2_trial_num_frames = []
    # TODO: calculate discard_frames if it becomes available in the ExperimentSummary
    # + account for eliminating some discarded_frames from the given traces?
    discard_frames = []

    print(f"getting concatenated traces for dmd {dmd}, trace type {trace_type1}")
    for trial_idx in range(exp.n_trials):
        trial = trial_idx + 1  # 0-indexed to 1-indexed

        # invalid trials are defined as having 0 frames
        if trial not in exp.valid_trials[dmd_idx]:
            slap2_trial_num_frames.append(0)
            continue

        traces = exp.get_traces(dmd, trial, trace_type1=trace_type1, trace_type2=trace_type2)
        if traces.ndim > 2:
            traces = traces[0]

        # sum(slap2_trial_num_frames) should equal all_traces.shape[1]
        # i.e. the total number of frames in the session
        slap2_trial_num_frames.append(traces.shape[1])
        if all_traces is None:
            all_traces = traces.T
        else:
            all_traces = np.concatenate((all_traces, traces.T))

    print(f"shape of traces: {all_traces.shape}")
    print(f"sum of trial_num_frames: {np.sum(slap2_trial_num_frames)}")
    assert np.sum(slap2_trial_num_frames) == all_traces.shape[0], "The recorded number of frames does not match the concatenated traces shape"
    return all_traces, np.array(slap2_trial_num_frames), np.array(discard_frames, dtype=bool)


def expsum_mat_to_h5(mat_path, h5_path, harp_path):
    """
    Convert ophys-SCBC-analysis experimentsummary.mat to experiment_summary.h5 with the required structure using ExperimentSummary class.
    Always extracts all HARP arrays and uses timing data for dF, dFF, F0 traces.
    """
    print(f"Converting {mat_path} to {h5_path}")
    expsum = ExperimentSummary(mat_path)
    with h5py.File(h5_path, "w") as h5:
        for dmd_idx in range(expsum.n_dmds):
            dmd = dmd_idx+1
            dmd_grp = h5.create_group(f"DMD{dmd}")
            # visualizations
            vis_grp = dmd_grp.create_group("visualizations")
            vis_grp.create_dataset("mean_im", data=expsum.get_summary_image(dmd, 'meanIM'))
            vis_grp.create_dataset("act_im", data=expsum.get_summary_image(dmd, 'actIM'))
            # vis_grp.create_dataset("per_trial_mean_im", data=np.zeros((expsum.n_trials, 1, 1)))
            # vis_grp.create_dataset("per_trial_act_im", data=np.zeros((expsum.n_trials, 1, 1)))

            # global
            # global_grp = dmd_grp.create_group("global")
            # global_grp.create_dataset("F", data=np.zeros((n_frames, 1)))
            
            # user rois
            add_rois_group(dmd_grp, expsum, dmd)

            # sources
            sources_grp = dmd_grp.create_group("sources")

            spatial_grp = sources_grp.create_group("spatial")
            trial_to_use = expsum.valid_trials[dmd_idx][0]
            fp_masks, fp_coords = expsum.get_footprints(dmd=dmd, trial=trial_to_use)
            spatial_grp.create_dataset("fp_masks",data=fp_masks,compression="gzip")
            spatial_grp.create_dataset("fp_coords",data=fp_coords,compression="gzip")

            # temporal group
            temporal_grp = sources_grp.create_group("temporal")
            dF_traces, _, _ = get_concatenated_traces(expsum, dmd, 'dF', 'denoised')
            dFF_traces, _, _ = get_concatenated_traces(expsum, dmd, 'dFF', 'denoised')
            F0_traces, trial_num_frames, discard_frames = get_concatenated_traces(expsum, dmd, 'F0', None)
            temporal_grp.create_dataset("dF", data=dF_traces)
            temporal_grp.create_dataset("dFF", data=dFF_traces)
            temporal_grp.create_dataset("F0", data=F0_traces)

            # frame_info
            fi_grp = dmd_grp.create_group("frame_info")
            fi_grp.create_dataset("trial_num_frames", data=trial_num_frames)
            fi_grp.create_dataset("discard_frames", data=discard_frames)


def generate_metadata_jsons(session_id, session_path, project_name, overwrite: bool = False) -> None:
    print(f'\ngenerating metadata for session {session_id}')

    print(session_path)
    projects_info = pd.read_csv(pl.Path(__file__).parent.parent / 'data/projects_info.csv', index_col='project_name')
    project_info = projects_info.loc[project_name]
    
    openscope_session_settings = Slap2HarpJobSettings(
        session_type="slap2",
        project_name="OpenScope",
        iacuc_protocol=str(project_info["iacuc_protocol"]),
        description=project_info["description"],
        overwrite_tables=True,
        mtrain_server="http://mtrain:5000",
        session_id=session_id,
        input_source=session_path,
        output_directory=session_path 
    )
    session_mapper = Slap2HarpSessionEtl(openscope_session_settings)
    session_mapper.run_job()

    if not os.path.exists(session_path / "rig.json") or overwrite:
        print(f"Copying rig.json")
        shutil.copy(SLAP2_RIG_JSON, session_path / 'rig.json')
    else:
        print(f"rig.json already exists, skipping copy, use --overwrite to force copy")


def upload_session(session_path, project_name, subject_id, test_upload: bool = False, no_upload: bool = False, force: bool = False):
    endpoint = "http://aind-data-transfer-service-dev" if test_upload else "http://aind-data-transfer-service"
    timestamp_str = session_path.name.split('_')[-1]
    acq_datetime = datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")

    slap2_task = Task(
        job_settings={
            "input_source": (
                (session_path / "slap2").as_posix()
            )
        }
    )
    harp_behavior_task = Task(
        job_settings={
            "input_source": (
                (session_path / "behavior").as_posix()
            )
        }
    )
    modality_transformation_settings = {Modality.SLAP.abbreviation: slap2_task, Modality.BEHAVIOR.abbreviation: harp_behavior_task}
    gather_preliminary_metadata = Task(
        job_settings={
            "metadata_dir": (
                session_path.as_posix()
            )
        }
    )
    print("FORCING UPLOAD?: ", force)
    upload_job_configs_v2 = UploadJobConfigsV2(
        job_type="default",
        project_name="Ophys Platform - SLAP2",
        platform=Platform.SLAP2,
        modalities=[Modality.SLAP, Modality.BEHAVIOR],
        subject_id=subject_id,
        acq_datetime=acq_datetime,
        tasks={
            "modality_transformation_settings": modality_transformation_settings,
            "gather_preliminary_metadata": gather_preliminary_metadata,
            "check_s3_folder_exists": {"skip_task": force},
        },
    )

    submit_request_v2 = SubmitJobRequestV2(
        upload_jobs=[upload_job_configs_v2],
    )
    post_request_content = submit_request_v2.model_dump(
        mode="json", exclude_none=True
    )
    submit_job_response = requests.post(
        url=f"{endpoint}/api/v2/submit_jobs",
        json=post_request_content,
    )
    print(submit_job_response.status_code)
    print(submit_job_response.json())


def prepare_session(session_id, overwrite: bool = False, no_upload: bool = False, test_upload: bool = False, force: bool = False):
    session_paths = list(SLAP2_PATH.rglob(session_id))
    if len(session_paths) != 1:
        return f"Expected 1 session path for {session_id}, found {len(session_paths)}"
    else:
        session_path = session_paths[0]

    project_name = "OpenScopePredictiveProcessing"
    subject_id = session_path.name.split('_')[0]

    # Find summary .mat file
    mat_matches = list(session_path.rglob("*Summary*.mat"))
    if not mat_matches:
        return f"No *Summary*.mat file found in {session_path}"
    mat_path = mat_matches[0]
    print(f"Found summary .mat file: {mat_path}")
    # Find .harp folder
    harp_matches = list(session_path.rglob("*.harp"))
    if not harp_matches:
        return f"No *.harp folder found in {session_path}"
    harp_path = harp_matches[0]
    print(f"Found .harp folder: {harp_path}")

    h5_path = session_path / "slap2" / "experiment_summary.h5"
    if os.path.exists(h5_path) and overwrite:
        print(f"{h5_path} already exists, overwriting...")
    if os.path.exists(h5_path) and not overwrite:
        return f"{h5_path} already exists, skipping conversion. Use --overwrite to force conversion."
    if not os.path.exists(h5_path) or overwrite:
        try:
            expsum_mat_to_h5(mat_path, h5_path, harp_path)
        except Exception as e:
            return f"Error converting {session_id} from a .mat to .h5:\n{traceback.format_exc()}"

    generate_metadata_jsons(session_id, session_path, project_name, overwrite=overwrite)

    if not no_upload:
        try:
            upload_session(session_path, project_name, subject_id, test_upload, no_upload, force)
            return f"{session_id} upload succesfully triggered!"
        except Exception as e:
            return f"{session_id} upload failed with error:\n{traceback.format_exc()}"
    if no_upload:
        print("not uploading!")
        return f"{session_id} upload skipped"


def main():
    parser = argparse.ArgumentParser(description="Convert experimentsummary.mat to experiment_summary.h5.")
    parser.add_argument('session_ids', nargs='+', help='one or more session IDs for slap2 sessions (mouse_date)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing metadata or experiment_summary.h5 files if they exist')
    parser.add_argument('--no_upload', action='store_true', help='Don\'t run an upload job, just generate metadata files')
    parser.add_argument('--test_upload', action='store_true', help='Run the upload at the dev endpoint')
    parser.add_argument('--force', action='store_true', help="enable `force_cloud_sync` option, re-uploading and re-making raw asset even if data exists on S3")
    args = parser.parse_args()

    log = []
    for session_id in args.session_ids:
        log.append(prepare_session(session_id, args.overwrite, args.no_upload, args.test_upload, args.force))
    print("="*64)
    for logstr in log:
        print(logstr)
    print("="*64)



if __name__ == "__main__":
    main()

