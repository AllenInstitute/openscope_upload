from codeocean.computation import RunParams, DataAssetsRunParam, PipelineProcessParams, NamedRunParam
from codeocean import CodeOcean
from aind_codeocean_api.codeocean import CodeOceanClient
from aind_codeocean_pipeline_monitor.models import (CaptureSettings,
                                                    PipelineMonitorSettings)

import time
from datetime import datetime as dt
from datetime import timezone as tz
import argparse
import pandas as pd
from pathlib import Path
import os
from io import StringIO


# canonical monitor capsule
# monitor_pipeline_capsule_id = "fbdddd96-6d2a-4e40-88c8-45f3f84cfaf3"

# carter's personal use
# monitor_pipeline_capsule_id = "779ec012-5d39-4cff-a3fb-23b050903d02"

# all user's monitor
# monitor_pipeline_capsule_id = "567b5b98-8d41-413b-9375-9ca610ca2fd3"

# monitor that will run with users credentials
# monitor_pipeline_capsule_id = "b8d1f143-d0f6-40d7-b5a3-cb1f1eb85bc4"

# carter's monitor will run with my credentials
monitor_pipeline_capsule_id = "7a6dd345-e3fe-4f86-b2c3-c42c0d673c3c"

co_api_token = os.getenv("CODEOCEAN_TOKEN")
co_domain = "https://codeocean.allenneuraldynamics.org"
client = CodeOcean(domain=co_domain, token=co_api_token)
print(client.token)

def get_monitor_settings(pipeline_id, subject_id, data_assets, dandiset_id) -> PipelineMonitorSettings:
    return PipelineMonitorSettings(
        run_params=RunParams(
            capsule_id=pipeline_id,
            data_assets=data_assets,
            processes=[
                PipelineProcessParams(name="capsule_nwb_zarr_hdf_5_conversion_12",named_parameters=[NamedRunParam(param_name='input_nwb_dir',value='ecephys_sorted/nwb')]),
                PipelineProcessParams(name="capsule_nwb_packaging_stimulus_6",named_parameters=[NamedRunParam(param_name='input_csv_dir',value='stim_tables')]),
                PipelineProcessParams(name="capsule_aind_ccf_nwb_18",named_parameters=[NamedRunParam(param_name='convert_ibl_bregma_to_ccf',value='true')]),
                PipelineProcessParams(name="capsule_aind_running_speed_nwb_14",named_parameters=[NamedRunParam(param_name='use_input_nwb',value='True')]),
                PipelineProcessParams(name="capsule_aind_eye_tracking_nwb_15",named_parameters=[NamedRunParam(param_name='use_input_nwb',value='True')]),
                PipelineProcessParams(name="capsule_carters_aind_dandi_upload_16",named_parameters=[NamedRunParam(param_name='dandiset_id',value=dandiset_id)]),
            ]
        ),
        capture_settings={
            "process_name_suffix": "nwb",
            "tags": ["derived", "ephys", "nwb", subject_id],
            "custom_metadata": {
                "data level": "derived",
                "experiment type": "ephys",
                "subject id": subject_id
            },
            "permissions": {"everyone": "viewer"},
            "target": {"aws": {"bucket": "aind-open-data"}}
        },
    )


def trigger_monitor(jobs, pipeline_id, subject_id, data_assets, dandiset_id):
    settings = get_monitor_settings(pipeline_id, subject_id, data_assets, dandiset_id)
    pipeline_params = settings.model_dump_json(exclude_none=True)
    print(pipeline_params)
    monitor_params = RunParams(
        capsule_id=monitor_pipeline_capsule_id, parameters=[pipeline_params]
    )
    print(monitor_params)
    monitor_run_comp = client.computations.run_capsule(monitor_params)
    job_id = monitor_run_comp.id
    jobs.append(monitor_run_comp)
    print(f"Job {job_id} started")
    print(f"Jobs started: {len(jobs)}")
    return jobs



# while len(jobs) >= args.max_jobs:
#     for job in jobs:
#         if client.computations.get_computation(job.id).state in [
#             ComputationState.Completed,
#             ComputationState.Failed,
#         ]:
#             if job.state == ComputationState.Failed:
#                 logging.error(f"Job {job.id} failed")
#             jobs.remove(job)
#             break
#     time.sleep(args.sleep)


def main():
    # v2 NWB packaging pipeline for ephys
    pipeline_id = "111e7ef9-bc20-4faa-9169-d8e9f373e752"

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_mounts_csv", type=str, required=True)
    parser.add_argument("--dandiset_id", type=str, required=True)
    parser.add_argument("--exclude_n_cols", type=int, default=1)
    args = parser.parse_args()

    # asset_mounts = pd.read_csv(Path(args.asset_mounts_csv))
    # eliminate trailing commas
    with open(args.asset_mounts_csv) as f:
        lines = [line.rstrip(",\n") for line in f]
    asset_mounts = pd.read_csv(StringIO("\n".join(lines)))

    dandiset_id = args.dandiset_id
    enc = args.exclude_n_cols
    assert type(dandiset_id) is str and len(dandiset_id) == 6

    jobs = []
    for index, row in asset_mounts.iterrows():
        data_assets = [DataAssetsRunParam(id=asset_id,mount=mount) for mount, asset_id in zip(asset_mounts.columns[enc:], row[enc:])]
        print(data_assets)
        subject_id = str(row["mouse_id"])
        trigger_monitor(jobs, pipeline_id, subject_id, data_assets, dandiset_id)


# def main():
#     pipeline_id = "77d3322c-be67-4797-b9e6-c518c56fd248"

#     co_api_token = os.getenv("CODEOCEAN_TOKEN")
#     co_domain = "https://codeocean.allenneuraldynamics.org"
#     client = CodeOcean(domain=co_domain, token=co_api_token)
#     print(client.token)

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--asset_mounts_csv", type=str, required=True)
#     parser.add_argument("--dandiset_id", type=str, required=True)
#     parser.add_argument("--exclude_n_cols", type=int, default=1)
#     args = parser.parse_args()

#     asset_mounts = pd.read_csv(Path(args.asset_mounts_csv))
#     dandiset_id = args.dandiset_id
#     enc = args.exclude_n_cols
#     assert type(dandiset_id) is str and len(dandiset_id) == 6

#     data = []
#     for index, row in asset_mounts.iterrows():
#         data_assets = [DataAssetsRunParam(id=asset_id,mount=mount) for mount, asset_id in zip(asset_mounts.columns[enc:], row[enc:])]

#         print(data_assets)
#         processes=[
#             PipelineProcessParams(name="capsule_carters_aind_dandi_upload_9",named_parameters=[NamedRunParam(param_name='dandiset_id',value=dandiset_id)]),
#             PipelineProcessParams(name="capsule_combine_nwb_8",named_parameters=[NamedRunParam(param_name='output_format',value='hdf5')]),
#         ]
#         run_params = RunParams(capsule_id=pipeline_id,data_assets=data_assets,processes=processes)
#         print("="*64,run_params,"="*64)
#         print(f"\n\n\nRunning dataset {row[0]}")
#         computation = client.computations.run_capsule(run_params)
#         run_response = computation
#         print(f"Run response: {run_response}")
#         data.append(run_response)
#         time.sleep(5)


if __name__ == "__main__":
    main()


# run_asset = {
#     "data_assets": [
#         {
#             "id": session_asset_id,
#             "mount": "ecephys_session"
#         },
#         {
#             "id": sorted_asset_id,
#             "mount": "ecephys_sorted"
#         },
#         {
#             "id": eye_asset_id,
#             "mount": "eye_tracking"
#         },
#         {
#             "id": ccf_asset_id,
#             "mount": "ccf"
#         },
#         {
#             "id": stim_templates_asset_id,
#             "mount": "stim_templates"
#         }
#     ],
#     "pipeline_id": pipeline_id,
#     "parameters": ["hdf5"]
# }
# run_response = co_client.run_capsule(run_asset).json()
