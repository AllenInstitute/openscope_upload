from aind_codeocean_pipeline_monitor.models import (
     CaptureSettings,
     PipelineMonitorSettings,
)
from codeocean import CodeOcean
from codeocean.computation import (
     DataAssetsRunParam,
     RunParams,
     PipelineProcessParams,
     NamedRunParam
)
import os, argparse
import pandas as pd
from pathlib import Path

# Users will need to be made owners of this capsule. Attach your token and
# specify the code ocean domain.

domain = "https://codeocean.allenneuraldynamics.org"
token = os.getenv("CODEOCEAN_TOKEN")
# monitor_pipeline_capsule_id = "567b5b98-8d41-413b-9375-9ca610ca2fd3"
# monitor_pipeline_capsule_id = "fbdddd96-6d2a-4e40-88c8-45f3f84cfaf3"
# carter's monitor capsule which passes along credentials
monitor_pipeline_capsule_id = "7a6dd345-e3fe-4f86-b2c3-c42c0d673c3c"
client = CodeOcean(domain=domain, token=token)

# dandiset_id = "001568"
dandiset_id = ""

# pipeline_id="bd9bc5d9-d691-4a8f-8820-b4562c74237b",
# pipeline_id = "111e7ef9-bc20-4faa-9169-d8e9f373e752"
pipeline_id = "ab7a0799-f6d5-4bc3-9a9d-06e0ba2ea721"

# Construct the parameters for the pipeline you wish to monitor.
# The capture_settings are optional. If set to None, then the results will not
# be captured.
def get_monitor_settings(pipeline_id, sub_id, data_assets, dandiset_id):
    settings = PipelineMonitorSettings(
        run_params=RunParams(
            pipeline_id=pipeline_id,
            data_assets=data_assets,
            processes=[
                PipelineProcessParams(name="capsule_carters_aind_dandi_upload_16",named_parameters=[NamedRunParam(param_name='dandiset_id',value=dandiset_id)]),
                PipelineProcessParams(name="capsule_nwb_zarr_hdf_5_conversion_12",named_parameters=[NamedRunParam(param_name='input_nwb_dir',value='ecephys_sorted/nwb')]),
                PipelineProcessParams(name="capsule_aind_running_speed_nwb_13",named_parameters=[NamedRunParam(param_name='use_input_nwb',value='True')]),
                PipelineProcessParams(name="capsule_aind_eye_tracking_nwb_14",named_parameters=[NamedRunParam(param_name='use_input_nwb',value='True')]),
            ]
        ),
        capture_settings={
            "process_name_suffix": "nwb",
            "tags": ["derived", "ecephys", "neuropixels", "nwb", sub_id],
            "custom_metadata": {
                "data level": "derived",
                "experiment type": "ecephys",
                "subject id": sub_id
            },
            "permissions": {"everyone": "viewer"},
            "target": {"aws": {"bucket": "aind-open-data"}}
        },
    )
    return settings



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



parser = argparse.ArgumentParser()
parser.add_argument("--asset_mounts_csv", type=str, required=True)
parser.add_argument("--dandiset_id", type=str, required=True)
parser.add_argument("--exclude_n_cols", type=int, default=1)
args = parser.parse_args()

asset_mounts = pd.read_csv(Path(args.asset_mounts_csv))
dandiset_id = args.dandiset_id
enc = args.exclude_n_cols
assert type(dandiset_id) is str and len(dandiset_id) == 6

jobs = []
for index, row in asset_mounts.iterrows():
    data_assets = [DataAssetsRunParam(id=asset_id,mount=mount) for mount, asset_id in zip(asset_mounts.columns[enc:], row[enc:])]
    print(data_assets)
    sub_id = row["session"].split("_")[1]
    trigger_monitor(jobs, pipeline_id, sub_id, data_assets, dandiset_id)

