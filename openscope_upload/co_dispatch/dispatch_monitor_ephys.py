from aind_codeocean_pipeline_monitor.models import (
     CaptureSettings,
     PipelineMonitorSettings,
)
from codeocean import CodeOcean
from codeocean.computation import (
     DataAssetsRunParam,
     RunParams,
)
import os

# Users will need to be made owners of this capsule. Attach your token and
# specify the code ocean domain.

domain = "https://codeocean.allenneuraldynamics.org"
token = os.getenv("CODEOCEAN_TOKEN")
monitor_pipeline_capsule_id = "567b5b98-8d41-413b-9375-9ca610ca2fd3"
client = CodeOcean(domain=domain, token=token)

# Construct the parameters for the pipeline you wish to monitor.
# The capture_settings are optional. If set to None, then the results will not
# be captured.

stim_templates_asset_id = "7f8cf734-7640-4d3c-a50a-20d1f8f299e4"
mount_names = ["ecephys_session","ecephys_sorted","eye_tracking","ccf"]
datasets_ids = {
    "747824": ["0fd11c2a-7ea2-4f44-bad3-35ca41e9301e","f8b7b4c9-a7b6-4420-9fa1-04e43de78ad9","3fade0c2-a569-4e2c-9e90-985b6b69e9cd","a3ddffb9-75d5-46da-ab66-684db9bb5f49"],
}

for sub_id, data_ids in datasets_ids.items():
    input_assets = [DataAssetsRunParam(id=id,mount=mountname) for id, mountname in zip(data_ids,mount_names) ]
    input_assets.append(DataAssetsRunParam(id=stim_templates_asset_id, mount="stim_templates"))

    settings = PipelineMonitorSettings(
        run_params=RunParams(
            pipeline_id="bd9bc5d9-d691-4a8f-8820-b4562c74237b",
            data_assets=input_assets
        ),
        capture_settings=CaptureSettings(
            process_name_suffix="nwb",
            tags=["derived", "test", "ecephys", sub_id],
            custom_metadata = {
            "data level": "derived",
            "experiment type": "ecephys",
            "subject id": sub_id
            }
        )
    )

    # Send the pipeline run parameters to the pipeline monitor capsule to execute
    pipeline_params = settings.model_dump_json()
    monitor_params = RunParams(capsule_id=monitor_pipeline_capsule_id, parameters=[pipeline_params])
    monitor_run_comp = client.computations.run_capsule(monitor_params)
    print(monitor_run_comp)
