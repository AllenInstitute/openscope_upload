from codeocean.computation import RunParams, DataAssetsRunParam, PipelineProcessParams, NamedRunParam
from codeocean import CodeOcean
from aind_codeocean_api.codeocean import CodeOceanClient

import time
from datetime import datetime as dt
from datetime import timezone as tz
import argparse
import pandas as pd
from pathlib import Path
import os


def main():
    # id for NWB test packaging pipeline
    # pipeline_id = "b8928b3e-ef9d-4fef-9a78-4204cd55b0d9"
    # pipeline_id = "060a47bf-2ce4-4664-a2c7-922f31afaf3b"
    # pipeline_id = "1e510cd0-0b0c-4891-ba08-ce12db9680a3"
    # pipeline_id = "55957064-6fb9-4369-9b49-410393b9fdfe"
    # P3 Pipeline
    # pipeline_id = "ab7a0799-f6d5-4bc3-9a9d-06e0ba2ea721"
    # Psycode pipeline
    pipeline_id = "55957064-6fb9-4369-9b49-410393b9fdfe"

    co_api_token = os.getenv("CODEOCEAN_TOKEN")
    co_domain = "https://codeocean.allenneuraldynamics.org"
    client = CodeOcean(domain=co_domain, token=co_api_token)
    print(client.token)

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_mounts_csv", type=str, required=True)
    parser.add_argument("--dandiset_id", type=str, required=True)
    parser.add_argument("--additional_mount", nargs=2, metavar=("mount_name","asset_id"), default=None)
    args = parser.parse_args()

    asset_mounts = pd.read_csv(Path(args.asset_mounts_csv))
    dandiset_id = args.dandiset_id
    assert type(dandiset_id) is str and len(dandiset_id) == 6
    if args.additional_mount:
        additional_mount = tuple(args.additional_mount)
        assert len(additional_mount) == 2
    else:
        additional_mount = None

    data = []
    for index, row in asset_mounts.iterrows():
        data_assets = [DataAssetsRunParam(id=asset_id,mount=mount) for mount, asset_id in zip(asset_mounts.columns, row)]
        print(data_assets)
        if additional_mount:
            data_assets.append(DataAssetsRunParam(id=additional_mount[1],mount=additional_mount[0]))

        print(data_assets)
        processes=[
            PipelineProcessParams(name="capsule_carters_aind_dandi_upload_17",named_parameters=[NamedRunParam(param_name='dandiset_id',value=dandiset_id)]),
            PipelineProcessParams(name="capsule_nwb_zarr_hdf_5_conversion_12",named_parameters=[NamedRunParam(param_name='input_nwb_dir',value='ecephys_sorted/nwb')]),
            PipelineProcessParams(name="capsule_aind_ccf_nwb_15",named_parameters=[NamedRunParam(param_name='convert_ibl_bregma_to_ccf',value='true'),
                                                                                #    NamedRunParam(param_name='skip_ccf',value='true')
            ]),
            PipelineProcessParams(name="capsule_nwb_packaging_stimulus_6",named_parameters=[NamedRunParam(param_name='input_csv_dir',value='stim_tables')]),
            PipelineProcessParams(name="capsule_nwb_running_packaging_running_speed_7",named_parameters=[NamedRunParam(param_name='use_input_nwb',value='True')]),
            PipelineProcessParams(name="capsule_nwb_packaging_eye_tracking_fixed_8",named_parameters=[NamedRunParam(param_name='use_input_nwb',value='True')]),
        ]
        run_params = RunParams(capsule_id=pipeline_id,data_assets=data_assets,processes=processes)
        print("="*64,run_params,"="*64)
        print(f"\n\n\nRunning dataset {row[0]}")
        computation = client.computations.run_capsule(run_params)
        run_response = computation
        print(f"Run response: {run_response}")
        data.append(run_response)
        time.sleep(5)


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
