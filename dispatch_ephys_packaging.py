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
    pipeline_id = "b8928b3e-ef9d-4fef-9a78-4204cd55b0d9"

    # for loop project
    # stim_templates_asset_id = "7f8cf734-7640-4d3c-a50a-20d1f8f299e4"

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
        for asset_id, mount in zip(row, asset_mounts.columns):
            print(asset_id,mount)
        data_assets = [DataAssetsRunParam(id=asset_id,mount=mount) for mount, asset_id in zip(asset_mounts.columns, row)]
        if additional_mount:
            data_assets.append(DataAssetsRunParam(id=additional_mount[1],mount=additional_mount[0]))

        processes=[
            PipelineProcessParams(name="capsule_nwb_packaging_subject_capsule_1",parameters=["hdf5",""]),
            PipelineProcessParams(name="capsule_aind_stim_templates_nwb_10",named_parameters=[NamedRunParam(param_name='template_name_suffix',value='')]),
            PipelineProcessParams(name="capsule_carters_aind_dandi_upload_13",named_parameters=[NamedRunParam(param_name='dandiset_id',value=dandiset_id)]),
            PipelineProcessParams(name="capsule_nwb_zarr_hdf_5_conversion_14",named_parameters=[NamedRunParam(param_name='input_nwb_dir',value='ecephys_sorted/nwb')]),            
        ]
        run_params = RunParams(capsule_id=pipeline_id,data_assets=data_assets,processes=processes)
        print(f"\n\n\nRunning dataset {row[0]}")
        computation = client.computations.run_capsule(run_params)
        run_response = computation
        print(f"Run response: {run_response}")
        data.append(run_response)

        # data_assets=[
        #     DataAssetsRunParam(id=session_asset_id,mount="ecephys_session"),
        #     DataAssetsRunParam(id=sorted_asset_id,mount="ecephys_sorted"),
        #     DataAssetsRunParam(id=eye_asset_id,mount="eye_tracking"),
        #     # DataAssetsRunParam(id=ccf_asset_id,mount="ccf"),
        #     # DataAssetsRunParam(id=stim_templates_asset_id,mount="stim_templates"),
        # ]   
        # processes=[
        #     # PipelineProcessParams(name="capsule_nwb_packaging_subject_capsule_1",parameters=["hdf5",""]),
        #     # PipelineProcessParams(name="capsule_aind_stim_templates_nwb_10",named_parameters=[NamedRunParam(param_name='template_name_suffix',value='')]),
        #     PipelineProcessParams(name="capsule_carters_aind_dandi_upload_13",named_parameters=[NamedRunParam(param_name='dandiset_id',value='001417')]),
        #     PipelineProcessParams(name="capsule_nwb_zarr_hdf_5_conversion_14",named_parameters=[NamedRunParam(param_name='input_nwb_dir',value='ecephys_sorted/nwb')]),            
        # ]
        # run_params = RunParams(capsule_id=pipeline_id,data_assets=data_assets,processes=processes)

        # print(f"\n\n\nRunning dataset {session_asset_id}")
        # computation = client.computations.run_capsule(run_params)
        # run_response = computation
        # print(f"Run response: {run_response}")
        # data.append(run_response)
    #     proc_time = dt.now(tz.utc).strftime("%Y-%m-%d_%H-%M-%S")
        # time.sleep(5)
    #     try:
    #         data_asset = co_client.get_data_asset(session_asset_id).json()["name"]
    #         processed_asset_name = data_asset + "_nwb"
    #         run_response["session"] = identifier
    #         run_response["asset_name_processed"] = processed_asset_name
    #         run_response["session_asset_id"] = session_asset_id
    #         run_response["sorted_asset_id"] = sorted_asset_id
    #         run_response["eye_asset_id"] = eye_asset_id
    #         run_response["ccf_asset_id"] = ccf_asset_id
    #         run_response["stim_templates_asset_id"] = stim_templates_asset_id
    #         data.append(run_response)
    #         time.sleep(30)
    #     except KeyError:
    #         print(f"Data asset {session_asset_id} not found")
    # timestamp = dt.now().strftime("%Y%m%dT%H%M%S")
    # with open(f"run_results_{timestamp}.json", "w") as fp:
    #     json.dump(data, fp, indent=4)


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
