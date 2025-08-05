from aind_codeocean_api.codeocean import CodeOceanClient
from codeocean.computation import RunParams, DataAssetsRunParam, PipelineProcessParams, NamedRunParam
from codeocean import CodeOcean

import time
from datetime import datetime as dt
from datetime import timezone as tz
import json
import os


def main():
    # id for NWB Kilosort pipeline
    pipeline_id = "daef0b82-2f12-4122-964d-efa5f608ad69"

    datasets_ids = (
        ("752311", "001c3101-c190-4b0d-a0f7-653dc6e2d8ab"),
        ("752312", "80686a5f-b0c7-4f0a-95e6-adac5b4c6cfa"),
        ("752312", "84d79052-83c1-43bd-84f4-af7e4ba6d4cd"),
        ("753316", "1fc4ee81-feef-430e-aae3-93479d9d908f"),
        ("760322", "9bd673fd-fd2d-4a2e-80f9-10bfe83c002b"),
        ("760322", "21f268dd-44f3-47ab-97ed-dd869b8ac373"),
        ("767925", "a4b78c15-4c43-4b5d-bba6-8545672f3b34"),
        ("752309", "7979e980-4c7b-4cc4-b26c-1376ecc8fcd7"),
        ("752309", "4d9d1787-c7d8-41e9-885e-ff6591483516"),
        ("760324", "92c69c49-d180-4bad-9868-85b83c72094e"),
        ("760324", "fb50c156-a5bc-44be-8c53-90a18ac6832b"),
        ("767926", "ef4133b1-919d-4408-88d1-7cfa3a83c0a7"),
        ("767926", "832fc1be-2f39-4207-a501-9344bbab0ec4")
    )
    co_api_token = os.getenv("CODEOCEAN_TOKEN")
    co_domain = "https://codeocean.allenneuraldynamics.org"
    client = CodeOcean(domain=co_domain, token=co_api_token)
    print(client.token)

    data = []
    for identifier, session_asset_id in datasets_ids:
        data_assets=[
            DataAssetsRunParam(id=session_asset_id,mount="ecephys"),
        ]   
        run_params = RunParams(capsule_id=pipeline_id,data_assets=data_assets,processes=[])

        print(f"\n\n\nRunning dataset {identifier}")
        computation = client.computations.run_capsule(run_params)
        run_response = computation
        print(f"Run response: {run_response}")
        data.append(run_response)
    #     proc_time = dt.now(tz.utc).strftime("%Y-%m-%d_%H-%M-%S")
        time.sleep(5)

    timestamp = dt.now().strftime("%Y%m%dT%H%M%S")
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
