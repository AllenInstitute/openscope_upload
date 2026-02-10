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
    # pipeline_id = "daef0b82-2f12-4122-964d-efa5f608ad69"
    pipeline_id = "e16bc028-30b1-4aa2-89f9-a2cb27aaf844"

    datasets_ids = (
        ("830794", "7b676266-5bce-488f-a9f6-d267445906b7"),
        ("830794", "6b8ed5e2-30ce-404b-b69b-863c7d7d1998"),
        ("830794", "efae5289-eb8e-4024-9c35-0c58ab34c54d"),
        ("830794", "10c17d4c-6c93-43c0-9daf-1888e0599b09"),
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
        run_params = RunParams(capsule_id=pipeline_id,version=4,data_assets=data_assets,processes=[])

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
