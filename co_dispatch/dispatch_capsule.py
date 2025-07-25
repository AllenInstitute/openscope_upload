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
    capsule_id = "7935d378-ce0e-4129-8774-81e9c8573bc2"

    co_api_token = os.getenv("CODEOCEAN_TOKEN")
    co_domain = "https://codeocean.allenneuraldynamics.org"
    client = CodeOcean(domain=co_domain, token=co_api_token)
    print(client.token)

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_mounts_csv", type=str, required=True)
    parser.add_argument("--additional_mount", nargs=2, metavar=("mount_name","asset_id"), default=None)
    args = parser.parse_args()

    asset_mounts = pd.read_csv(Path(args.asset_mounts_csv))
    if args.additional_mount:
        additional_mount = tuple(args.additional_mount)
        assert len(additional_mount) == 2
    else:
        additional_mount = None

    data = []
    for index, row in asset_mounts.iterrows():
        data_assets=[]
        for mount, asset_id in zip(asset_mounts.columns[1:], row[1:]):
            data_asset = client.data_assets.get_data_asset(data_asset_id=asset_id)
            default_mount = data_asset.mount
            print(default_mount)
            data_assets.append(DataAssetsRunParam(id=asset_id,mount=default_mount))
        if additional_mount:
            data_assets.append(DataAssetsRunParam(id=additional_mount[1],mount=additional_mount[0]))

        print(data_assets)
        run_params = RunParams(capsule_id=capsule_id,data_assets=data_assets)
        print("="*64,run_params,"="*64)
        print(f"\n\n\nRunning dataset {row[0]}")
        computation = client.computations.run_capsule(run_params)
        run_response = computation
        print(f"Run response: {run_response}")
        data.append(run_response)
        time.sleep(5)


main()