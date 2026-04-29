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
import json


# ============================================================================
# GLOBAL PARAMETERS CONFIGURATION
# ============================================================================
# Define your parameters here as needed for your capsule/pipeline

# For direct named parameters to capsule (not in a pipeline):
# NAMED_PARAMETERS = [
#     NamedRunParam(param_name="processor_full_name", value="carter peene"),
# ]
NAMED_PARAMETERS = None

# For ordered parameters (uncomment and use if needed):
# PARAMETERS = ["param1", "param2"]
PARAMETERS = None

# For pipeline process parameters:
PROCESSES = [
    PipelineProcessParams(
        name="capsule_aind_pipeline_processing_metadata_aggregator_7",
        named_parameters=[
            NamedRunParam(param_name="processor_full_name", value="carter peene"),
        ]
    ),
]

# ============================================================================


def main():
    co_api_token = os.getenv("CODEOCEAN_TOKEN")
    co_domain = "https://codeocean.allenneuraldynamics.org"
    client = CodeOcean(domain=co_domain, token=co_api_token)
    print(client.token)

    parser = argparse.ArgumentParser()
    parser.add_argument("capsule_id", type=str,
                        help="CodeOcean capsule ID to run")
    parser.add_argument("--asset_mounts_csv", type=str, required=False, default=None,
                        help="Path to CSV file containing asset mounts")
    parser.add_argument("--asset_mount", nargs='+', default=None,
                        help="Alternating mount names and asset IDs (e.g., mount1 asset_id1 mount2 asset_id2)")
    parser.add_argument("--exclude_n_cols", type=int, default=0,
                        help="Number of columns to exclude from the beginning of CSV (default: 0)")
    parser.add_argument("--version", type=int, default=None,
                        help="Capsule version to run (default: None, uses latest)")
    # parser.add_argument("--parameters", type=str, default=None,
    #                     help="Parameters as JSON string or path to JSON file (list for ordered, object for named)")
    parser.add_argument("--additional_mount", nargs=2, metavar=("mount_name","asset_id"), default=None)
    args = parser.parse_args()

    capsule_id = args.capsule_id
    exclude_n_cols = args.exclude_n_cols
    version = args.version

    # Validate that at least one input method is provided
    if not args.asset_mounts_csv and not args.asset_mount:
        parser.error("Must provide either --asset_mounts_csv or --asset_mount")
    
    # Validate asset_mount has even number of elements
    if args.asset_mount and len(args.asset_mount) % 2 != 0:
        parser.error("--asset_mount must have an even number of arguments (alternating mount names and asset IDs)")

    # Process asset_mount into a single-row DataFrame if provided
    asset_mount_df = None
    if args.asset_mount:
        mount_names = []
        asset_ids = []
        for i in range(0, len(args.asset_mount), 2):
            mount_names.append(args.asset_mount[i])
            asset_ids.append(args.asset_mount[i + 1])
        asset_mount_df = pd.DataFrame([asset_ids], columns=mount_names)

    # Load asset mounts from CSV if provided
    if args.asset_mounts_csv:
        asset_mounts = pd.read_csv(Path(args.asset_mounts_csv))
        
        # If both CSV and asset_mount are provided, validate columns match and combine
        if asset_mount_df is not None:
            # Skip excluded columns when comparing
            csv_mount_columns = list(asset_mounts.columns[exclude_n_cols:])
            asset_mount_columns = list(asset_mount_df.columns)
            
            if csv_mount_columns != asset_mount_columns:
                parser.error(
                    f"Column mismatch: CSV has mounts {csv_mount_columns}, "
                    f"but --asset_mount provides {asset_mount_columns}"
                )
            
            # Add excluded columns to asset_mount_df to match CSV structure
            for i in range(exclude_n_cols - 1, -1, -1):
                asset_mount_df.insert(0, asset_mounts.columns[i], 'asset_mount_row')
            # Append the asset_mount row to CSV
            asset_mounts = pd.concat([asset_mounts, asset_mount_df], ignore_index=True)
    else:
        # Only asset_mount provided, no excluded columns needed
        if asset_mount_df is not None:
            asset_mounts = asset_mount_df
        else:
            asset_mounts = None
    
    # Always use column names as mount names (CSV columns or --asset_mount argument names)
    use_column_names_as_mounts = True
    
    if args.additional_mount:
        additional_mount = tuple(args.additional_mount)
        assert len(additional_mount) == 2
    else:
        additional_mount = None

    data = []
    
    # Process all rows (from CSV, asset_mount, or both)
    if asset_mounts is not None:
        for index, row in asset_mounts.iterrows():
            data_assets = []
            
            # Iterate through mount columns (skip excluded columns)
            for mount, asset_id in zip(asset_mounts.columns[exclude_n_cols:], row[exclude_n_cols:]):
                if use_column_names_as_mounts:
                    # Use column name as mount name (for --asset_mount mode)
                    data_assets.append(DataAssetsRunParam(id=asset_id, mount=mount))
                else:
                    # Use default mount from CodeOcean (for CSV-only mode)
                    data_asset = client.data_assets.get_data_asset(data_asset_id=asset_id)
                    default_mount = data_asset.mount
                    print(default_mount)
                    data_assets.append(DataAssetsRunParam(id=asset_id, mount=default_mount))
            
            if additional_mount:
                data_assets.append(DataAssetsRunParam(id=additional_mount[1], mount=additional_mount[0]))

            print(data_assets)
            # Build RunParams with all optional fields
            run_params_kwargs = {
                'capsule_id': capsule_id,
                'data_assets': data_assets,
            }
            if version is not None:
                run_params_kwargs['version'] = version
            if PARAMETERS is not None:
                run_params_kwargs['parameters'] = PARAMETERS
            if NAMED_PARAMETERS is not None:
                run_params_kwargs['named_parameters'] = NAMED_PARAMETERS
            if PROCESSES is not None:
                run_params_kwargs['processes'] = PROCESSES
            
            # Use first column as identifier if exclude_n_cols > 0, otherwise use index
            identifier = row[0] if exclude_n_cols > 0 else f"row_{index}"
            print(f"\n\n\nRunning dataset {identifier}")
            run_params = RunParams(**run_params_kwargs)
            print("="*64, "\n", run_params, "\n","="*64)
            computation = client.computations.run_capsule(run_params)
            run_response = computation
            print(f"Run response: {run_response}")
            data.append(run_response)
            time.sleep(5)


main()