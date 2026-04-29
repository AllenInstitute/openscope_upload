from codeocean.computation import RunParams, DataAssetsRunParam
from codeocean import CodeOcean
from aind_codeocean_pipeline_monitor.models import CaptureSettings, PipelineMonitorSettings

import argparse
import csv
import time
import os
from pathlib import Path


# Spike sorting pipeline
PIPELINE_ID = "e16bc028-30b1-4aa2-89f9-a2cb27aaf844"

# Monitor capsule (runs pipeline with capture settings under user credentials)
MONITOR_CAPSULE_ID = "a8a0f763-6d31-420f-95c6-f247de8ac8ea"

CO_API_TOKEN = os.getenv("CODEOCEAN_TOKEN")
CO_DOMAIN = "https://codeocean.allenneuraldynamics.org"
CLIENT = CodeOcean(domain=CO_DOMAIN, token=CO_API_TOKEN)


def get_monitor_settings(session_asset_id: str) -> PipelineMonitorSettings:
    return PipelineMonitorSettings(
        run_params=RunParams(
            capsule_id=PIPELINE_ID,
            version=6,
            data_assets=[DataAssetsRunParam(id=session_asset_id, mount="ecephys")],
            processes=[],
        ),
        capture_settings=CaptureSettings(
            process_name_suffix="sorted",
            tags=["derived", "ecephys_sorted"],
            permissions={"everyone": "viewer"},
            target={"aws": {"bucket": "aind-open-data"}},
            custom_metadata={"data level": "derived"},
        ),
    )


def parse_session_assets_from_csv(csv_path):
    """
    Parse session asset IDs from a CSV file.

    Args:
        csv_path: Path to CSV file containing session assets

    Returns:
        List of session asset ID strings
    """
    session_asset_ids = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        if 'session_assets' in fieldnames:
            column_name = 'session_assets'
        elif 'ecephys_session' in fieldnames:
            column_name = 'ecephys_session'
        else:
            raise ValueError(
                f"CSV must contain either 'session_assets' or 'ecephys_session' column. "
                f"Found columns: {fieldnames}"
            )

        for row in reader:
            asset_id = row.get(column_name, '').strip()
            if asset_id:
                session_asset_ids.append(asset_id)

    return session_asset_ids


def dispatch_kilosort_pipeline(session_asset_ids):
    """
    Dispatch Kilosort pipeline runs via the monitor capsule for given session assets.

    Args:
        session_asset_ids: List of session asset ID strings

    Returns:
        List of monitor run response objects
    """
    print(f"Processing {len(session_asset_ids)} session(s)")

    jobs = []
    for session_asset_id in session_asset_ids:
        settings = get_monitor_settings(session_asset_id)
        pipeline_params = settings.model_dump_json(exclude_none=True)
        print(pipeline_params)

        monitor_params = RunParams(
            capsule_id=MONITOR_CAPSULE_ID,
            parameters=[pipeline_params],
        )

        print(f"\n\nDispatching monitor for asset {session_asset_id}")
        computation = CLIENT.computations.run_capsule(monitor_params)
        print(f"Monitor job id: {computation.id}")
        jobs.append(computation)
        time.sleep(5)

    return jobs


def main():
    """Parse command line arguments and dispatch Kilosort pipeline runs."""
    parser = argparse.ArgumentParser(
        description='Dispatch Kilosort pipeline runs on CodeOcean',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using command line session asset IDs
  python dispatch_kilosort.py 7b676266-5bce-488f-a9f6-d267445906b7 6b8ed5e2-30ce-404b-b69b-863c7d7d1998
  
  # Using a CSV file
  python dispatch_kilosort.py --session_assets_csv ./data/p3_ephys_latest_assets.csv
        """
    )
    
    parser.add_argument(
        'session_asset_ids',
        nargs='*',
        help='Session asset IDs to process (space-separated)'
    )
    parser.add_argument(
        '--session_assets_csv',
        type=Path,
        help='Path to CSV file containing session assets (must have "session_assets" or "ecephys_session" column)'
    )
    
    args = parser.parse_args()
    
    # Ensure exactly one input method is provided
    if args.session_assets_csv and args.session_asset_ids:
        parser.error("Cannot specify both session_asset_ids and --session_assets_csv")
    
    # Get session asset IDs from either command line or CSV
    if args.session_assets_csv:
        print(f"Reading session assets from CSV: {args.session_assets_csv}")
        session_asset_ids = parse_session_assets_from_csv(args.session_assets_csv)
    elif args.session_asset_ids:
        session_asset_ids = args.session_asset_ids
    else:
        parser.error("Must provide either session_asset_ids or --session_assets_csv")
    
    if not session_asset_ids:
        print("Error: No session asset IDs provided")
        return
    
    # Dispatch the pipeline runs
    dispatch_kilosort_pipeline(session_asset_ids)


if __name__ == "__main__":
    main()

