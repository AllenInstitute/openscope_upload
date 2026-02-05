import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass
import json

from aind_codeocean_pipeline_monitor.models import (CaptureSettings,
                                                    PipelineMonitorSettings)
from aind_data_access_api.document_db import MetadataDbClient
from codeocean import CodeOcean
from codeocean.computation import (ComputationState, DataAssetsRunParam,
                                   RunParams, NamedRunParam)
from dataclasses_json import dataclass_json

logging.basicConfig(
    filename="batch.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Set environment variables
API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"
# domain = os.getenv("CODEOCEAN_DOMAIN")
domain = "https://codeocean.allenneuraldynamics.org"
token = os.getenv("CODEOCEAN_TOKEN")

# monitor_pipeline_capsule_id = os.getenv("CO_MONITOR_PIPELINE")
monitor_pipeline_capsule_id = "7a6dd345-e3fe-4f86-b2c3-c42c0d673c3c"
client = CodeOcean(domain=domain, token=token)


@dataclass_json
@dataclass(frozen=True)
class JobSettings:
    pipeline_id: str
    asset_id: str
    subject_id: str


def get_monitor_settings(
    job_settings: [JobSettings | dict],
) -> PipelineMonitorSettings:
    """Get the pipeline monitor settings.

    Parameters
    ----------
    job_setting: [JobSettings | dict]
        Settings defining the job to run

    Returns
    -------
    PipelineMonitorSettings
        The pipeline monitor settings.
    """
    if isinstance(job_settings, dict):
        job_settings = JobSettings.from_dict(job_settings)
    return PipelineMonitorSettings(
        run_params=RunParams(
            capsule_id=job_settings.pipeline_id,
            version=11,
            data_assets=[
                DataAssetsRunParam(
                    id=job_settings.asset_id,
                    mount="ophys_mount",
                )
            ],
            named_parameters=[
                NamedRunParam(
                    param_name="acquisition_data_type",
                    value="multiplane",
                ),
                NamedRunParam(
                    param_name="pipeline_version",
                    value="11.0",
                )
            ],
        ),
        capture_settings=CaptureSettings(
            process_name_suffix="processed",
            tags=["derived", "multiplane-ophys", job_settings.subject_id],
            custom_metadata={
                "data level": "derived",
                "experiment type": "multiplane-ophys",
                "subject id": job_settings.subject_id,
            },
        ),
    )
 

def get_asset_id(docdb_api_client, asset_name) -> str:
    """Get the asset ID from the data access api

    Parameters
    ----------
    docdb_api_client : MetadataDbClient
        The data access api client
    asset_name : str
        The asset name

    Returns
    -------
    str
        The asset ID
    """
    query = {"name": asset_name}
    projection = {"external_links": 1}
    response = docdb_api_client.retrieve_docdb_records(
        filter_query=query, projection=projection
    )
    external_links = response[0].get("external_links", None)
    if type(external_links) is str:
        external_links = json.loads(external_links)
        external_links = external_links.get("Code Ocean", None)
    if type(external_links) is list and len(external_links) > 1:
        external_links = external_links[0]
        external_links = external_links.get("Code Ocean", None)
    if type(external_links) is dict:
        try:
            external_links = external_links.get("Code Ocean", None)[0]
        except IndexError:
            external_links = "None"
    if type(external_links) is list:
        try:
            external_links = external_links[0]
        except IndexError:
            external_links = "None"
    return external_links


def run():
    """Example usage below
    > python batch_process.py --csv_file <path_to_csv> --pipeline_id <pipeline_id> --max-jobs 10 --sleep 600 

    csv files must contain the subject_id or asset_id (or asset_name)
    if using the asset_name, add flag --asset_name True
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--pipeline_id", type=str, required=True)
    parser.add_argument("--max-jobs", type=int, default=10)
    parser.add_argument("--sleep", type=int, default=600)
    parser.add_argument("--asset_name", type=bool, default=False)
    args = parser.parse_args()
    rows = []
    asset_name = args.asset_name

    with open(
        args.csv_file,
        "r",
    ) as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            rows.append(row)
    header = rows[0]
    header = [i.strip() for i in header]
    data = rows[1:]
    jobs = []
    docdb_api_client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )
    if asset_name:
        asset_id_index = header.index("asset_name")
    else:
        asset_id_index = header.index("asset_id")
    subject_id_index = header.index("subject_id")
    if not asset_id_index:
        logging.warning("No asset_id column found in CSV file, using asset_name")
        logging.info("Using asset_name")
    for row in data:
        if not row:
            continue
        if asset_name:
            data_asset_id = get_asset_id(docdb_api_client, row[asset_id_index])
        else:
            data_asset_id = row[asset_id_index]

        job_settings = JobSettings(
            pipeline_id=args.pipeline_id,
            asset_id=data_asset_id,
            subject_id=row[subject_id_index],
        )
        settings = get_monitor_settings(job_settings)
        pipeline_params = settings.model_dump_json(exclude_none=True)
        monitor_params = RunParams(
            capsule_id=monitor_pipeline_capsule_id, parameters=[pipeline_params]
        )
        monitor_run_comp = client.computations.run_capsule(monitor_params)
        job_id = monitor_run_comp.id
        jobs.append(monitor_run_comp)
        logging.info(f"Job {job_id} started")
        print(f"Jobs started: {len(jobs)}")
        while len(jobs) >= args.max_jobs:
            for job in jobs:
                if client.computations.get_computation(job.id).state in [
                    ComputationState.Completed,
                    ComputationState.Failed,
                ]:
                    if job.state == ComputationState.Failed:
                        logging.error(f"Job {job.id} failed")
                    jobs.remove(job)
                    break
            time.sleep(args.sleep)


if __name__ == "__main__":
    run()