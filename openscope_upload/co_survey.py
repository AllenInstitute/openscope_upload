
from aind_data_access_api.document_db import MetadataDbClient
import argparse
from codeocean.data_asset import DataAssetSearchParams
from codeocean import CodeOcean
import os
import aind_session
import pandas as pd
from pathlib import Path
from datetime import datetime

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"

THIS_FILE_PATH = Path(__file__).resolve()

client = CodeOcean(domain="https://allenneuraldynamics.org", token=os.getenv("CODEOCEAN_TOKEN"))

# def get_asset_id(docdb_api_client, asset_name) -> str:
#     """Get the asset ID from the data access api
#     Parameters
#     ----------
#     docdb_api_client : MetadataDbClient
#         The data access api client
#     asset_name : str
#         The asset name
#     Returns
#     -------
#     str
#         The asset ID
#     """
#     query = {"name": asset_name}
#     projection = {"external_links": 1}
#     response = docdb_api_client.retrieve_docdb_records(
#         filter_query=query, projection=projection
#     )
#     print(response)
#     external_links = response[0].get("external_links", None)
#     if type(external_links) is str:
#         external_links = json.loads(external_links)
#         external_links = external_links.get("Code Ocean", None)
#     if type(external_links) is list and len(external_links) > 1:
#         external_links = external_links[0]
#         external_links = external_links.get("Code Ocean", None)
#     if type(external_links) is dict:
#         try:
#             external_links = external_links.get("Code Ocean", None)[0]
#         except IndexError:
#             external_links = "None"
#     if type(external_links) is list:
#         try:
#             external_links = external_links[0]
#         except IndexError:
#             external_links = "None"
#     return external_links


def find_co_assets_by_mouse_ids(mouse_ids: list[int]) -> dict[int, list[str]]:
    """
    Find all Code Ocean asset names associated with a list of mouse IDs.
    Parameters
    ----------
    docdb_api_client : MetadataDbClient
        The data access API client.
    mouse_ids : list of int
        List of mouse IDs to search for.
    Returns
    -------
    dict
        Dictionary mapping each mouse_id to a list of asset names.
    """
    mouse_assets = {}
    for mouse_id in mouse_ids:
        # docDB query
        # print(mouse_id)
        # query = {"name": {"$regex": f".*{mouse_id}.*"}}
        # projection = {"name": 1}
        # response = docdb_api_client.retrieve_docdb_records(
        #     filter_query=query, projection=projection
        # )
        # print(response)
        # mouse_assets[mouse_id] = [r["name"] for r in response if "name" in r]

        # codeocean query
        # data_asset_params = DataAssetSearchParams(
        #     offset=0,
        #     limit=100,
        #     sort_order="desc",
        #     sort_field="name",
        #     query=mouse_id
        # )
        # data_assets = client.data_assets.search_data_assets(data_asset_params)
        mouse_assets[mouse_id] = [asset for asset in aind_session.utils.get_subject_data_assets(mouse_id)]
        # mouse_assets[mouse_id] = aind_session.utils.get_data_assets(f'multiplane-ophys_{mouse_id}')

    return mouse_assets


def get_session_asset_row(mouse_id, session_id, session_datetime_str, assets, take_newest=False):
    session_assets = []
    processed_assets = []
    nwb_assets = []
    has_non_output = lambda asset: asset.files > 1
    for asset in assets:
        if mouse_id not in asset.name:
            continue
        if f"{mouse_id}_{session_datetime_str}" not in asset.name:
            continue

        try:
            if 'nwb' in asset.name.lower() and has_non_output(asset):
                nwb_assets.append(asset)
            elif 'processed' in asset.name.lower() and has_non_output(asset):
                processed_assets.append(asset)
            elif has_non_output(asset):
                session_assets.append(asset)
        except TypeError:
            continue

    sort_by_create = lambda asset: asset.created
    session_asset_ids = [asset.id for asset in sorted(session_assets, key=sort_by_create)]
    processed_asset_ids = [asset.id for asset in sorted(processed_assets, key=sort_by_create)]
    nwb_asset_ids = [asset.id for asset in sorted(nwb_assets, key=sort_by_create)]

    session_assets = {
        "mouse_id": mouse_id,
        "session_id": session_id,
        "session_datetime": session_datetime_str,
        "session_assets": str(tuple(session_assets)),
        "processed_assets": str(tuple(processed_assets)),
        "nwb_assets": str(tuple(nwb_assets))
    }
    if take_newest:
        session_assets["session_assets"] = str(session_asset_ids[-1]) if session_asset_ids else ""
        session_assets["processed_assets"] = str(processed_asset_ids[-1]) if processed_asset_ids else ""
        session_assets["nwb_assets"] = str(nwb_asset_ids[-1]) if nwb_asset_ids else ""
    return session_assets


def main():
    parser = argparse.ArgumentParser(description="Convert experimentsummary.mat to experiment_summary.h5.")
    # parser.add_argument('mouse_ids', nargs='+', help='one or more session IDs for slap2 sessions (mouse_date)')
    parser.add_argument("--mouse_ids_csv", type=str, required=True, help="CSV file containing mouse IDs")
    parser.add_argument("--take_newest", type=bool, default=False, help="Take the most recent asset for each asset type")
    parser.add_argument("--output_csv_name", type=str, default="co_session_assets", help="Output CSV file name (without path)")
    args = parser.parse_args()
    docdb_api_client = MetadataDbClient(
        host=API_GATEWAY_HOST,
        database=DATABASE,
        collection=COLLECTION,
    )
    mouse_ids_csv = Path(args.mouse_ids_csv)
    if not mouse_ids_csv.is_absolute():
        mouse_ids_csvs = THIS_FILE_PATH.parent.parent.rglob(str(mouse_ids_csv))
        mouse_ids_csv = next(mouse_ids_csvs)
    if len(list(mouse_ids_csvs)) > 1:
        raise ValueError(f"Multiple files found for {mouse_ids_csv}")
    
    mouse_ids_csv = pd.read_csv(mouse_ids_csv)
    all_mouse_ids = [str(mid) for mid in mouse_ids_csv['mouse_id']]
    mouse_assets = find_co_assets_by_mouse_ids(all_mouse_ids)

    # print([asset.name for asset in mouse_assets[all_mouse_ids[0]]])

    df_rows = []
    session_ids = set()
    for i, row in mouse_ids_csv.iterrows():
        mouse_id = str(row['mouse_id'])
        # print(i, mouse_id)
        session_id = str(row['sessionid'])
        if session_id in session_ids:
            continue
        session_ids.add(session_id)
        datetime_string = datetime.strptime(row['date_of_acquisition'], "%m/%d/%Y").strftime("%Y-%m-%d")
        df_rows.append(get_session_asset_row(mouse_id, session_id, datetime_string, mouse_assets[mouse_id], take_newest=args.take_newest))
    sessions_df = pd.DataFrame(df_rows)
    output_csv = THIS_FILE_PATH.parent.parent / "data" / f"{args.output_csv_name}.csv"
    sessions_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
