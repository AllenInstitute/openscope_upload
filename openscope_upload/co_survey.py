
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
    if session_id is None:
        session_id = f"{mouse_id}_{session_datetime_str}"
    
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



def survey_co_assets(session_ids, take_newest=True, output_csv_name=None):
    """
    Survey Code Ocean assets for sessions in the provided dataframe.
    
    Parameters
    ----------
    session_ids : pd.DataFrame
        DataFrame with columns: 'mouse_id', 'sessionid', 'date_of_acquisition' (MM/DD/YYYY format)
    take_newest : bool, default=True
        If True, only return the most recent asset for each asset type
    output_csv_name : str, optional
        If provided, writes results to CSV in data/ directory with this name
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: mouse_id, session_id, session_datetime, session_assets, processed_assets, nwb_assets
    """
    all_mouse_ids = [str(mid) for mid in session_ids['mouse_id']]
    mouse_assets = find_co_assets_by_mouse_ids(all_mouse_ids)

    df_rows = []
    session_ids_seen = set()
    for i, row in session_ids.iterrows():
        mouse_id = str(row['mouse_id'])
        session_id = str(row['sessionid'])
        if session_id in session_ids_seen:
            continue
        session_ids_seen.add(session_id)
        datetime_string = datetime.strptime(row['date_of_acquisition'], "%m/%d/%Y").strftime("%Y-%m-%d")
        df_rows.append(get_session_asset_row(mouse_id, session_id, datetime_string, mouse_assets[mouse_id], take_newest=take_newest))
    sessions_df = pd.DataFrame(df_rows)
    
    # Only write CSV if output_csv_name is provided
    if output_csv_name:
        output_csv = THIS_FILE_PATH.parent.parent / "data" / f"{output_csv_name}.csv"
        sessions_df.to_csv(output_csv, index=False)
    
    return sessions_df


def main():
    parser = argparse.ArgumentParser(description="Convert experimentsummary.mat to experiment_summary.h5.")
    parser.add_argument('session_ids', nargs='*', help='one or more session IDs in format sessionid_mouseid_datetime (e.g., 1488336340_830794_20260126)')
    parser.add_argument("--session_ids_csv", type=str, help="CSV file containing session IDs")
    parser.add_argument("--take_newest", type=bool, default=True, help="Take the most recent asset for each asset type")
    parser.add_argument("--output_csv_name", type=str, default=None, help="Output CSV file name (without path). If not provided, returns dataframe without writing to file.")
    args = parser.parse_args()
    
    # Validate that either session_ids or session_ids_csv is provided
    if not args.session_ids and not args.session_ids_csv:
        parser.error("Either provide session IDs as arguments or use --session_ids_csv")
    if args.session_ids and args.session_ids_csv:
        parser.error("Cannot use both session ID arguments and --session_ids_csv at the same time")
    
    # Get mouse IDs either from command line session IDs or CSV
    if args.session_ids_csv:
        session_ids_path = Path(args.session_ids_csv)
        if not session_ids_path.is_absolute():
            session_idss = THIS_FILE_PATH.parent.parent.rglob(str(session_ids_path))
            session_ids_path = next(session_idss)
        if len(list(session_idss)) > 1:
            raise ValueError(f"Multiple files found for {session_ids_path}")
        
        session_ids = pd.read_csv(session_ids_path)
    else:
        # Parse session IDs in format mouseid_datetimeT or sessionid_mouseid_datetime
        parsed_sessions = []
        for session_id in args.session_ids:
            parts = session_id.split('_')
            
            if len(parts) == 2:
                # Format: mouseid_datetimeT (e.g., 776270_20250702T115006)
                mouse_id, datetime_str = parts
                if 'T' in datetime_str:
                    # Parse datetime with time component
                    try:
                        dt = datetime.strptime(datetime_str, "%Y%m%dT%H%M%S")
                        date_formatted = dt.strftime("%m/%d/%Y")
                    except ValueError:
                        raise ValueError(f"Invalid datetime format in session ID: {session_id}. Expected YYYYMMDDTHHMMSS")
                else:
                    raise ValueError(f"Invalid session ID format: {session_id}. Expected mouseid_datetimeT")
            elif len(parts) == 3:
                # Format: sessionid_mouseid_datetime (e.g., 1488336340_830794_20260126)
                _, mouse_id, datetime_str = parts
                try:
                    dt = datetime.strptime(datetime_str, "%Y%m%d")
                    date_formatted = dt.strftime("%m/%d/%Y")
                except ValueError:
                    raise ValueError(f"Invalid datetime format in session ID: {session_id}. Expected YYYYMMDD")
            else:
                raise ValueError(f"Invalid session ID format: {session_id}. Expected mouseid_datetimeT or sessionid_mouseid_datetime")
            
            parsed_sessions.append({
                'sessionid': session_id,  # Full session ID
                'mouse_id': mouse_id,
                'date_of_acquisition': date_formatted
            })
        
        session_ids = pd.DataFrame(parsed_sessions)
    
    # Call the core function with the prepared dataframe
    return survey_co_assets(session_ids, take_newest=args.take_newest, output_csv_name=args.output_csv_name)


if __name__ == "__main__":
    main()
