import harp
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import requests, yaml, io
import scipy.io
import h5py
import argparse
from scbc.slap2.experiment_summary import ExperimentSummary

def _get_yml_from_who_am_i(who_am_i: int, release: str = "main") -> io.BytesIO:
    try:
        device = _get_who_am_i_list()[who_am_i]
    except KeyError as e:
        raise KeyError(f"WhoAmI {who_am_i} not found in whoami.yml") from e

    repository_url = device.get("repositoryUrl", None)

    if repository_url is None:
        raise ValueError("Device's repositoryUrl not found in whoami.yml")
    else:  # attempt to get the device.yml from the repository
        _repo_hint_paths = [
            "{repository_url}/{release}/device.yml",
            "{repository_url}/{release}/software/bonsai/device.yml",
        ]

        yml = None
        for hint in _repo_hint_paths:
            url = hint.format(repository_url=repository_url, release=release)
            if "github.com" in url:
                url = url.replace("github.com", "raw.githubusercontent.com")
            response = requests.get(url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                yml = io.BytesIO(response.content)
                break
        if yml is None:
            raise FileNotFoundError("device.yml not found in any repository")
        else:
            return yml

def _get_who_am_i_list(url: str = "https://raw.githubusercontent.com/harp-tech/protocol/main/whoami.yml"):
    response = requests.get(url, allow_redirects=True, timeout=5)
    content = response.content.decode("utf-8")
    content = yaml.safe_load(content)
    devices = content["devices"]
    return devices


def fetch_yml(harp_path):
    with open(harp_path / 'Behavior_0.bin',mode='rb') as reg_0:
        who_am_i = int(harp.read(reg_0).values[0][0])
        yml_bytes = _get_yml_from_who_am_i(who_am_i)
    yaml_content = yml_bytes.getvalue()
    with open(harp_path / "device.yml", "wb") as f:
        f.write(yaml_content)
    return harp_path / "device.yml"


def extract_harp(harp_path):
    """
    Extract all relevant arrays from a HARP folder (timing and data arrays).
    Returns a dict with all read arrays as values, using clear names and normalized time arrays.
    """
    harp_path = Path(harp_path)
    if not (harp_path / "device.yml").exists():
        fetch_yml(harp_path)
    reader = harp.create_reader(harp_path)
    analog_data = reader.AnalogData.read()
    analog_times = analog_data.index.to_numpy()
    photodiode = analog_data["AnalogInput0"].to_numpy()
    wheel = analog_data["Encoder"].to_numpy()
    slap2_start_signal = reader.PulseDO0.read()["PulseDO0"].to_numpy()
    slap2_start_times = reader.PulseDO0.read()["PulseDO0"].index.to_numpy()
    slap2_end_signal = reader.PulseDO1.read()["PulseDO1"].to_numpy()
    slap2_end_times = reader.PulseDO1.read().index.to_numpy()
    grating_signal = reader.PulseDO2.read()["PulseDO2"].to_numpy()
    grating_times = reader.PulseDO2.read()["PulseDO2"].index.to_numpy()

    # Set the time reference (time_0)
    time_reference = slap2_start_times[0]
    # Calculate normalized times
    normalized_start_gratings = grating_times - time_reference
    normalized_slap2_start = slap2_start_times - time_reference
    normalized_slap2_end = slap2_end_times - time_reference
    normalized_analog_times = analog_times - time_reference

    return {
        "analog_times": analog_times,
        "photodiode": photodiode,
        "wheel": wheel,
        "slap2_start_signal": slap2_start_signal,
        "slap2_start_times": slap2_start_times,
        "slap2_end_signal": slap2_end_signal,
        "slap2_end_times": slap2_end_times,
        "grating_signal": grating_signal,
        "grating_times": grating_times,
        "time_reference": time_reference,
        "normalized_start_gratings": normalized_start_gratings,
        "normalized_slap2_start": normalized_slap2_start,
        "normalized_slap2_end": normalized_slap2_end,
        "normalized_analog_times": normalized_analog_times
    }


def get_concatenated_traces(exp, dmd, trace_type1, trace_type2, harp_data):
    """
    Concatenate traces and timestamps for all valid trials for a given DMD and trace type.
    Returns (all_traces, all_timestamps)
    """
    dmd_idx = dmd-1
    all_timestamps = None
    all_traces = None
    trial_start_idxs = []
    discarded_frames = []
    for trial in exp.valid_trials[dmd_idx]:
        trial_idx = trial - 1  # 1-indexed to 0-indexed
        traces = exp.get_traces(dmd, trial, trace_type1=trace_type1, trace_type2=trace_type2)
        if traces.ndim > 2:
            traces = traces[0]
        start_trial = harp_data['slap2_start_times'][trial_idx]
        end_trial = harp_data['slap2_end_times'][trial_idx]
        num_timepoints = traces.shape[1]
        timestamps = np.linspace(start_trial, end_trial, num_timepoints)
        if all_traces is None or all_timestamps is None:
            all_traces = traces.T
            all_timestamps = timestamps
        else:
            all_traces = np.concatenate((all_traces, traces.T))
            all_timestamps = np.concatenate((all_timestamps, timestamps))

        # record the start index of each trial
        if len(timestamps) > 0:
            trial_start_idxs.append(len(all_timestamps))
        # record whether each frame was 'invalid'
        if trial in exp.valid_trials[dmd_idx]:
            discarded_frames.extend([False] * len(timestamps))
        else:
            discarded_frames.extend([True] * len(timestamps))

    return all_traces, all_timestamps, np.array(trial_start_idxs), np.array(discarded_frames, dtype=bool)


def expsum_mat_to_h5(mat_path, h5_path, harp_path):
    """
    Convert ophys-SCBC-analysis experimentsummary.mat to experiment_summary.h5 with the required structure using ExperimentSummary class.
    Always extracts all HARP arrays and uses timing data for dF, dFF, F0 traces.
    """
    import h5py
    import numpy as np
    expsum = ExperimentSummary(mat_path)
    harp_data = extract_harp(harp_path)
    with h5py.File(h5_path, "w") as h5:
        # Save all harp arrays at root for reference
        # harp_grp = h5.create_group("harp")
        # for k, v in harp_data.items():
        #     harp_grp.create_dataset(k, data=v)
        for dmd_idx in range(expsum.n_dmds):
            dmd = dmd_idx+1
            dmd_grp = h5.create_group(f"DMD{dmd}")
            # visualizations
            vis_grp = dmd_grp.create_group("visualizations")
            vis_grp.create_dataset("mean_im", data=expsum.get_summary_image(dmd, 'meanIM'))
            vis_grp.create_dataset("act_im", data=expsum.get_summary_image(dmd, 'actIM'))
            # vis_grp.create_dataset("per_trial_mean_im", data=np.zeros((expsum.n_trials, 1, 1)))
            # vis_grp.create_dataset("per_trial_act_im", data=np.zeros((expsum.n_trials, 1, 1)))
            # global
            # global_grp = dmd_grp.create_group("global")
            # global_grp.create_dataset("F", data=np.zeros((n_frames, 1)))
            
            # user rois
            add_rois_group(dmd_grp, expsum, dmd)

            # sources
            sources_grp = dmd_grp.create_group("sources")

            spatial_grp = sources_grp.create_group("spatial")
            trial_to_use = expsum.valid_trials[dmd_idx][0]
            fp_masks, fp_coords = expsum.get_footprints(dmd=dmd, trial=trial_to_use)
            spatial_grp.create_dataset("fp_masks",data=fp_masks,compression="gzip")
            spatial_grp.create_dataset("fp_coords",data=fp_coords,compression="gzip")

            # temporal group
            temporal_grp = sources_grp.create_group("temporal")
            dF_traces, _, _, _ = get_concatenated_traces(expsum, dmd, 'dF', 'denoised', harp_data)
            dFF_traces, _, _, _ = get_concatenated_traces(expsum, dmd, 'dFF', 'denoised', harp_data)
            F0_traces, _, trial_start_idxs, discarded_frames = get_concatenated_traces(expsum, dmd, 'F0', None, harp_data)
            temporal_grp.create_dataset("dF", data=dF_traces)
            temporal_grp.create_dataset("dFF", data=dFF_traces)
            temporal_grp.create_dataset("F0", data=F0_traces)

            # frame_info
            fi_grp = dmd_grp.create_group("frame_info")
            fi_grp.create_dataset("trial_start_idxs", data=trial_start_idxs)
            fi_grp.create_dataset("discard_frames", data=discarded_frames)


def add_rois_group(dmd_grp, expsum, dmd):
    dmd_idx = dmd-1
    rois_grp = dmd_grp.create_group("user_rois")

    user_rois = expsum.get_user_rois_info()
    if user_rois['masks'][dmd_idx]:
        masks = np.stack(user_rois['masks'][dmd_idx], axis=-1)
    else:
        masks = np.zeros((1,1,1))
    rois_grp.create_dataset("mask", data=masks)

    F_list = []
    Fsvd_list = []
    for trial in expsum.valid_trials[dmd_idx]:
        F = expsum.get_user_roi_traces(dmd, trial)
        F_list.append(F)
        Fsvd = expsum.get_user_roi_traces(dmd, trial, trace_type='Fsvd')
        Fsvd_list.append(Fsvd)

    F_concat = np.concatenate(F_list, axis=2)
    Fsvd_concat = np.concatenate(Fsvd_list, axis=2)
    rois_grp.create_dataset("F", data=F_concat)
    rois_grp.create_dataset("Fsvd", data=Fsvd_concat)


def main():
    parser = argparse.ArgumentParser(description="Convert experimentsummary.mat to experiment_summary.h5.")
    parser.add_argument("session_path", type=str, help="Path to session directory (will search for *Summary*.mat and *.harp)")
    parser.add_argument("h5_path", type=str, help="Path to output experiment_summary.h5 file")
    args = parser.parse_args()
    session_path = Path(args.session_path)
    # Find summary .mat file
    mat_matches = list(session_path.rglob("*Summary*.mat"))
    if not mat_matches:
        raise FileNotFoundError(f"No *Summary*.mat file found in {session_path}")
    mat_path = mat_matches[0]
    print(f"Found summary .mat file: {mat_path}")
    # Find .harp folder
    harp_matches = list(session_path.rglob("*.harp"))
    if not harp_matches:
        raise FileNotFoundError(f"No *.harp folder found in {session_path}")
    harp_path = harp_matches[0]
    print(f"Found .harp folder: {harp_path}")
    expsum_mat_to_h5(mat_path, args.h5_path, harp_path)

if __name__ == "__main__":
    main()

