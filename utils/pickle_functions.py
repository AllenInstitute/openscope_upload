import pandas as pd
import numpy as np

import pickle


def load_pkl(path):
    data = pd.read_pickle(path)
    return data

def load_img_pkl(pstream):
    return pickle.load(pstream, encoding="bytes")

def get_stimuli(pkl):
    return pkl['stimuli']


def get_fps(pkl):
    return pkl['fps']


def get_pre_blank_sec(pkl):
    return pkl['pre_blank_sec']


def angular_wheel_velocity(pkl):
    return get_fps(pkl) * get_angular_wheel_rotation(pkl)


def get_angular_wheel_rotation(pkl):
    return get_running_array(pkl, "dx")


def vsig(pkl):
    return get_running_array(pkl, "vsig")


def vin(pkl):
    return get_running_array(pkl, "vin")


def get_running_array(pkl, key):
    try:
        result = pkl['items']['foraging']['encoders'][0][key]
    except (KeyError, IndexError):
        try:
            result = pkl[key]
        except KeyError:
            raise KeyError(f'unable to extract {key} from this stimulus pickle')
            
    return np.array(result)