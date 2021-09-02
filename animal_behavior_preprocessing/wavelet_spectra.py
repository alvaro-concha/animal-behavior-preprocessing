"""Computes wavelet spectra of whitened joint angles."""
import numpy as np
from utilities import read_pickle, write_pickle
import config


learning_angles = {}
for key in learning_xys.keys():
    angles = get_joint_angles(learning_xys[key], angle_marker_idx)
    angles -= np.mean(angles, axis=0)
    angles /= np.std(angles, axis=0)
    learning_angles[key] = angles


def get_whitened_angs(angs, stats):
    pass


def get_single_wavelet_spectrum(whitened_angs):
    pass


def get_wavelet_spectra(dep_pickle_paths, target_pickle_path):
    """
    Computes wavelet spectra of whitened joint angles.

    Parameters
    ----------
    dep_pickle_paths : dict of pathlib.Path
        Paths of dependency pickle files to read
    target_pickle_path : pathlib.Path
        Path of target pickle file to save
    """
    angs = read_pickle(dep_pickle_paths["ang"])
    stats = read_pickle(dep_pickle_paths["stat"])
    whitened_angs = get_whitened_angs(angs, stats)
    wavs = get_single_wavelet_spectrum(whitened_angs)
    write_pickle(wavs, target_pickle_path)
