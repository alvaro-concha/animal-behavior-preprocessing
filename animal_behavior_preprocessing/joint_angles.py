"""Computes joint angles and their statistics.

Computes joint angles defined by predefined 3-tuples of markers.
Trimmed mean smoother is applied to the angles.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from utilities import read_pickle, write_pickle
import config
from quantile_filter_smoother import get_window_trim_mean


def get_single_joint_angle(marker_a, marker_b, marker_c):
    """
    Returns single continuous angles between markers a, b and c.

    Parameters
    ----------
    marker_{a, b, c} : ndarray
        Markers a, b, and c

    Returns
    -------
    ndarray
        Angle defined by markers abc
    """
    vector_ba = marker_a - marker_b
    vector_bc = marker_c - marker_b
    return np.unwrap(
        np.arctan2(vector_bc[:, 1], vector_bc[:, 0])
        - np.arctan2(vector_ba[:, 1], vector_ba[:, 0])
    )


def get_joint_angles(dep_pickle_paths, target_pickle_path):
    """
    Computes and saves joint angles, up to latency to fall time.
    Trimmed mean smoother is applied to the angles.

    Parameters
    ----------
    dep_pickle_paths : pathlib.Path
        Paths of dependency pickle files to read
    target_pickle_path : dict of pathlib.Path
        Path of target pickle file to save
    """
    latency = read_pickle(dep_pickle_paths["latency"])
    xys = read_pickle(dep_pickle_paths["xys"])[:latency]
    angs = []
    for idx in config.ANG_MARKER_IDX:
        angs.append(
            get_single_joint_angle(
                xys[:, idx[0], :],
                xys[:, idx[1], :],
                xys[:, idx[2], :],
            )
        )
    angs = np.array(angs).T
    angs += np.pi - np.median(angs, axis=0)
    angs %= 2 * np.pi
    angs = np.apply_along_axis(
        get_window_trim_mean,
        0,
        angs,
        window=config.ANG_TRIM_WIN,
        cut=config.ANG_TRIM_CUT,
    )
    write_pickle(angs, target_pickle_path)


def get_standard_scaler_fit_joint_angles(dep_pickle_paths, target_pickle_path):
    """Fit standard scaler (aka whitening or z-score), to joint angles."""
    scaler_angs = StandardScaler()
    for path in dep_pickle_paths["ang"]:
        scaler_angs.partial_fit(read_pickle(path))
    del dep_pickle_paths["ang"]
    write_pickle(scaler_angs, target_pickle_path)
