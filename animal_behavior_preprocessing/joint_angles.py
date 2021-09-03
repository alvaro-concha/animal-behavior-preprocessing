"""Computes joint angles and their statistics.

Computes new X-Y coords, with and additional mid-bottom-rotarod marker.
Computes joint angles defined by some select thruples of markers.
Computes number of frames, means and variances of joint angles.
"""
import numpy as np
from utilities import read_pickle, write_pickle
import config


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


def get_extended_xys(xys):
    """
    Returns a new X-Y coords, with and additional mid-bottom-rotarod marker.

    Parameters
    ----------
    xys : ndarray
        Combined X-Y coordinates from both cameras in same perspective

    Returns
    -------
    extended_xys : ndarray
        Extended X-Y coordinates with mid-bottom-rotarod marker
    """
    extended_xys = np.zeros((xys.shape[0], xys.shape[1] + 1, xys.shape[2]))
    extended_xys[:, : xys.shape[1], :] = xys
    extended_xys[:, xys.shape[1], :] = [config.ROTAROD_HEIGHT / 2.0, 0.0]
    return extended_xys


def get_joint_angles(xys):
    """
    Returns joint angles.

    Parameters
    ----------
    xys : ndarray
        Combined X-Y coordinates from both cameras in same perspective

    Returns
    -------
    angs : ndarray
        Array of joint angles
    """
    extended_xys = get_extended_xys(xys)
    angs = []
    for idx in config.ANGLE_MARKER_IDX:
        angs.append(
            get_single_joint_angle(
                extended_xys[:, idx[0], :],
                extended_xys[:, idx[1], :],
                extended_xys[:, idx[2], :],
            )
        )
    angs = np.array(angs).T
    angs += np.pi - np.median(angs, axis=0)
    angs %= 2 * np.pi
    return angs


def get_statistics(angs):
    """
    Returns number of frames, means and variances for each single joint angle.

    Parameters
    ----------
    angs : ndarray
        Array of joint angles

    Returns
    -------
    num_frames, mean, var : tuple(int, ndarray, ndarray)
        Number of frames. Means and variances for each single joint angle
    """
    num_frames = len(angs)
    mean = angs.mean(axis=0)
    var = angs.var(axis=0)
    return num_frames, mean, var


def get_joint_angles_statistics(dep_pickle_path, target_pickle_paths):
    """
    Computes joint angles.
    Computes number of frames, means and variances for each single joint angle.
    Saves results.

    Parameters
    ----------
    dep_pickle_path : pathlib.Path
        Path of dependency pickle file to read
    target_pickle_paths : dict of pathlib.Path
        Paths of target pickle files to save
    """
    xys = read_pickle(dep_pickle_path)
    angs = get_joint_angles(xys)
    stats = get_statistics(angs)
    write_pickle(angs, target_pickle_paths["ang"])
    write_pickle(stats, target_pickle_paths["stat"])


def get_global_statistics(dep_pickle_paths, target_pickle_path):
    """
    Computes joint angles global statistics.
    That is, total number of frames, global mean and variance.
    Saves results.

    Parameters
    ----------
    dep_pickle_paths : dict of pathlib.Path
        Paths of dependency pickle files to read
    target_pickle_path : pathlib.Path
        Path of target pickle file to save
    """
    num_frames_list, mean_list, var_list = [], [], []
    for file in dep_pickle_paths:
        num_frames, mean, var = read_pickle(file)
        num_frames_list.append(num_frames)
        mean_list.append(mean)
        var_list.append(var)
    num_frames_list = np.array(num_frames_list)[:, np.newaxis]
    mean_list = np.array(mean_list)
    var_list = np.array(var_list)
    num_frames_total = num_frames_list.sum()
    mean_global = (mean_list * num_frames_list).sum(axis=0) / num_frames_total
    var_global = (
        (num_frames_list - 1) * (var_list ** 2)
        + num_frames_list * ((mean_list - mean_global) ** 2)
    ).sum(axis=0) / (num_frames_total - 1)
    stats_global = (num_frames_total, mean_global, var_global)
    write_pickle(stats_global, target_pickle_path)
